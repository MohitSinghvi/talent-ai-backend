import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

app = FastAPI()

# Allow CORS so React frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess data
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop irrelevant column
if 'Over18' in data.columns:
    data.drop(columns=['Over18'], inplace=True)

# Store original categorical data for later use
data['OriginalEmployeeNumber'] = data['EmployeeNumber']
data['OriginalGender'] = data['Gender']
data['OriginalDepartment'] = data['Department']
data['OriginalAge'] = data['Age']

# Exclude original columns from encoding
categorical_columns = data.select_dtypes(include='object').columns.tolist()
exclude_from_encoding = ['OriginalEmployeeNumber', 'OriginalGender', 'OriginalDepartment', 'OriginalAge']
categorical_columns = [col for col in categorical_columns if col not in exclude_from_encoding]

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Identify numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Attrition', 'EmployeeNumber', 'OriginalEmployeeNumber', 'OriginalGender', 'OriginalDepartment', 'OriginalAge']
numeric_to_scale = [c for c in numeric_cols if c not in exclude_cols]

# Scale numeric features
scaler = StandardScaler()
data[numeric_to_scale] = scaler.fit_transform(data[numeric_to_scale])

# Split data into employees who left vs those who stayed
attrition_positive = data[data['Attrition'] == 1]
attrition_negative = data[data['Attrition'] == 0]

# Train-test split (70% for training, 30% for testing)
train_size = int(len(attrition_negative) * 0.7)
test_size = int(len(attrition_negative) * 0.3)

# Training data
train_data = pd.concat([attrition_positive, attrition_negative.iloc[:train_size]])

# Testing data
test_data = pd.concat([attrition_positive.iloc[:test_size], attrition_negative.iloc[train_size:]])

# Prediction data (remaining employees)
predict_data = attrition_negative.iloc[train_size:].copy()

# Define feature columns (exclude EmployeeNumber and non-relevant columns)
exclude_features = ['Attrition', 'EmployeeNumber', 'OriginalEmployeeNumber', 'OriginalGender', 'OriginalDepartment', 'OriginalAge']
X_train = train_data.drop(columns=exclude_features)
y_train = train_data['Attrition']
X_test = test_data.drop(columns=exclude_features)
y_test = test_data['Attrition']
X_predict = predict_data.drop(columns=exclude_features)

# Train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Store feature importance
feature_columns = X_train.columns.tolist()
feature_importances = pd.Series(model.feature_importances_, index=feature_columns)

# Model Evaluation Metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# Predict attrition risk
predict_data.loc[:, 'Attrition_Risk'] = model.predict_proba(X_predict)[:, 1]

# Restore original employee details
predict_data.loc[:, 'OriginalEmployeeNumber'] = data.loc[predict_data.index, 'OriginalEmployeeNumber']
predict_data.loc[:, 'OriginalGender'] = data.loc[predict_data.index, 'OriginalGender']
predict_data.loc[:, 'OriginalDepartment'] = data.loc[predict_data.index, 'OriginalDepartment']
predict_data.loc[:, 'OriginalAge'] = data.loc[predict_data.index, 'OriginalAge']

# Define age bins for analysis
bins = [18, 25, 35, 45, 55, 65]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
predict_data.loc[:, 'Age_Range'] = pd.cut(predict_data['OriginalAge'], bins=bins, labels=labels)

# API Endpoints

@app.get("/")
def read_root():
    return {"message": "Welcome to the Attrition Prediction API! Use /docs for API documentation."}

@app.get("/risk_curve_data")
def get_risk_curve_data():
    grouped = predict_data.groupby(['Age_Range', 'OriginalGender']).agg({
        'Attrition_Risk': 'mean',
        'OriginalEmployeeNumber': 'count'
    }).reset_index()
    grouped['Attrition_Percentage'] = grouped['Attrition_Risk'] * 100
    return grouped.to_dict(orient='records')

@app.get("/top_employees_data")
def get_top_employees_data():
    top_employees = predict_data.nlargest(10, 'Attrition_Risk').copy()

    reasons = []
    for index, row in top_employees.iterrows():
        # Ensure we only use model-trained features
        employee_features = X_predict.loc[index, feature_columns]

        # Calculate absolute contributions based on feature importance
        abs_contributions = employee_features * feature_importances

        # Get the most impactful feature
        top_reason = abs_contributions.idxmax()

        reasons.append({
            "OriginalEmployeeNumber": row["OriginalEmployeeNumber"],
            "Attrition_Risk_Percentage": round(row["Attrition_Risk"] * 100, 2),
            "Top_Contributing_Factor": top_reason  # The actual contributing feature
        })

    return reasons

@app.get("/department_pie_data")
def get_department_pie_data():
    risky_employees = predict_data[predict_data['Attrition_Risk'] > 0.5]
    department_counts = risky_employees['OriginalDepartment'].value_counts(normalize=True) * 100
    department_df = department_counts.reset_index()
    department_df.columns = ['Department', 'Percentage']
    return department_df.to_dict(orient='records')

@app.get("/model_evaluation")
def get_model_evaluation():
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC Score": round(auc_score, 4),
        "Confusion Matrix": conf_matrix.tolist()
    }