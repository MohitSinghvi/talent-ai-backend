import pandas as pd
import numpy as np
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from xgboost import XGBClassifier
from api_routes import nl_chat

app = FastAPI()
app.include_router(nl_chat.router)

# Allow CORS so React frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop irrelevant column if it exists
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

# Label Encoding for categorical columns
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

# -----------------------------------------
# Splitting the Data
# -----------------------------------------
# Separate employees who left vs. stayed
attrition_positive = data[data['Attrition'] == 1]
attrition_negative = data[data['Attrition'] == 0]

# Train-test split (70% for training, 30% for testing)
train_size = int(len(attrition_negative) * 0.7)
test_size = int(len(attrition_negative) * 0.3)

# Training data: All who left + 70% of those who stayed
train_data = pd.concat([attrition_positive, attrition_negative.iloc[:train_size]])

# Testing data: 30% of those who stayed + a subset of those who left
test_data = pd.concat([attrition_positive.iloc[:test_size], attrition_negative.iloc[train_size:]])

# Prediction data (remaining employees from attrition_negative used for hiring predictions)
predict_data = attrition_negative.iloc[train_size:].copy()

# Define feature columns (exclude EmployeeNumber and non-relevant columns)
exclude_features = [
    'Attrition',
    'EmployeeNumber',
    'OriginalEmployeeNumber',
    'OriginalGender',
    'OriginalDepartment',
    'OriginalAge'
]
X_train = train_data.drop(columns=exclude_features)
y_train = train_data['Attrition']
X_test = test_data.drop(columns=exclude_features)
y_test = test_data['Attrition']
X_predict = predict_data.drop(columns=exclude_features)

# -----------------------------------------
# Training the XGBoost Model
# -----------------------------------------
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate feature importances
feature_columns = X_train.columns.tolist()
feature_importances = pd.Series(model.feature_importances_, index=feature_columns)

# -----------------------------------------
# Model Evaluation
# -----------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# -----------------------------------------
# Predicting Attrition Risk for Hiring
# -----------------------------------------
predict_data.loc[:, 'Attrition_Risk'] = model.predict_proba(X_predict)[:, 1]

# Restore original columns
predict_data.loc[:, 'OriginalEmployeeNumber'] = data.loc[predict_data.index, 'OriginalEmployeeNumber']
predict_data.loc[:, 'OriginalGender'] = data.loc[predict_data.index, 'OriginalGender']
predict_data.loc[:, 'OriginalDepartment'] = data.loc[predict_data.index, 'OriginalDepartment']
predict_data.loc[:, 'OriginalAge'] = data.loc[predict_data.index, 'OriginalAge']

# Define age bins for analysis
bins = [18, 25, 35, 45, 55, 65]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
predict_data.loc[:, 'Age_Range'] = pd.cut(predict_data['OriginalAge'], bins=bins, labels=labels)

# -----------------------------------------
# API Endpoints
# -----------------------------------------
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
        # Use model-trained features only
        employee_features = X_predict.loc[index, feature_columns]

        # Calculate absolute contributions based on feature importance
        abs_contributions = employee_features * feature_importances
        # Find the most impactful feature
        top_reason = abs_contributions.idxmax()

        reasons.append({
            "OriginalEmployeeNumber": row["OriginalEmployeeNumber"],
            "Attrition_Risk_Percentage": round(row["Attrition_Risk"] * 100, 2),
            "Top_Contributing_Factor": top_reason,
            "Department": row["OriginalDepartment"]
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

# -----------------------------------------
# NEW: Predictive Hiring Endpoint
# -----------------------------------------
@app.get("/predictive_hiring")
def predictive_hiring():
    # Define a risk threshold; here, employees with a risk above 0.5 are predicted to leave.
    threshold = 0.5
    # Count the number of employees in the prediction set with risk above the threshold.
    num_employees_at_risk = predict_data[predict_data['Attrition_Risk'] > threshold].shape[0]
    
    # For simplicity, we assume that the number of hires needed equals the number of at-risk employees.
    number_of_hires = num_employees_at_risk
    
    # Dummy talent pool; in practice, this data would come from your HR system.
    talent_pool = [
        {"CandidateID": 101, "Name": "Alice Smith", "Experience": "5 years", "Skill": "Sales"},
        {"CandidateID": 102, "Name": "Bob Johnson", "Experience": "3 years", "Skill": "Marketing"},
        {"CandidateID": 103, "Name": "Carol Davis", "Experience": "6 years", "Skill": "Engineering"},
        {"CandidateID": 104, "Name": "David Wilson", "Experience": "4 years", "Skill": "Customer Service"},
        {"CandidateID": 105, "Name": "Eva Brown", "Experience": "7 years", "Skill": "HR"}
    ]

    # If the number of hires needed is more than the available candidates, return the full talent pool.
    if number_of_hires > len(talent_pool):
        suggested_candidates = talent_pool
    else:
        # Randomly select candidates from the talent pool as suggestions.
        suggested_candidates = random.sample(talent_pool, number_of_hires)
        
    return {
        "number_of_hires": number_of_hires,
        "suggested_candidates": suggested_candidates
    }