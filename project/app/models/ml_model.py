import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Tuple, Dict

class AttritionModel:
    def __init__(self, data: pd.DataFrame):
        self.model = XGBClassifier(random_state=42)
        self.feature_columns = None
        self.feature_importances = None
        self.metrics = {}
        self.predict_data = None
        self._train_model(data)

    def _train_model(self, data: pd.DataFrame):
        """Train the XGBoost model."""
        # Split data
        attrition_positive = data[data['Attrition'] == 1]
        attrition_negative = data[data['Attrition'] == 0]
        train_size = int(len(attrition_negative) * 0.7)
        test_size = int(len(attrition_negative) * 0.3)

        train_data = pd.concat([attrition_positive, attrition_negative.iloc[:train_size]])
        test_data = pd.concat([attrition_positive.iloc[:test_size], attrition_negative.iloc[train_size:]])
        self.predict_data = attrition_negative.iloc[train_size:].copy()

        exclude_features = [
            'Attrition', 'EmployeeNumber', 'OriginalEmployeeNumber',
            'OriginalGender', 'OriginalDepartment', 'OriginalAge'
        ]
        self.feature_columns = [col for col in data.columns if col not in exclude_features]

        X_train = train_data[self.feature_columns]
        y_train = train_data['Attrition']
        X_test = test_data[self.feature_columns]
        y_test = test_data['Attrition']
        X_predict = self.predict_data[self.feature_columns]

        # Train model
        self.model.fit(X_train, y_train)

        # Feature importances
        self.feature_importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        # Predict attrition risk
        self.predict_data['Attrition_Risk'] = self.model.predict_proba(X_predict)[:, 1]
        self.predict_data['OriginalEmployeeNumber'] = data.loc[self.predict_data.index, 'OriginalEmployeeNumber']
        self.predict_data['OriginalGender'] = data.loc[self.predict_data.index, 'OriginalGender']
        self.predict_data['OriginalDepartment'] = data.loc[self.predict_data.index, 'OriginalDepartment']
        self.predict_data['OriginalAge'] = data.loc[self.predict_data.index, 'OriginalAge']

        bins = [18, 25, 35, 45, 55, 65]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
        self.predict_data['Age_Range'] = pd.cut(self.predict_data['OriginalAge'], bins=bins, labels=labels)

    def get_predict_data(self) -> pd.DataFrame:
        """Return prediction data."""
        return self.predict_data

    def get_metrics(self) -> Dict:
        """Return model evaluation metrics."""
        return self.metrics

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances."""
        return self.feature_importances

    def get_feature_columns(self) -> list:
        """Return feature columns."""
        return self.feature_columns