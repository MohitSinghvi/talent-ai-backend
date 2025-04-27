import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict
from app.config import settings

class DataLoader:
    def __init__(self):
        self.hr_data = None
        self.skills_data = None
        self.applicants_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._load_data()

    def _load_data(self):
        """Load and preprocess all datasets."""
        try:
            # Load HR data
            self.hr_data = pd.read_csv(f"{settings.DATA_DIR}/{settings.EMPLOYEE_CSV}")
            self.hr_data['Attrition'] = self.hr_data['Attrition'].map({'Yes': 1, 'No': 0})
            if 'Over18' in self.hr_data.columns:
                self.hr_data.drop(columns=['Over18'], inplace=True)

            # Store original columns
            self.hr_data['OriginalEmployeeNumber'] = self.hr_data['EmployeeNumber']
            self.hr_data['OriginalGender'] = self.hr_data['Gender']
            self.hr_data['OriginalDepartment'] = self.hr_data['Department']
            self.hr_data['OriginalAge'] = self.hr_data['Age']

            # Encode categorical columns
            categorical_cols = self.hr_data.select_dtypes(include='object').columns.tolist()
            exclude_cols = ['OriginalEmployeeNumber', 'OriginalGender', 'OriginalDepartment', 'OriginalAge']
            categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

            for col in categorical_cols:
                le = LabelEncoder()
                self.hr_data[col] = le.fit_transform(self.hr_data[col])
                self.label_encoders[col] = le

            # Scale numeric features
            numeric_cols = self.hr_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            exclude_cols.extend(['Attrition', 'EmployeeNumber', 'OriginalEmployeeNumber'])
            numeric_to_scale = [col for col in numeric_cols if col not in exclude_cols]
            self.hr_data[numeric_to_scale] = self.scaler.fit_transform(self.hr_data[numeric_to_scale])

            # Load skills data
            self.skills_data = pd.read_csv(f"{settings.DATA_DIR}/{settings.SKILLS_CSV}")
            self.skills_data["skills"] = self.skills_data["skills"].apply(
                lambda x: literal_eval(x) if pd.notnull(x) else []
            )

            # Load applicants data
            self.applicants_data = pd.read_csv(f"{settings.DATA_DIR}/{settings.APPLICANTS_CSV}")
            self.applicants_data["Skills"] = self.applicants_data["Skills"].apply(
                lambda x: literal_eval(x) if pd.notnull(x) else []
            )

        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return loaded datasets."""
        return self.hr_data, self.skills_data, self.applicants_data

data_loader = DataLoader()