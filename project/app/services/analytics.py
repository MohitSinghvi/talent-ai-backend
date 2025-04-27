from fastapi.encoders import jsonable_encoder
import pandas as pd
from app.config import settings
from app.services.data_loader import data_loader
from typing import List, Dict

def get_department_pie_data(predict_data: pd.DataFrame) -> List[Dict]:
    """Calculate mean attrition risk per department for pie chart."""
    department_counts = predict_data.groupby("OriginalDepartment")["Attrition_Risk"].mean() * 100
    df = department_counts.reset_index()
    df.columns = ["Department", "Percentage"]
    return jsonable_encoder(df.to_dict(orient="records"))

def get_department_metrics() -> List[Dict]:
    """Calculate attrition risk and job involvement by department."""
    raw = pd.read_csv(f"{settings.DATA_DIR}/{settings.EMPLOYEE_CSV}")
    _, _, predict_data = data_loader.get_data()
    df = raw.merge(
        predict_data[['OriginalEmployeeNumber', 'Attrition_Risk']],
        left_on='EmployeeNumber',
        right_on='OriginalEmployeeNumber',
        how='left'
    )
    df['Attrition_Risk'].fillna(0, inplace=True)
    grouped = df.groupby('Department').agg({
        'Attrition_Risk': 'mean',
        'JobInvolvement': 'mean'
    }).reset_index()
    result = [
        {
            'Department': row['Department'],
            'Attrition_Risk': round(row['Attrition_Risk'] * 100, 2),
            'JobInvolvement': round(row['JobInvolvement'], 2)
        } for _, row in grouped.iterrows()
    ]
    return jsonable_encoder(result)

def get_top_risk_factors(feature_importances: pd.Series, top_n: int = 5) -> List[Dict]:
    """Get top N risk factors from feature importances."""
    sorted_importances = feature_importances.sort_values(ascending=False)
    top_factors = sorted_importances.head(top_n)
    result = [
        {"risk_factor": feature, "importance": round(importance, 4)}
        for feature, importance in top_factors.items()
    ]
    return jsonable_encoder(result)