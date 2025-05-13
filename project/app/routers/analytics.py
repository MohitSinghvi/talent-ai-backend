from fastapi import APIRouter, Query
from fastapi.encoders import jsonable_encoder
from app.services.data_loader import data_loader
from app.models.ml_model import AttritionModel
from app.services.analytics import get_department_pie_data, get_department_metrics, get_top_risk_factors

router = APIRouter()

# Initialize ML model
hr_data, skills_data, applicants_data = data_loader.get_data()
attrition_model = AttritionModel(hr_data, skills_data)
predict_data = attrition_model.get_predict_data()
feature_columns = attrition_model.get_feature_columns()
feature_importances = attrition_model.get_feature_importances()

@router.get("/risk_curve_data")
def get_risk_curve_data():
    """Get average attrition risk by age range and gender."""
    grouped = predict_data.groupby(['Age_Range', 'OriginalGender']).agg({
        'Attrition_Risk': 'mean',
        'OriginalEmployeeNumber': 'count'
    }).reset_index()
    grouped['Attrition_Percentage'] = grouped['Attrition_Risk'] * 100
    return jsonable_encoder(grouped.to_dict(orient='records'))

@router.get("/top_employees_data")
def get_top_employees_data():
    """Get the top 10 highest-risk employees with their top 3 contributing factors."""
    top_employees = predict_data.nlargest(10, 'Attrition_Risk').copy()
    reasons = []

    for index, row in top_employees.iterrows():
        employee_features = predict_data.loc[index, feature_columns].astype(float)
        feature_importances_numeric = feature_importances.astype(float)
        abs_contributions = employee_features * feature_importances_numeric
        # Get the top 3 contributing factors
        top_3_reasons = abs_contributions.nlargest(3).index.tolist()

        emp_number = row["OriginalEmployeeNumber"]
        emp_record = skills_data[skills_data["EmployeeNumber"] == emp_number].squeeze()

        reasons.append({
            "OriginalEmployeeNumber": int(emp_number),
            "FirstName": emp_record.get("FirstName", ""),
            "LastName": emp_record.get("LastName", ""),
            "Attrition_Risk_Percentage": round(float(row["Attrition_Risk"]) * 100, 2),
            "Top_Contributing_Factors": top_3_reasons,
            "Department": row["OriginalDepartment"]
        })

    return jsonable_encoder(reasons)

@router.get("/model_evaluation")
def get_model_evaluation():
    """Get model performance metrics."""
    return jsonable_encoder(attrition_model.get_metrics())

@router.get("/department_pie_data")
def get_department_pie_data_endpoint():
    """Get mean attrition risk per department for pie chart."""
    return get_department_pie_data(predict_data)

@router.get("/department_metrics")
def get_department_metrics():
    """Get attrition risk and job involvement by department."""
    return get_department_metrics()

@router.get("/top_risk_factors")
def get_top_risk_factors_endpoint(top_n: int = Query(5, ge=1)):
    """Get top N risk factors from feature importances."""
    return get_top_risk_factors(feature_importances, top_n)