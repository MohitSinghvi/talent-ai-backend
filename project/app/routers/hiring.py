from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
import pandas as pd
import random
import math
from ast import literal_eval
from app.services.data_loader import data_loader
from app.models.ml_model import AttritionModel

router = APIRouter()

# Initialize data and model
hr_data, skills_data, applicants_data = data_loader.get_data()
attrition_model = AttritionModel(hr_data, skills_data)
predict_data = attrition_model.get_predict_data()
feature_columns = attrition_model.get_feature_columns()
feature_importances = attrition_model.get_feature_importances()

@router.get("/predictive_hiring")
def predictive_hiring():
    """Suggest hires based on attrition risk threshold."""
    threshold = 0.5
    num_employees_at_risk = predict_data[predict_data['Attrition_Risk'] > threshold].shape[0]
    retention_rate = attrition_model.get_retention_rate()  
    replacement_cost = attrition_model.get_total_replacement_cost()

    number_of_hires = num_employees_at_risk
    talent_pool = [
        {"CandidateID": 101, "Name": "Alice Smith", "Experience": "5 years", "Skill": "Sales"},
        {"CandidateID": 102, "Name": "Bob Johnson", "Experience": "3 years", "Skill": "Marketing"},
        {"CandidateID": 103, "Name": "Carol Davis", "Experience": "6 years", "Skill": "Engineering"},
        {"CandidateID": 104, "Name": "David Wilson", "Experience": "4 years", "Skill": "Customer Service"},
        {"CandidateID": 105, "Name": "Eva Brown", "Experience": "7 years", "Skill": "HR"}
    ]
    suggested_candidates = talent_pool if number_of_hires > len(talent_pool) else random.sample(talent_pool, number_of_hires)
    return {
        "num_employees_at_risk":num_employees_at_risk,
        "number_of_hires": number_of_hires,
        "suggested_candidates": suggested_candidates,
        "retention_rate": retention_rate,
        "replacement_cost": replacement_cost
        # "percentage_change": round(percentage_change, 1),
    }


@router.get("/talent_pool")
def get_talent_pool(
    
    page: int = Query(1, ge=1),
    limit: int = Query(6, ge=1),
    search: str = Query("", alias="search"),
    experience: str = Query("", regex="^(0-2|3-5|6\+)?$"),
    roles: str = Query("", alias="roles"),
    sort_by: str = Query("matchScore", regex="^(matchScore|experience|education)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$")
):
    """Search and paginate the talent pool with filters."""
    if applicants_data.empty:
        return {"candidates": [], "total": 0, "page": page, "totalPages": 0}

    df = applicants_data.copy()

    # Apply search filter
    if search:
        search_lower = search.lower()
        df = df[df.apply(
            lambda row: search_lower in f"{row['First Name']} {row['Last Name']}".lower() or
                        search_lower in row['Job Title'].lower() or
                        search_lower in row.get('City', '').lower() or
                        search_lower in row.get('State', '').lower() or
                        any(search_lower in skill.lower() for skill in row['Skills']),
            axis=1
        )]

    # Apply experience filter
    if experience:
        if experience == "0-2":
            df = df[df["Years of Experience"].between(0, 2)]
        elif experience == "3-5":
            df = df[df["Years of Experience"].between(3, 5)]
        elif experience == "6+":
            df = df[df["Years of Experience"] >= 6]
    # Apply role filter
    if roles:
        role_list = [r.strip() for r in roles.split(",") if r.strip()]
        if role_list:
            df = df[df["Job Title"].isin(role_list)]

    # Derived values for sorting
    df["matchScore"] = 75 + (df.index % 25)
    df["experienceValue"] = df["Years of Experience"].fillna(0)
    df["education"] = df["Education Level"].fillna("")

    # Sorting
    if sort_by == "matchScore":
        df = df.sort_values(by="matchScore", ascending=(sort_order == "asc"))
    elif sort_by == "experience":
        df = df.sort_values(by="experienceValue", ascending=(sort_order == "asc"))
    elif sort_by == "education":
        df = df.sort_values(by="education", ascending=(sort_order == "asc"))

    # Pagination
    total = len(df)
    total_pages = math.ceil(total / limit)
    start = (page - 1) * limit
    end = start + limit
    sliced = df.iloc[start:end]

    # Format response
    candidates = [
        {
            "id": int(row["Applicant ID"]),
            "name": f"{row['First Name']} {row['Last Name']}",
            "role": row["Job Title"],
            "email": row["Email"],
            "experience": f"{int(row['Years of Experience'])} years" if pd.notnull(row["Years of Experience"]) else "N/A",
            "education": row.get("Education Level", "N/A"),
            "location": f"{row.get('City', '')}, {row.get('State', '')}",
            "skills": row["Skills"],
            "matchScore": int(row["matchScore"]),
            "availability": "Immediate" if idx % 3 == 0 else "2 weeks",
            "image": f"https://source.unsplash.com/80x80/?portrait&sig={idx}"
        } for idx, row in sliced.iterrows()
    ]

    return {
        "candidates": candidates,
        "total": total,
        "page": page,
        "totalPages": total_pages
    }


@router.get("/hire_alternatives")
def hire_alternatives():
    """Match high-risk employees with candidate alternatives."""
    try:
        top_employees = predict_data.nlargest(5, 'Attrition_Risk')
        results = []

        for _, emp in top_employees.iterrows():
            emp_id = int(emp["OriginalEmployeeNumber"])
            emp_record = skills_data[skills_data["EmployeeNumber"] == emp_id].squeeze()

            employee_features = predict_data.loc[emp.name, feature_columns]
            abs_contributions = employee_features * feature_importances
            top_reason = abs_contributions.idxmax()

            emp_skills = set(literal_eval(emp_record["skills"])) if isinstance(emp_record["skills"], str) else set(emp_record["skills"])
            emp_obj = {
                "name": f"{emp_record['FirstName']} {emp_record['LastName']}",
                "role": emp_record["JobRole"],
                "department": emp_record["Department"],
                "tenure": f"{emp_record['YearsAtCompany']} years",
                "riskScore": round(float(emp["Attrition_Risk"]) * 100, 2),
                "reason": top_reason,
                "skills": list(emp_skills),
                "image": f"https://source.unsplash.com/80x80/?portrait&sig={emp_id}"
            }

            candidate_matches = []
            for _, cand in applicants_data.iterrows():
                cand_skills = set(cand["Skills"])
                overlap = len(emp_skills & cand_skills)
                emp_exp = emp_record.get("TotalWorkingYears", 0)
                cand_exp = cand.get("Years of Experience", 0)
                exp_score = max(0, 1 - abs(emp_exp - cand_exp) / 10)
                skill_score = (overlap / len(emp_skills)) if emp_skills else 0
                final_score = round((0.7 * skill_score + 0.3 * exp_score) * 100, 2)

                candidate_matches.append({
                    "name": f"{cand['First Name']} {cand['Last Name']}",
                    "role": cand["Job Title"],
                    "experience": f"{int(cand['Years of Experience'])} years",
                    "matchScore": final_score,
                    "skills": list(cand_skills),
                    "image": f"https://source.unsplash.com/80x80/?portrait&sig={cand['Applicant ID']}"
                })

            top_candidates = sorted(candidate_matches, key=lambda x: -x["matchScore"])[:3]
            results.append({
                "employee": emp_obj,
                "alternatives": top_candidates
            })

        return jsonable_encoder(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/department_data")
def get_department_data():
    """Get department-level employee data and metrics."""
    if skills_data.empty:
        raise HTTPException(status_code=404, detail="No employee metadata loaded.")

    merged = skills_data.merge(
        predict_data[["OriginalEmployeeNumber", "Attrition_Risk"]],
        left_on="EmployeeNumber",
        right_on="OriginalEmployeeNumber",
        how="left"
    )
    merged["Attrition_Risk"].fillna(0, inplace=True)
    departments = []
    department_list = sorted(merged["Department"].unique().tolist())  
    department_id_map = {dept: idx for idx, dept in enumerate(department_list)}  
    for dept_name, grp in merged.groupby("Department"):
        preview_emps = [
            {
                "name": f"{row['FirstName']} {row['LastName']}",
                "role": row["JobRole"],
                "email": f"{row['FirstName'].lower()}.{row['LastName'].lower()}@company.com",
                "location": "Unknown",
                "image": f"https://source.unsplash.com/80x80/?portrait&sig={row['EmployeeNumber']}"
            } for _, row in grp.head(5).iterrows()
        ]

        departments.append({
            "id": department_id_map[dept_name],
            "name": dept_name,
            "headCount": int(grp.shape[0]),
            "manager": {
                "name": f"Manager {dept_name}",
                "email": f"manager.{dept_name.lower()}@company.com",
                "phone": "+1 (555) 000-0000",
                "image": f"https://source.unsplash.com/80x80/?portrait&sig={hash(dept_name) % 1000}"
            },
            "employees": preview_emps,
            "metrics": {
                "attritionRate": f"{round(grp['Attrition_Risk'].mean() * 100, 1)}%",
                "avgTenure": f"{round(grp['YearsAtCompany'].mean(), 1)} years",
                "openPositions": int(grp.shape[0] * 0.1)
            }
        })

    return jsonable_encoder(departments)

@router.get("/department_data/name/{department}")
def get_department_employees_by_department(department: str):
    """Get all employees for a specific department by name (case-insensitive)."""
    if not department:
        raise HTTPException(status_code=400, detail="Invalid department name")

    merged = skills_data.merge(
        predict_data[["OriginalEmployeeNumber", "Attrition_Risk"]],
        left_on="EmployeeNumber",
        right_on="OriginalEmployeeNumber",
        how="left"
    )
    merged["Attrition_Risk"].fillna(0, inplace=True)

    department_normalized = department.strip().lower()
    filtered = merged[merged["Department"].str.lower() == department_normalized]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No employees found in this department")

    employees = [
        {
            "name": f"{row['FirstName']} {row['LastName']}",
            "role": row["JobRole"],
            "email": (
                f"{str(row.get('FirstName', '')).strip().lower()}."
                f"{str(row.get('LastName', '')).strip().lower()}@company.com"
            ),
            "image": f"https://source.unsplash.com/80x80/?portrait&sig={row['EmployeeNumber']}"
        }
        for _, row in filtered.iterrows()
    ]

    return jsonable_encoder(employees)

@router.get("/employee_by_name")
def get_employee_by_name(name: str = Query(..., description="Full, first, or last name of the employee")):
    """Search employees by name."""
    name_lower = name.lower()

    def match_score(row):
        first = str(row.get("FirstName", "") or "").lower()
        last = str(row.get("LastName", "") or "").lower()
        full = f"{first} {last}"
        score = 0
        if name_lower == full:
            score += 3
        elif name_lower in full:
            score += 2
        elif name_lower in first or name_lower in last:
            score += 1
        return score

    df = skills_data.copy()
    df["score"] = df.apply(match_score, axis=1)
    matched = df[df["score"] > 0].sort_values(by="score", ascending=False).drop(columns=["score"])
    return jsonable_encoder(matched.to_dict(orient="records"))

@router.get("/candidate_by_name")
def get_candidate_by_name(name: str = Query(..., description="First, last, or full name of the candidate")):
    """Search candidates by name."""
    if applicants_data.empty:
        raise HTTPException(status_code=404, detail="Candidate dataset not available.")

    name_query = name.lower().strip()

    def get_match_score(row):
        first = row["First Name"].lower().strip()
        last = row["Last Name"].lower().strip()
        full = f"{first} {last}"
        if name_query == full:
            return 3
        elif name_query == first:
            return 2
        elif name_query == last:
            return 1
        return 0

    df = applicants_data.copy()
    df["match_score"] = df.apply(get_match_score, axis=1)
    matched = df[df["match_score"] > 0].sort_values(by="match_score", ascending=False)

    if matched.empty:
        return {"message": f"No candidates found matching '{name}'."}

    results = [
        {
            **row.to_dict(),
            "name": f"{row['First Name']} {row['Last Name']}",
            "image": f"https://source.unsplash.com/80x80/?portrait&sig={row['Applicant ID']}"
        } for _, row in matched.iterrows()
    ]
    return jsonable_encoder(results)

@router.get("/employee/{emp_number}")
def get_employee_detail(emp_number: int):
    """Get employee details by EmployeeNumber."""
    if skills_data.empty:
        raise HTTPException(status_code=404, detail="Employee data not loaded.")
    row = skills_data[skills_data["EmployeeNumber"] == emp_number]
    if row.empty:
        raise HTTPException(status_code=404, detail="Employee not found.")
    r = row.iloc[0].to_dict()
    risk = predict_data.loc[predict_data["OriginalEmployeeNumber"] == emp_number, "Attrition_Risk"]
    r["Attrition_Risk"] = float(risk.iloc[0]) if not risk.empty else None
    return jsonable_encoder(r)

@router.get("/candidate/{app_id}")
def get_candidate_detail(app_id: int):
    """Get candidate details by Applicant ID."""
    if applicants_data.empty:
        raise HTTPException(status_code=404, detail="Candidate data not loaded.")
    row = applicants_data[applicants_data["Applicant ID"] == app_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return jsonable_encoder(row.iloc[0].to_dict())