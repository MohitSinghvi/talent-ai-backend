from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import TeamMemberSuggestion, ProjectAnalysisResponse
from app.services.data_loader import data_loader
from app.services.pdf_processor import extract_pdf_text, analyze_project
import pandas as pd
import re
from typing import List

router = APIRouter()

@router.post("/upload_project_pdf", response_model=ProjectAnalysisResponse)
async def upload_project_pdf(file: UploadFile = File(...)):
    """Upload a PDF and analyze project requirements."""
    contents = await file.read()
    text = extract_pdf_text(contents)
    parsed = analyze_project(text)

    _, _, applicants_data = data_loader.get_data()
    final_team_plan = []

    team_items = parsed.get("team", [])
    if not team_items:
        raise HTTPException(status_code=400, detail="No team plan found in response.")

    for item in team_items:
        role = item.get("title", "Unknown Role")
        required_skills = item.get("skills", [])
        count = item.get("count", 1)

        def normalize(skill):
            return re.sub(r"[^\w]", "", skill.lower()) if isinstance(skill, str) else ""

        def skill_match(row):
            candidate_skills = [normalize(s) for s in row["Skills"] if isinstance(s, str)]
            required_skills_normalized = [normalize(s) for s in required_skills if isinstance(s, str)]
            match_count = sum(1 for req in required_skills_normalized for cand in candidate_skills if req in cand or cand in req)
            return match_count

        applicants_data["matchScore"] = applicants_data.apply(skill_match, axis=1)
        top_candidates = applicants_data[applicants_data["matchScore"] > 0] \
            .sort_values(by="matchScore", ascending=False) \
            .head(max(count, 4))

        matching = [
            {
                "name": f"{row['First Name']} {row['Last Name']}",
                "experience": f"{row['Years of Experience']} years",
                "skills": row["Skills"],
                "role": row["Job Title"],
                "location": f"{row.get('City', '')}, {row.get('State', '')}",
            } for _, row in top_candidates.iterrows()
        ]

        final_team_plan.append(TeamMemberSuggestion(
            role=role,
            required_skills=required_skills,
            count_needed=count,
            matching_candidates=matching
        ))

    return ProjectAnalysisResponse(
        extracted_text=text,
        required_skills=parsed.get("required_skills", []),
        team_plan=final_team_plan
    )