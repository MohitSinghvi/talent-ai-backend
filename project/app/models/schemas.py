from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

class TeamMemberSuggestion(BaseModel):
    role: str
    required_skills: List[str]
    count_needed: int
    matching_candidates: List[Dict]

class ProjectAnalysisResponse(BaseModel):
    extracted_text: str
    required_skills: List[str]
    team_plan: List[TeamMemberSuggestion]