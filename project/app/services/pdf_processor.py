import fitz
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from fastapi import HTTPException

def extract_pdf_text(contents: bytes) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(stream=contents, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def analyze_project(text: str) -> dict:
    """Analyze project description with LLM to extract skills and team plan."""
    prompt_template = PromptTemplate(
        input_variables=["project_description"],
        template="""You are an assistant that reads project descriptions and extracts:
1. Required skills
2. Experience level (junior/mid/senior)
3. Team composition (roles + number of people needed)

Skills should be from: 
    "Healthcare Representative": [
        "Patient Communication", "Empathy", "CRM", "Conflict Resolution", "Healthcare Knowledge",
        "Emotional Intelligence", "Time Management", "Regulatory Compliance", "Customer Service"
    ],
    "Human Resources": [
        "Recruitment", "Employee Engagement", "Conflict Resolution", "Interviewing", "People Analytics",
        "HR Policies", "Onboarding", "Performance Appraisal", "Communication", "Payroll Processing"
    ],
    "Laboratory Technician": [
        "Lab Safety", "Data Entry", "Sample Analysis", "Time Management", "Equipment Handling",
        "Chemical Handling", "Standard Operating Procedures (SOP)", "Attention to Detail", "Microscopy"
    ],
    "Manager": [
        "Leadership", "Project Management", "Strategic Planning", "Team Management", "Budget Management",
        "Conflict Resolution", "Performance Reviews", "Cross-functional Collaboration", "KPI Tracking"
    ],
    "Manufacturing Director": [
        "Lean Manufacturing", "Operations Management", "Inventory Planning", "Supply Chain Coordination",
        "Six Sigma", "Workforce Scheduling", "Compliance Management", "Budgeting", "Process Improvement"
    ],
    "Research Director": [
        "Strategic Planning", "Research Management", "Innovation", "Machine Learning", "Data Science",
        "Academic Publishing", "Grant Writing", "Team Leadership", "Ethics & Compliance"
    ],
    "Research Scientist": [
        "Data Analysis", "Python", "Scientific Writing", "Hypothesis Testing", "Machine Learning",
        "Statistics", "Experiment Design", "Microscopy", "Literature Review", "Lab Safety"
    ],
    "Sales Executive": [
        "Negotiation", "Customer Acquisition", "Salesforce", "Time Management", "Lead Scoring",
        "Cold Calling", "B2B Sales", "Presentation Skills", "Revenue Forecasting", "CRM Tools"
    ],
    "Sales Representative": [
        "Communication", "Product Knowledge", "CRM", "Customer Service", "Sales Presentations",
        "Relationship Building", "Upselling", "Territory Management", "Time Management", "Networking"
    ]

Based on this project:
{project_description}

Return the result in this JSON format:
{{
  "required_skills": [],
  "team": [
    {{
      "title": "Senior Frontend Developer",
      "experience_level": "Senior (5+ years)",
      "skills": ["React", "TypeScript"],
      "count": 2
    }},
    {{
      "title": "Backend Developer",
      "skills": ["Node.js", "AWS"],
      "experience_level": "Senior (5+ years)",
      "count": 1
    }}
  ]
}}
"""
    )
    llm = ChatOpenAI(temperature=0)
    chain = prompt_template | llm
    response = chain.invoke({"project_description": text})
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse LLM response.")