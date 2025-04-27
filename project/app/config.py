from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATA_DIR: str = str(Path(__file__).parent.parent / "data")
    EMPLOYEE_CSV: str = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    SKILLS_CSV: str = "employee_with_skills.csv"
    APPLICANTS_CSV: str = "recruitment_applicants_corrected.csv"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()