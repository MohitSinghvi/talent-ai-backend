from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from api_routes.mongo_chain import generate_pipeline_with_langchain
import os

router = APIRouter()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "hr"
COLLECTION_NAME = "employees"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

class NLQuery(BaseModel):
    question: str

@router.post("/chat")
def chat_to_mongo(nlq: NLQuery):
    try:
        schema_hint = "Fields: Age (int), Department (str), Salary (float), Gender (str), Education (str)"
        pipeline = generate_pipeline_with_langchain(nlq.question, schema_hint)
        results = list(collection.aggregate(pipeline))
        return {
            "question": nlq.question,
            "pipeline": pipeline,
            "results": results
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
