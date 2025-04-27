from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import analytics, chat, hiring, project

app = FastAPI(title="HR Analytics API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analytics.router, prefix="/analytics")
app.include_router(chat.router, prefix="/chat")
app.include_router(hiring.router, prefix="/hiring")
app.include_router(project.router, prefix="/project")

@app.get("/")
def read_root():
    return {"message": "Welcome to the HR Analytics API! Use /docs for documentation."}