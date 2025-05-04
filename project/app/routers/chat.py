from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import setup_rag, setup_agent

router = APIRouter()

# Setup RAG and agent
# rag_chain, _ = setup_rag()
# agent = setup_agent(rag_chain)

@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Handle chat requests using the LangChain agent."""
    result = agent.run(req.question)
    return ChatResponse(answer=f"**Answer:**\n\n{result}", sources=[])