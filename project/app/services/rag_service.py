from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
import httpx
from app.config import settings

def setup_rag():
    """Set up RAG with FAISS vectorstore."""
    loader_emp = CSVLoader(f"{settings.DATA_DIR}/{settings.SKILLS_CSV}")
    loader_cand = CSVLoader(f"{settings.DATA_DIR}/{settings.APPLICANTS_CSV}")
    docs = loader_emp.load() + loader_cand.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return rag_chain, vectorstore

def setup_agent(rag_chain):
    """Initialize LangChain agent with tools."""
    api_tools = [
        Tool(
            name="risk_curve_data",
            func=lambda _: httpx.get("http://localhost:8000/analytics/risk_curve_data").text,
            description="Get average attrition risk by age range and gender."
        ),
        Tool(
            name="department_pie_data",
            func=lambda _: httpx.get("http://localhost:8000/analytics/department_pie_data").text,
            description="Get mean attrition risk per department."
        ),
        Tool(
            name="top_employees_data",
            func=lambda _: httpx.get("http://localhost:8000/analytics/top_employees_data").text,
            description="Get the top 10 highest-risk employees."
        ),
        Tool(
            name="predictive_hiring",
            func=lambda _: httpx.get("http://localhost:8000/hiring/predictive_hiring").text,
            description="Get number of hires needed and suggested candidates."
        ),
        Tool(
            name="hire_alternatives",
            func=lambda _: httpx.get("http://localhost:8000/hiring/hire_alternatives").text,
            description="Get high-risk employees and candidate alternatives."
        ),
        Tool(
            name="employee_detail",
            func=lambda q: httpx.get(f"http://localhost:8000/hiring/employee/{''.join(filter(str.isdigit, q))}").text,
            description="Get employee details by EmployeeNumber."
        ),
        Tool(
            name="candidate_detail",
            func=lambda q: httpx.get(f"http://localhost:8000/hiring/candidate/{''.join(filter(str.isdigit, q))}").text,
            description="Get candidate details by Applicant ID."
        ),
        Tool(
            name="talent_pool",
            func=lambda q: httpx.get("http://localhost:8000/hiring/talent_pool", params={"search": q, "page": 1, "limit": 10}).text,
            description="Search the candidate talent pool by skills, name, or location."
        ),
        Tool(
            name="get_employee_by_name",
            func=lambda name: httpx.get("http://localhost:8000/hiring/employee_by_name", params={"name": name}).text,
            description="Get employee details by name."
        ),
        Tool(
            name="get_candidate_by_name",
            func=lambda name: httpx.get("http://localhost:8000/hiring/candidate_by_name", params={"name": name}).text,
            description="Get candidate details by name."
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=api_tools + [
            Tool(
                name="employee_facts",
                func=lambda q: rag_chain.run(q),
                description="Answer questions about employees or candidates."
            )
        ],
        llm=ChatOpenAI(temperature=0),
        agent="conversational-react-description",
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent