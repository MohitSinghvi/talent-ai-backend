from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt_template = PromptTemplate.from_template("""
You are a MongoDB expert assistant.
Your task is to convert natural language questions into MongoDB aggregation pipelines.
Respond ONLY with valid JSON representing the aggregation pipeline.

Collection schema:
{schema}

Question:
{question}
""")

def generate_pipeline_with_langchain(question: str, schema: str):
    prompt = prompt_template.format(schema=schema, question=question)
    response = llm.predict(prompt)

    try:
        return json.loads(response.strip())
    except Exception as e:
        raise ValueError(f"Invalid JSON from LLM: {response}")
