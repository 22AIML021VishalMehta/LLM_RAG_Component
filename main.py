# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import run_query

app = FastAPI(title="LLM RAG Medical Assistant")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: list

@app.post("/query", response_model=QueryResponse)
def query_handler(request: QueryRequest):
    result = run_query(request.query)
    return result
