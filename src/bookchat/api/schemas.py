from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    question: str
    answer: str


class IngestResponse(BaseModel):
    message: str
    files_processed: List[str]

class ResetResponse(BaseModel):
    message: str