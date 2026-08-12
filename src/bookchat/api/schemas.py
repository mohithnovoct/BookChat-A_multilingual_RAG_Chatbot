from typing import List

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty.")
        return stripped


class QueryResponse(BaseModel):
    question: str
    answer: str


class IngestResponse(BaseModel):
    message: str
    files_processed: List[str]
    chunks_created: int
    files_replaced: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ResetResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
