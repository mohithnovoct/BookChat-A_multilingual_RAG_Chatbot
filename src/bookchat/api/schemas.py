from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    k: int = Field(default=8, ge=1, le=20)
    answer_language: Literal["en"] = "en"
    query_language: Literal["auto", "en", "kn", "pa"] = "auto"

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
    sources: List["SourceReference"] = Field(default_factory=list)


class SourceReference(BaseModel):
    filename: str
    page: int | None = None
    chunk_id: int | None = None
    score: float | None = None


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
