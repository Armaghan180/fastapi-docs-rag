"""Pydantic request/response models for the /chat endpoint."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chunking_strategy: str = "heading_based"
    retrieval_method: str = "dense"


class CitationOut(BaseModel):
    index: int
    url: str
    doc_path: str
    heading_path: list[str]


class RetrievedChunkOut(BaseModel):
    doc_path: str
    heading_path: list[str]
    url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved_chunks: list[RetrievedChunkOut]
    chunking_strategy: str
    retrieval_method: str


class StrategyOption(BaseModel):
    chunking_strategy: str
    retrieval_method: str
    label: str


class StrategiesResponse(BaseModel):
    strategies: list[StrategyOption]
    default: StrategyOption
