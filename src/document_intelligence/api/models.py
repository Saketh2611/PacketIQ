"""FastAPI request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class AnalyzeResponse(BaseModel):
    packet_id: str
    page_count: int
    documents: list[dict[str, Any]]
    boundaries: list[dict[str, Any]]
    processing_time_seconds: float
    warnings: list[str] = Field(default_factory=list)


class IndexRequest(BaseModel):
    structured_dir: str | None = None
    stage1_path: str | None = None
    pdf_path: str | None = None


class IndexResponse(BaseModel):
    chunks_indexed: int
    index_path: str
    indexing_time_seconds: float


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    document_type: str | None = None
    use_reranker: bool | None = None


class RetrieveResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    latency_seconds: float
    warnings: list[str] | None = None
