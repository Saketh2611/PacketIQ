"""Typed schemas for structured documents and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    packet_id: str
    source_file: str
    page_start: int
    page_end: int


class ContentBlock(BaseModel):
    type: Literal["paragraph", "list", "table", "figure", "heading"]
    text: str = ""
    page: int
    headers: list[str] | None = None
    rows: list[list[str]] | None = None
    items: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    heading: str | None = None
    page: int
    blocks: list[ContentBlock] = Field(default_factory=list)


class DocumentContent(BaseModel):
    title: str | None = None
    sections: list[Section] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    packet_id: str
    document_id: str
    document_type: str
    source_file: str
    page_start: int
    page_end: int
    page_count: int
    extraction_method: str = "native"
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_hash: str = ""
    model_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    document_id: str
    document_type: str
    confidence: float
    source: SourceInfo
    content: DocumentContent
    metadata: DocumentMetadata


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_type: str
    page_start: int
    page_end: int
    section: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    document_id: str
    document_type: str
    page: int
    chunk_id: str
    evidence: str
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None


class Packet(BaseModel):
    packet_id: str
    source_file: str
    page_count: int
    documents: list[StructuredDocument] = Field(default_factory=list)


class Page(BaseModel):
    page_id: str
    page_number: int
    text: str
    extraction_method: str
