"""FastAPI application."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from document_intelligence.api.models import (
    AnalyzeResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from document_intelligence.config.settings import get_settings
from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.utils.io import read_json, write_json

app = FastAPI(title="Document Intelligence API", version="0.1.0")
pipeline = DocumentIntelligencePipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile | None = File(None),
    pdf_path: str | None = None,
    method: str = "baseline",
) -> AnalyzeResponse:
    settings = get_settings()
    if file is not None:
        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            path = tmp.name
    elif pdf_path:
        path = pdf_path
    else:
        raise HTTPException(status_code=400, detail="Provide file upload or pdf_path")

    try:
        result = pipeline.run_stage1(path, method=method)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalyzeResponse(
        packet_id=result["packet_id"],
        page_count=result["page_count"],
        documents=result["documents"],
        boundaries=result["boundaries"],
        processing_time_seconds=result["processing_time_seconds"],
        warnings=result.get("warnings", []),
    )


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    settings = get_settings()
    structured_docs = []

    if request.stage1_path and request.pdf_path:
        stage1 = read_json(request.stage1_path)
        structured_docs = pipeline.run_stage2(stage1, request.pdf_path)
        out_dir = settings.outputs_dir / "structured"
        out_dir.mkdir(parents=True, exist_ok=True)
        for doc in structured_docs:
            write_json(out_dir / f"{doc.document_id}.json", doc.model_dump())
    elif request.structured_dir:
        from document_intelligence.stage2.schema import StructuredDocument

        for f in Path(request.structured_dir).glob("*.json"):
            structured_docs.append(StructuredDocument(**read_json(f)))
    else:
        raise HTTPException(status_code=400, detail="Provide structured_dir or stage1_path+pdf_path")

    stats = pipeline.build_index(structured_docs)
    return IndexResponse(**stats)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        result = pipeline.query(request.query, top_k=request.top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found. Run /index first.")
    return RetrieveResponse(**result)
