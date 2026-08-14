"""Stage 2 evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from document_intelligence.stage2.schema import StructuredDocument


@dataclass
class Stage2Metrics:
    documents_processed: int
    avg_sections_per_doc: float
    avg_blocks_per_doc: float
    provenance_correct: int
    provenance_total: int
    processing_time_seconds: float
    warnings_count: int


def evaluate_stage2(
    documents: list[StructuredDocument],
    processing_time: float,
) -> Stage2Metrics:
    sections = [len(d.content.sections) for d in documents]
    blocks = [sum(len(s.blocks) for s in d.content.sections) for d in documents]
    prov_ok = 0
    for d in documents:
        if d.source.page_start <= d.source.page_end and d.metadata.page_count > 0:
            prov_ok += 1
    warnings = sum(len(d.metadata.warnings) for d in documents)
    return Stage2Metrics(
        documents_processed=len(documents),
        avg_sections_per_doc=sum(sections) / len(sections) if sections else 0.0,
        avg_blocks_per_doc=sum(blocks) / len(blocks) if blocks else 0.0,
        provenance_correct=prov_ok,
        provenance_total=len(documents),
        processing_time_seconds=processing_time,
        warnings_count=warnings,
    )
