"""Generate consistent document metadata."""

from __future__ import annotations

from document_intelligence.config.settings import get_settings
from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.stage1.grouping import DocumentGroup
from document_intelligence.stage1.document_classifier import ClassificationResult
from document_intelligence.stage2.schema import DocumentMetadata
from document_intelligence.utils.hashing import hash_text


def build_document_metadata(
    packet_id: str,
    group: DocumentGroup,
    classification: ClassificationResult,
    pages: list[PageRepresentation],
    source_file: str,
) -> DocumentMetadata:
    combined_text = "\n".join(p.text for p in pages)
    methods = {p.extraction_method for p in pages}
    extraction = methods.pop() if len(methods) == 1 else "mixed"
    warnings = []
    for p in pages:
        warnings.extend(p.warnings)

    settings = get_settings()
    return DocumentMetadata(
        packet_id=packet_id,
        document_id=group.document_id,
        document_type=classification.document_type,
        source_file=source_file,
        page_start=group.page_start,
        page_end=group.page_end,
        page_count=group.page_count,
        extraction_method=extraction,
        content_hash=hash_text(combined_text),
        model_versions={"embedding": settings.embedding_model},
        warnings=list(set(warnings)),
    )
