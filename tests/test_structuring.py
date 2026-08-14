"""Tests for structured output."""

from document_intelligence.stage1.document_classifier import ClassificationResult
from document_intelligence.stage1.grouping import DocumentGroup
from document_intelligence.stage2.schema import StructuredDocument
from document_intelligence.stage2.structure import StructureExtractor


def test_structured_json_schema(sample_pages):
    extractor = StructureExtractor()
    group = DocumentGroup(
        document_id="doc_001",
        page_numbers=[1, 2],
        page_start=1,
        page_end=2,
    )
    classification = ClassificationResult(document_type="invoice", confidence=0.9)
    doc = extractor.extract("packet_001", "test.pdf", group, sample_pages[:2], classification)
    assert isinstance(doc, StructuredDocument)
    assert doc.document_id == "doc_001"
    assert doc.source.page_start == 1
    validated = StructuredDocument.model_validate(doc.model_dump())
    assert validated.document_type == "invoice"
