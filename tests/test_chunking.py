"""Tests for chunking."""

from document_intelligence.stage1.document_classifier import ClassificationResult
from document_intelligence.stage1.grouping import DocumentGroup
from document_intelligence.stage2.chunking import DocumentChunker
from document_intelligence.stage2.structure import StructureExtractor


def test_chunking_provenance(sample_pages):
    extractor = StructureExtractor()
    group = DocumentGroup(document_id="doc_001", page_numbers=[1, 2], page_start=1, page_end=2)
    classification = ClassificationResult(document_type="invoice", confidence=0.9)
    doc = extractor.extract("pkt", "test.pdf", group, sample_pages[:2], classification)
    chunker = DocumentChunker(max_chars=500)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "doc_001"
        assert c.page_start >= 1
        assert c.chunk_id.startswith("doc_001_chunk")
