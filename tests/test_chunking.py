"""Tests for chunking."""

from document_intelligence.stage1.document_classifier import ClassificationResult
from document_intelligence.stage1.grouping import DocumentGroup
from document_intelligence.stage2.chunking import DocumentChunker
from document_intelligence.stage2.schema import (
    ContentBlock,
    DocumentContent,
    DocumentMetadata,
    Section,
    SourceInfo,
    StructuredDocument,
)
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


def test_chunking_merges_short_label_with_following_content():
    doc = StructuredDocument(
        document_id="doc_001",
        document_type="resume",
        confidence=1.0,
        source=SourceInfo(packet_id="pkt", source_file="test.pdf", page_start=1, page_end=1),
        content=DocumentContent(
            title="Resume",
            sections=[
                Section(
                    heading="Resume",
                    page=1,
                    blocks=[
                        ContentBlock(type="paragraph", text="Skills:", page=1),
                        ContentBlock(
                            type="list",
                            text="Python\nMachine Learning",
                            page=1,
                            items=["Python", "Machine Learning"],
                        ),
                    ],
                )
            ],
        ),
        metadata=DocumentMetadata(
            packet_id="pkt",
            document_id="doc_001",
            document_type="resume",
            source_file="test.pdf",
            page_start=1,
            page_end=1,
            page_count=1,
        ),
    )

    chunks = DocumentChunker(max_chars=500).chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Skills:\nPython\nMachine Learning"
    assert chunks[0].metadata["block_type"] == "merged"
