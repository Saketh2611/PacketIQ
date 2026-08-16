"""Tests for structured output."""

from document_intelligence.stage1.document_classifier import ClassificationResult
from document_intelligence.stage1.grouping import DocumentGroup
from document_intelligence.ingestion.page_extractor import PageRepresentation, TextBlock
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


def test_structuring_preserves_tables_and_figures():
    page = PageRepresentation(
        page_id="p1",
        page_number=1,
        text="Item Qty Amount\nLaptop 2 50000\nMouse 5 500",
        blocks=[
            TextBlock("Item          Qty    Amount", 72, 100, 260, 112),
            TextBlock("Laptop         2     50000", 72, 116, 260, 128),
            TextBlock("Mouse          5       500", 72, 132, 260, 144),
            TextBlock("Figure 1", 300, 100, 420, 220, block_type="image"),
        ],
    )
    extractor = StructureExtractor()
    group = DocumentGroup(document_id="doc_001", page_numbers=[1], page_start=1, page_end=1)
    classification = ClassificationResult(document_type="invoice", confidence=0.9)

    doc = extractor.extract("packet_001", "test.pdf", group, [page], classification)
    blocks = [block for section in doc.content.sections for block in section.blocks]

    table = next(block for block in blocks if block.type == "table")
    figure = next(block for block in blocks if block.type == "figure")
    assert table.headers == ["Item", "Qty", "Amount"]
    assert table.rows == [["Laptop", "2", "50000"], ["Mouse", "5", "500"]]
    assert figure.metadata["source_block_type"] == "image"
