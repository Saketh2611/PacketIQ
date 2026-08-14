"""Tests for document classification."""

from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.stage1.document_classifier import DocumentClassifier, HeuristicDocumentClassifier


def test_invoice_classification():
    clf = HeuristicDocumentClassifier()
    result = clf.classify_text("INVOICE\nBill To: Acme\nTotal Amount: 5000\nSubtotal: 4500")
    assert result.document_type == "invoice"
    assert result.confidence > 0


def test_unknown_classification():
    clf = HeuristicDocumentClassifier()
    result = clf.classify_text("Random text with no patterns")
    assert result.document_type == "unknown"
    assert result.confidence == 0.0


def test_low_confidence():
    pages = [PageRepresentation(page_id="p1", page_number=1, text="x")]
    clf = DocumentClassifier(use_embedding=False)
    result = clf.classify_pages(pages)
    assert result.document_type == "unknown"
