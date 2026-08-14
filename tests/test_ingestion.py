"""Tests for PDF ingestion."""

import fitz
import pytest

from document_intelligence.ingestion.page_extractor import PageExtractor
from document_intelligence.ingestion.pdf_loader import PDFLoader, PDFLoadError


def test_pdf_page_count(sample_pdf):
    loader = PDFLoader(sample_pdf)
    doc, packet = loader.load()
    assert packet.page_count == 3
    assert len(packet.pages) == 3
    doc.close()


def test_native_text_extraction(sample_pdf):
    loader = PDFLoader(sample_pdf)
    doc, packet = loader.load()
    extractor = PageExtractor()
    page = extractor.extract(doc, 1, packet.pages[0].page_id)
    assert "INVOICE" in page.text
    assert page.text_length > 0
    doc.close()


def test_empty_pdf_raises(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF")
    loader = PDFLoader(path)
    with pytest.raises(PDFLoadError):
        loader.load()


def test_missing_pdf_raises(tmp_path):
    loader = PDFLoader(tmp_path / "missing.pdf")
    try:
        loader.load()
        assert False
    except PDFLoadError:
        pass
