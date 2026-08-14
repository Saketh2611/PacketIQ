"""Tests for OCR fallback behavior."""

from unittest.mock import MagicMock, patch

from document_intelligence.ingestion.ocr import OCREngine
from document_intelligence.ingestion.page_extractor import PageRepresentation
from PIL import Image


def test_needs_ocr_empty_page():
    engine = OCREngine(engine="pytesseract")
    page = PageRepresentation(page_id="p1", page_number=1, text="", warnings=["empty_native_text"])
    assert engine.needs_ocr(page) is True


def test_needs_ocr_sufficient_text():
    engine = OCREngine(engine="none")
    page = PageRepresentation(page_id="p1", page_number=1, text="A" * 100)
    assert engine.needs_ocr(page) is False


def test_ocr_disabled():
    engine = OCREngine(engine="none")
    page = PageRepresentation(page_id="p1", page_number=1, text="")
    img = Image.new("RGB", (100, 100))
    result = engine.run_ocr(img, page)
    assert "ocr_disabled" in result.warnings
