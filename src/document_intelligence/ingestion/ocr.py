"""OCR fallback for scanned/image-only pages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_intelligence.config.settings import get_settings
from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.utils.logging import get_logger

if TYPE_CHECKING:
    import fitz
    from PIL import Image

logger = get_logger(__name__)


class OCREngine:
    """Configurable OCR engine with pytesseract fallback."""

    def __init__(self, engine: str | None = None) -> None:
        settings = get_settings()
        self.engine = engine or settings.ocr_engine

    def needs_ocr(self, page: PageRepresentation) -> bool:
        settings = get_settings()
        if self.engine == "none":
            return False
        return page.text_length < settings.ocr_min_text_length or "empty_native_text" in page.warnings

    def run_ocr(self, image: Image.Image, page: PageRepresentation) -> PageRepresentation:
        if self.engine == "none":
            page.warnings.append("ocr_disabled")
            return page

        try:
            if self.engine == "pytesseract":
                import pytesseract

                text = pytesseract.image_to_string(image)
                conf_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [c for c in conf_data.get("conf", []) if isinstance(c, (int, float)) and c >= 0]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            else:
                page.warnings.append(f"unknown_ocr_engine: {self.engine}")
                return page
        except Exception as exc:
            logger.warning("OCR failed", page_id=page.page_id, error=str(exc))
            page.warnings.append(f"ocr_failed: {exc}")
            return page

        if text.strip():
            page.text = text.strip()
            page.text_length = len(page.text)
            page.extraction_method = "ocr"
            page.metadata["ocr_confidence"] = avg_conf
        else:
            page.warnings.append("ocr_empty_result")

        return page


def enrich_page_with_ocr(
    doc: fitz.Document,
    page: PageRepresentation,
    renderer: object | None = None,
) -> PageRepresentation:
    """Run OCR on page if native extraction is insufficient."""
    from document_intelligence.ingestion.page_renderer import PageRenderer

    engine = OCREngine()
    if not engine.needs_ocr(page):
        return page

    render = renderer or PageRenderer()
    image = render.render_page(doc, page.page_number)
    return engine.run_ocr(image, page)
