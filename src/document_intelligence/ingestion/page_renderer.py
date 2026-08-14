"""Render PDF pages to images when visual/OCR processing is needed."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import fitz
from PIL import Image

from document_intelligence.config.settings import get_settings

if TYPE_CHECKING:
    pass


class PageRenderer:
    """Render pages to PIL images at configurable DPI."""

    def __init__(self, dpi: int | None = None) -> None:
        settings = get_settings()
        self.dpi = dpi or settings.render_dpi

    def render_page(self, doc: fitz.Document, page_number: int) -> Image.Image:
        page = doc.load_page(page_number - 1)
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        return img.convert("RGB")

    def should_render(self, text_length: int, force: bool = False) -> bool:
        if force:
            return True
        settings = get_settings()
        return text_length < settings.ocr_min_text_length
