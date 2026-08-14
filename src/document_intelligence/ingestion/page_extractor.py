"""Extract page-level native content using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import fitz

from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextBlock:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_type: str = "text"
    font_size: float = 0.0
    is_heading: bool = False


@dataclass
class PageRepresentation:
    page_id: str
    page_number: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    text_length: int = 0
    block_count: int = 0
    image_count: int = 0
    extraction_method: str = "native"
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text_length = len(self.text.strip())
        self.block_count = len(self.blocks)


class PageExtractor:
    """Extract native PDF text and layout blocks."""

    HEADING_SIZE_THRESHOLD = 14.0

    def extract(self, doc: fitz.Document, page_number: int, page_id: str) -> PageRepresentation:
        try:
            page = doc.load_page(page_number - 1)
        except Exception as exc:
            logger.error("Failed to load page", page_id=page_id, error=str(exc))
            return PageRepresentation(
                page_id=page_id,
                page_number=page_number,
                text="",
                warnings=[f"corrupted_page: {exc}"],
            )

        rect = page.rect
        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        blocks: list[TextBlock] = []
        text_parts: list[str] = []

        for block in raw_blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = ""
                max_font = 0.0
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                    max_font = max(max_font, span.get("size", 0.0))
                line_text = line_text.strip()
                if not line_text:
                    continue
                bbox = line.get("bbox", block.get("bbox", [0, 0, 0, 0]))
                blocks.append(
                    TextBlock(
                        text=line_text,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        font_size=max_font,
                        is_heading=max_font >= self.HEADING_SIZE_THRESHOLD,
                    )
                )
                text_parts.append(line_text)

        text = "\n".join(text_parts)
        image_list = page.get_images(full=True)

        rep = PageRepresentation(
            page_id=page_id,
            page_number=page_number,
            text=text,
            blocks=blocks,
            width=rect.width,
            height=rect.height,
            image_count=len(image_list),
            extraction_method="native",
            metadata={"rotation": page.rotation},
        )

        if rep.text_length == 0:
            rep.warnings.append("empty_native_text")

        return rep
