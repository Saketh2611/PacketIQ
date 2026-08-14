"""Load PDF packets safely."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz

from document_intelligence.utils.hashing import hash_file, make_page_id
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded."""


@dataclass
class PDFPageRef:
    page_number: int
    page_id: str
    width: float
    height: float


@dataclass
class PDFPacket:
    packet_id: str
    source_file: str
    page_count: int
    pages: list[PDFPageRef] = field(default_factory=list)
    file_hash: str = ""


class PDFLoader:
    """Open and iterate over PDF pages with deterministic IDs."""

    def __init__(self, source_path: Path | str, packet_id: str | None = None) -> None:
        self.source_path = Path(source_path)
        self.packet_id = packet_id or self.source_path.stem

    def load(self) -> tuple[fitz.Document, PDFPacket]:
        if not self.source_path.exists():
            raise PDFLoadError(f"PDF not found: {self.source_path}")

        try:
            doc = fitz.open(str(self.source_path))
        except Exception as exc:
            raise PDFLoadError(f"Failed to open PDF: {exc}") from exc

        if doc.is_encrypted:
            try:
                doc.authenticate("")
            except Exception:
                logger.warning("Encrypted PDF; authentication may be required", packet_id=self.packet_id)

        if doc.page_count == 0:
            raise PDFLoadError("PDF contains no pages")

        pages: list[PDFPageRef] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            rect = page.rect
            pages.append(
                PDFPageRef(
                    page_number=i + 1,
                    page_id=make_page_id(self.packet_id, i + 1),
                    width=rect.width,
                    height=rect.height,
                )
            )

        packet = PDFPacket(
            packet_id=self.packet_id,
            source_file=str(self.source_path.name),
            page_count=doc.page_count,
            pages=pages,
            file_hash=hash_file(self.source_path),
        )
        logger.info(
            "Loaded PDF packet",
            packet_id=packet.packet_id,
            page_count=packet.page_count,
        )
        return doc, packet

    def iter_page_numbers(self, doc: fitz.Document) -> Iterator[int]:
        for i in range(doc.page_count):
            yield i + 1
