"""Structural/semantic chunking for retrieval."""

from __future__ import annotations

from document_intelligence.config.settings import get_settings
from document_intelligence.stage2.schema import Chunk, ContentBlock, StructuredDocument
from document_intelligence.utils.hashing import make_chunk_id


class DocumentChunker:
    """Chunk structured documents preserving section boundaries and provenance."""

    def __init__(self, max_chars: int | None = None, overlap: int | None = None) -> None:
        settings = get_settings()
        self.max_chars = max_chars or settings.max_chunk_chars
        self.overlap = overlap or settings.chunk_overlap

    def _block_text(self, block: ContentBlock) -> str:
        if block.type == "table" and block.headers and block.rows:
            header = " | ".join(block.headers)
            rows = [" | ".join(r) for r in block.rows]
            return f"{header}\n" + "\n".join(rows)
        if block.type == "list" and block.items:
            return "\n".join(block.items)
        return block.text

    def _split_long_text(self, text: str, page: int, section: str | None) -> list[tuple[str, int]]:
        if len(text) <= self.max_chars:
            return [(text, page)]

        parts: list[tuple[str, int]] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            parts.append((text[start:end], page))
            if end >= len(text):
                break
            start = end - self.overlap
        return parts

    def chunk_document(self, doc: StructuredDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for section in doc.content.sections:
            section_name = section.heading
            for block in section.blocks:
                text = self._block_text(block)
                if not text.strip():
                    continue

                if block.type == "table":
                    subparts = [(text, block.page)]
                else:
                    subparts = self._split_long_text(text, block.page, section_name)

                for part_text, page in subparts:
                    chunk_idx += 1
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(doc.document_id, chunk_idx),
                            document_id=doc.document_id,
                            document_type=doc.document_type,
                            page_start=page,
                            page_end=page,
                            section=section_name,
                            text=part_text.strip(),
                            metadata={
                                "block_type": block.type,
                                "packet_id": doc.source.packet_id,
                                "source_file": doc.source.source_file,
                            },
                        )
                    )

        if not chunks and doc.content.title:
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.document_id, 1),
                    document_id=doc.document_id,
                    document_type=doc.document_type,
                    page_start=doc.source.page_start,
                    page_end=doc.source.page_end,
                    section=None,
                    text=doc.content.title,
                    metadata={"packet_id": doc.source.packet_id},
                )
            )

        return chunks
