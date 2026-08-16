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
        configured_overlap = overlap if overlap is not None else settings.chunk_overlap
        self.overlap = max(0, min(configured_overlap, self.max_chars - 1))

    def _block_text(self, block: ContentBlock) -> str:
        if block.type == "table" and block.headers and block.rows:
            header = " | ".join(block.headers)
            rows = [" | ".join(r) for r in block.rows]
            return f"{header}\n" + "\n".join(rows)
        if block.type == "list" and block.items:
            return "\n".join(block.items)
        if block.type == "figure":
            return block.text or "Figure"
        return block.text

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            parts.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.overlap
        return parts

    @staticmethod
    def _is_label_block(block: ContentBlock) -> bool:
        text = block.text.strip()
        return block.type == "paragraph" and "\n" not in text and text.endswith(":") and len(text) <= 80

    def _append_chunk(
        self,
        chunks: list[Chunk],
        doc: StructuredDocument,
        chunk_idx: int,
        text: str,
        page_start: int,
        page_end: int,
        section_name: str | None,
        metadata: dict,
    ) -> int:
        for part_text in self._split_long_text(text.strip()):
            chunk_idx += 1
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.document_id, chunk_idx),
                    document_id=doc.document_id,
                    document_type=doc.document_type,
                    page_start=page_start,
                    page_end=page_end,
                    section=section_name,
                    text=part_text.strip(),
                    metadata={
                        "packet_id": doc.source.packet_id,
                        "source_file": doc.source.source_file,
                        **metadata,
                    },
                )
            )
        return chunk_idx

    def chunk_document(self, doc: StructuredDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for section in doc.content.sections:
            section_name = section.heading
            block_idx = 0
            while block_idx < len(section.blocks):
                block = section.blocks[block_idx]
                text = self._block_text(block)
                if not text.strip():
                    block_idx += 1
                    continue

                metadata = {"block_type": block.type}
                page_start = block.page
                page_end = block.page

                if self._is_label_block(block) and block_idx + 1 < len(section.blocks):
                    next_block = section.blocks[block_idx + 1]
                    next_text = self._block_text(next_block)
                    if next_text.strip() and next_block.type not in {"table", "figure"}:
                        text = f"{text.strip()}\n{next_text.strip()}"
                        page_end = next_block.page
                        metadata = {
                            "block_type": "merged",
                            "block_types": [block.type, next_block.type],
                            "merged_label": block.text.strip(),
                        }
                        block_idx += 1

                if block.type == "table":
                    metadata.update(
                        {
                            "headers": block.headers or [],
                            "row_count": len(block.rows or []),
                        }
                    )
                elif block.type == "figure":
                    metadata.update(block.metadata)

                chunk_idx = self._append_chunk(
                    chunks,
                    doc,
                    chunk_idx,
                    text,
                    page_start,
                    page_end,
                    section_name,
                    metadata,
                )
                block_idx += 1

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
