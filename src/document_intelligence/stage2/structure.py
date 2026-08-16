"""Convert grouped documents to structured JSON."""

from __future__ import annotations

import re

from document_intelligence.ingestion.page_extractor import PageRepresentation, TextBlock
from document_intelligence.stage1.document_classifier import ClassificationResult
from document_intelligence.stage1.grouping import DocumentGroup
from document_intelligence.stage2.metadata import build_document_metadata
from document_intelligence.stage2.schema import (
    ContentBlock,
    DocumentContent,
    Section,
    SourceInfo,
    StructuredDocument,
)


def _detect_list_items(text: str) -> list[str] | None:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    bullet_lines = [ln for ln in lines if _is_list_item_line(ln)]
    if len(bullet_lines) >= 2:
        return bullet_lines
    return None


def _is_list_item_line(text: str) -> bool:
    return bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)])\s+", text.strip()))


def _split_table_columns(text: str) -> list[str] | None:
    stripped = text.strip().strip("|")
    if not stripped or not re.search(r"\s{2,}|\t|\|", stripped):
        return None
    columns = [col.strip() for col in re.split(r"\s{2,}|\t+|\|", stripped) if col.strip()]
    if len(columns) < 2:
        return None
    return columns


def _normalize_table_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    return normalized[0], normalized[1:]


def _bbox_metadata(block: TextBlock) -> dict:
    return {
        "bbox": [block.x0, block.y0, block.x1, block.y1],
        "source_block_type": block.block_type,
    }


def _text_block_to_content_block(block: TextBlock, page_number: int) -> ContentBlock:
    if block.block_type == "image":
        return ContentBlock(
            type="figure",
            text=block.text or f"Figure on page {page_number}",
            page=page_number,
            metadata=_bbox_metadata(block),
        )
    if block.is_heading:
        return ContentBlock(type="heading", text=block.text, page=page_number, metadata=_bbox_metadata(block))
    items = _detect_list_items(block.text)
    if items:
        return ContentBlock(type="list", text=block.text, page=page_number, items=items, metadata=_bbox_metadata(block))
    return ContentBlock(type="paragraph", text=block.text, page=page_number, metadata=_bbox_metadata(block))


def _blocks_to_content_blocks(blocks: list[TextBlock], page_number: int) -> list[ContentBlock]:
    content: list[ContentBlock] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.block_type == "image":
            content.append(_text_block_to_content_block(block, page_number))
            i += 1
            continue

        table_rows: list[list[str]] = []
        table_blocks: list[TextBlock] = []
        j = i
        while j < len(blocks):
            candidate = blocks[j]
            if candidate.block_type == "image" or candidate.is_heading:
                break
            columns = _split_table_columns(candidate.text)
            if columns is None:
                break
            table_rows.append(columns)
            table_blocks.append(candidate)
            j += 1

        if len(table_rows) >= 2:
            headers, rows = _normalize_table_rows(table_rows)
            text = "\n".join(" | ".join(row) for row in [headers, *rows])
            content.append(
                ContentBlock(
                    type="table",
                    text=text,
                    page=page_number,
                    headers=headers,
                    rows=rows,
                    metadata={
                        "bbox": [
                            min(b.x0 for b in table_blocks),
                            min(b.y0 for b in table_blocks),
                            max(b.x1 for b in table_blocks),
                            max(b.y1 for b in table_blocks),
                        ],
                        "source_block_count": len(table_blocks),
                    },
                )
            )
            i = j
            continue

        if _is_list_item_line(block.text):
            list_blocks = [block]
            j = i + 1
            while j < len(blocks) and _is_list_item_line(blocks[j].text):
                list_blocks.append(blocks[j])
                j += 1
            items = [b.text.strip() for b in list_blocks]
            content.append(
                ContentBlock(
                    type="list",
                    text="\n".join(items),
                    page=page_number,
                    items=items,
                    metadata={
                        "bbox": [
                            min(b.x0 for b in list_blocks),
                            min(b.y0 for b in list_blocks),
                            max(b.x1 for b in list_blocks),
                            max(b.y1 for b in list_blocks),
                        ],
                        "source_block_count": len(list_blocks),
                    },
                )
            )
            i = j
            continue

        content.append(_text_block_to_content_block(block, page_number))
        i += 1
    return content


def _group_blocks_into_sections(pages: list[PageRepresentation]) -> DocumentContent:
    sections: list[Section] = []
    current_section: Section | None = None

    for page in pages:
        page_blocks = _blocks_to_content_blocks(page.blocks, page.page_number)
        if not page_blocks and page.text.strip():
            page_blocks = [ContentBlock(type="paragraph", text=page.text.strip(), page=page.page_number)]

        for block in page_blocks:
            if block.type == "heading":
                if current_section and current_section.blocks:
                    sections.append(current_section)
                current_section = Section(heading=block.text, page=block.page, blocks=[])
            else:
                if current_section is None:
                    current_section = Section(heading=None, page=block.page, blocks=[])
                current_section.blocks.append(block)

    if current_section and (current_section.blocks or current_section.heading):
        sections.append(current_section)

    title = None
    if sections and sections[0].heading:
        title = sections[0].heading
    elif pages and pages[0].text.strip():
        first_line = pages[0].text.strip().split("\n")[0][:200]
        title = first_line

    return DocumentContent(title=title, sections=sections)


class StructureExtractor:
    """Convert page groups to structured documents."""

    def extract(
        self,
        packet_id: str,
        source_file: str,
        group: DocumentGroup,
        pages: list[PageRepresentation],
        classification: ClassificationResult,
    ) -> StructuredDocument:
        content = _group_blocks_into_sections(pages)
        metadata = build_document_metadata(packet_id, group, classification, pages, source_file)

        return StructuredDocument(
            document_id=group.document_id,
            document_type=classification.document_type,
            confidence=classification.confidence,
            source=SourceInfo(
                packet_id=packet_id,
                source_file=source_file,
                page_start=group.page_start,
                page_end=group.page_end,
            ),
            content=content,
            metadata=metadata,
        )
