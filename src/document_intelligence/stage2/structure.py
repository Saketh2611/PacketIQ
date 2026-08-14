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
    bullet_lines = [ln for ln in lines if re.match(r"^[\-\*\u2022\d+\.]", ln)]
    if len(bullet_lines) >= 2:
        return bullet_lines
    return None


def _blocks_to_content_blocks(blocks: list[TextBlock], page_number: int) -> list[ContentBlock]:
    content: list[ContentBlock] = []
    for block in blocks:
        if block.is_heading:
            content.append(ContentBlock(type="heading", text=block.text, page=page_number))
        else:
            items = _detect_list_items(block.text)
            if items:
                content.append(ContentBlock(type="list", text=block.text, page=page_number, items=items))
            else:
                content.append(ContentBlock(type="paragraph", text=block.text, page=page_number))
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
