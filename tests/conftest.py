"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import fitz
import pytest

from document_intelligence.ingestion.page_extractor import PageRepresentation


# @pytest.fixture(scope="session", autouse=True)
# def use_hash_embeddings():
#     os.environ["USE_HASH_EMBEDDINGS"] = "1"
#     yield
#     os.environ.pop("USE_HASH_EMBEDDINGS", None)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for text in [
        "INVOICE\nBill To: Test Corp\nTotal: 1000",
        "INVOICE page 2\nSubtotal: 900\nTax: 100",
        "RESUME\nJohn Smith\nExperience: Engineer",
    ]:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    path = tmp_path / "test_packet.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pages() -> list[PageRepresentation]:
    return [
        PageRepresentation(page_id="p1", page_number=1, text="INVOICE Bill To Test Total 1000"),
        PageRepresentation(page_id="p2", page_number=2, text="INVOICE Subtotal 900 Tax 100"),
        PageRepresentation(page_id="p3", page_number=3, text="RESUME John Smith Experience Engineer"),
    ]
