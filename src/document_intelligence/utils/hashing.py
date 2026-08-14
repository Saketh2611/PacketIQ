"""Content hashing for reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_page_id(packet_id: str, page_number: int) -> str:
    return f"{packet_id}_page_{page_number:04d}"


def make_document_id(packet_id: str, index: int) -> str:
    return f"{packet_id}_doc_{index:03d}"


def make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}_chunk_{index:03d}"
