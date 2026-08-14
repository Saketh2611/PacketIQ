#!/usr/bin/env python3
"""Build vector index from structured documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.stage2.schema import StructuredDocument
from document_intelligence.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument("--structured", default="outputs/structured/", help="Structured docs directory")
    args = parser.parse_args()

    structured_dir = Path(args.structured)
    docs = [StructuredDocument(**read_json(f)) for f in structured_dir.glob("*.json")]
    if not docs:
        print(f"No structured documents found in {structured_dir}")
        sys.exit(1)

    pipeline = DocumentIntelligencePipeline()
    stats = pipeline.build_index(docs)
    print(f"Indexed {stats['chunks_indexed']} chunks")
    print(f"Index saved to {stats['index_path']}")


if __name__ == "__main__":
    main()
