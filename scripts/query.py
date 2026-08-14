#!/usr/bin/env python3
"""Query the evidence retrieval index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.pipeline import DocumentIntelligencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve evidence")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", default=None, help="Index path")
    args = parser.parse_args()

    pipeline = DocumentIntelligencePipeline()
    if args.index:
        pipeline.load_index(args.index)
    result = pipeline.query(args.query, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
