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
    parser.add_argument("--document-type", default=None, help="Filter results to this document type")
    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=None,
        help="Force the cross-encoder reranker on for this query, overriding USE_RERANKER in .env",
    )
    parser.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Force the reranker off for this query, overriding USE_RERANKER in .env",
    )
    args = parser.parse_args()

    pipeline = DocumentIntelligencePipeline()
    if args.index:
        pipeline.load_index(args.index)
    result = pipeline.query(
        args.query,
        top_k=args.top_k,
        use_reranker=args.use_reranker,
        document_type=args.document_type,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()