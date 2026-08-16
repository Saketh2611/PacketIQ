"""Package console entry point for document intelligence workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: str | Path, payload: Any) -> None:
    from document_intelligence.utils.io import write_json

    write_json(path, payload)


def _run_stage1(args: argparse.Namespace) -> None:
    from document_intelligence.pipeline import DocumentIntelligencePipeline

    pipeline = DocumentIntelligencePipeline()
    result = pipeline.run_stage1(args.input, method=args.method)
    _write_json(args.output, result)
    print(f"Stage 1 complete: {len(result['documents'])} documents detected")
    print(f"Output saved to {args.output}")


def _run_stage2(args: argparse.Namespace) -> None:
    from document_intelligence.pipeline import DocumentIntelligencePipeline
    from document_intelligence.utils.io import read_json

    pipeline = DocumentIntelligencePipeline()
    stage1 = read_json(args.stage1)
    docs = pipeline.run_stage2(stage1, args.pdf)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        _write_json(out_dir / f"{doc.document_id}.json", doc.model_dump())
    print(f"Stage 2 complete: {len(docs)} structured documents saved to {out_dir}")


def _run_index(args: argparse.Namespace) -> None:
    from document_intelligence.pipeline import DocumentIntelligencePipeline
    from document_intelligence.stage2.schema import StructuredDocument
    from document_intelligence.utils.io import read_json

    pipeline = DocumentIntelligencePipeline()
    docs = [StructuredDocument(**read_json(path)) for path in sorted(Path(args.structured_dir).glob("*.json"))]
    stats = pipeline.build_index(docs)
    print(json.dumps(stats, indent=2))


def _run_query(args: argparse.Namespace) -> None:
    from document_intelligence.pipeline import DocumentIntelligencePipeline

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


def _add_reranker_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=None,
        help="Force the cross-encoder reranker on",
    )
    parser.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Force the cross-encoder reranker off",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-intel", description="Document intelligence command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1", help="Run Stage 1 boundary detection and classification")
    stage1.add_argument("--input", required=True, help="Path to PDF packet")
    stage1.add_argument("--output", default="outputs/stage1.json", help="Output JSON path")
    stage1.add_argument("--method", default="baseline", choices=["baseline", "embedding", "learned"])
    stage1.set_defaults(func=_run_stage1)

    stage2 = subparsers.add_parser("stage2", help="Run Stage 2 structured extraction")
    stage2.add_argument("--stage1", required=True, help="Stage 1 output JSON")
    stage2.add_argument("--pdf", required=True, help="Original PDF path")
    stage2.add_argument("--output", default="outputs/structured/", help="Output directory")
    stage2.set_defaults(func=_run_stage2)

    index = subparsers.add_parser("index", help="Build a retrieval index from structured JSON files")
    index.add_argument("--structured-dir", required=True, help="Directory containing StructuredDocument JSON files")
    index.set_defaults(func=_run_index)

    query = subparsers.add_parser("query", help="Retrieve evidence from an index")
    query.add_argument("--query", required=True, help="Search query")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--index", default=None, help="Index path")
    query.add_argument("--document-type", default=None, help="Filter results to this document type")
    _add_reranker_flags(query)
    query.set_defaults(func=_run_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
