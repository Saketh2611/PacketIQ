#!/usr/bin/env python3
"""Run a real Stage 3 retrieval benchmark against the sample PDF packet.

The benchmark builds the sample packet index, runs labeled queries through the
real retrieval pipeline, and scores returned chunk_ids against answer text
patterns resolved from the current Stage 2 chunker output. The page-stream
datasets used for Stage 1 do not provide retrieval query/evidence labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings
from document_intelligence.evaluation.benchmark import BenchmarkRunner
from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.stage2.schema import Chunk, StructuredDocument
from document_intelligence.utils.io import read_json


GROUND_TRUTH: list[dict] = [
    {
        "query": "What is the total amount on the invoice?",
        "document_type": "invoice",
        "match_text": "Total Amount",
    },
    {
        "query": "What is the invoice number?",
        "document_type": "invoice",
        "match_text": "Invoice #INV-2024-001",
    },
    {
        "query": "What skills does the candidate have?",
        "document_type": "resume",
        "match_text": "Python, Machine Learning, FastAPI",
    },
    {
        "query": "What is the candidate's most recent job title?",
        "document_type": "resume",
        "match_text": "Senior Developer at TechCo",
    },
    {
        "query": "What is the passport holder's date of birth?",
        "document_type": "passport",
        "match_text": "Date of Birth: 01 JAN 1990",
    },
    {
        "query": "What is the passport number?",
        "document_type": "passport",
        "match_text": "Passport No: AB1234567",
    },
]


def _result_name(use_reranker: bool | None) -> str:
    if use_reranker is True:
        return "stage3_reranker"
    if use_reranker is False:
        return "stage3_vector"
    return "stage3"


def _resolve_relevant_ids(chunks: list[Chunk], item: dict) -> list[str]:
    match_text = item["match_text"].lower()
    doc_type = item["document_type"]
    matches = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.document_type == doc_type and match_text in chunk.text.lower()
    ]
    if not matches:
        raise ValueError(f"No relevant chunk matched {item['match_text']!r} for query {item['query']!r}")
    return matches


def run_stage3_real_benchmark(
    runner: BenchmarkRunner | None = None,
    use_reranker: bool | None = None,
    top_k: int = 5,
) -> dict:
    settings = get_settings()
    structured_dir = settings.outputs_dir / "samples" / "structured"
    docs = [StructuredDocument(**read_json(path)) for path in sorted(structured_dir.glob("*.json"))]
    if not docs:
        raise FileNotFoundError(
            f"No structured documents found in {structured_dir}. Run scripts/generate_sample_outputs.py first."
        )

    pipeline = DocumentIntelligencePipeline()
    chunks = [chunk for doc in docs for chunk in pipeline.chunker.chunk_document(doc)]

    print(f"Building index from {len(docs)} structured documents ...")
    index_stats = pipeline.build_index(docs)

    print(f"Running {len(GROUND_TRUTH)} labeled queries (use_reranker={use_reranker}) ...")
    per_query_results = []
    eval_queries = []
    for item in GROUND_TRUTH:
        relevant_ids = _resolve_relevant_ids(chunks, item)
        result = pipeline.query(item["query"], top_k=top_k, use_reranker=use_reranker)
        retrieved_ids = [r["chunk_id"] for r in result["results"]]
        per_query_results.append(
            {
                "query": item["query"],
                "document_type": item["document_type"],
                "match_text": item["match_text"],
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "top1_correct": bool(retrieved_ids) and retrieved_ids[0] in relevant_ids,
                "latency_seconds": result["latency_seconds"],
            }
        )
        eval_queries.append(
            {
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
                "latency": result["latency_seconds"],
            }
        )

    runner = runner or BenchmarkRunner()
    metrics_results = runner.run_retrieval_benchmark(eval_queries, indexing_time=index_stats["indexing_time_seconds"])
    output_name = _result_name(use_reranker)
    metrics_results["output_name"] = output_name
    metrics_results["use_reranker"] = use_reranker
    metrics_results["per_query"] = per_query_results
    metrics_results["notes"] = [
        "Ground-truth relevant_ids are resolved from answer text patterns verified against "
        "outputs/samples/structured/*.json content, not sourced from a labeled dataset.",
        f"{len(GROUND_TRUTH)} queries across all 3 sample document types (invoice, resume, passport).",
    ]
    runner.save_results(output_name, metrics_results)
    return metrics_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real Stage 3 retrieval benchmark")
    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=None,
        help="Force the cross-encoder reranker on for every query",
    )
    parser.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Force the reranker off for every query",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        metrics_results = run_stage3_real_benchmark(use_reranker=args.use_reranker, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        sys.exit(1)

    print(json.dumps(metrics_results, indent=2, default=str))
    output_name = metrics_results["output_name"]
    print(f"\nResults saved to {BenchmarkRunner().output_dir / f'{output_name}.json'}")

    correct = sum(1 for result in metrics_results["per_query"] if result["top1_correct"])
    print(f"\nTop-1 accuracy: {correct}/{len(metrics_results['per_query'])}")
    for result in metrics_results["per_query"]:
        status = "OK" if result["top1_correct"] else "MISS"
        top1 = result["retrieved_ids"][0] if result["retrieved_ids"] else "(none)"
        print(f"  [{status}] {result['query']!r} -> top1={top1}")


if __name__ == "__main__":
    main()
