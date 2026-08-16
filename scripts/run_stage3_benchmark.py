#!/usr/bin/env python3
"""Run a real Stage 3 retrieval benchmark against the sample PDF packet.

scripts/run_benchmarks.py's "retrieval" stage scores a hardcoded, fabricated
list of retrieved_ids/relevant_ids that never touches your actual index or
pipeline. This script instead:

  1. Builds the sample packet's index (same as scripts/build_index.py)
  2. Runs a small set of real queries through the real pipeline
  3. Compares the real returned chunk_ids against hand-labeled ground-truth
     chunk_ids (verified against outputs/samples/structured/*.json)
  4. Scores with the same evaluate_retrieval() used elsewhere in this repo

Ground truth was built by inspecting the actual structured chunks in
outputs/samples/structured/*.json — see GROUND_TRUTH below. Add more
query -> relevant_chunk_ids pairs there to broaden coverage.

Run with --use-reranker / --no-reranker to compare both retrieval modes,
same flags as scripts/query.py.
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
from document_intelligence.stage2.schema import StructuredDocument
from document_intelligence.utils.io import read_json

# Query -> set of chunk_ids that count as a correct/relevant answer.
# Verified by hand against outputs/samples/structured/*.json content.
GROUND_TRUTH: list[dict] = [
    {
        "query": "What is the total amount on the invoice?",
        "relevant_ids": ["sample_packet_doc_001_chunk_013"],  # "Total Amount: ₹52,340"
    },
    {
        "query": "What is the invoice number?",
        "relevant_ids": ["sample_packet_doc_001_chunk_004"],  # "Invoice #INV-2024-001"
    },
    {
        "query": "What skills does the candidate have?",
        "relevant_ids": ["sample_packet_doc_002_chunk_011"],  # "Python, Machine Learning, FastAPI"
    },
    {
        "query": "What is the candidate's most recent job title?",
        "relevant_ids": ["sample_packet_doc_002_chunk_005"],  # "Senior Developer at TechCo (2020-2024)"
    },
    {
        "query": "What is the passport holder's date of birth?",
        "relevant_ids": ["sample_packet_doc_003_chunk_008"],  # "Date of Birth: 01 JAN 1990"
    },
    {
        "query": "What is the passport number?",
        "relevant_ids": ["sample_packet_doc_003_chunk_004"],  # "Passport No: AB1234567"
    },
]


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

    settings = get_settings()
    structured_dir = settings.outputs_dir / "samples" / "structured"
    docs = [StructuredDocument(**read_json(f)) for f in sorted(structured_dir.glob("*.json"))]
    if not docs:
        print(f"No structured documents found in {structured_dir}. Run scripts/generate_sample_outputs.py first.")
        sys.exit(1)

    pipeline = DocumentIntelligencePipeline()

    print(f"Building index from {len(docs)} structured documents ...")
    index_stats = pipeline.build_index(docs)

    print(f"Running {len(GROUND_TRUTH)} labeled queries (use_reranker={args.use_reranker}) ...")
    per_query_results = []
    eval_queries = []
    for item in GROUND_TRUTH:
        result = pipeline.query(item["query"], top_k=args.top_k, use_reranker=args.use_reranker)
        retrieved_ids = [r["chunk_id"] for r in result["results"]]
        per_query_results.append(
            {
                "query": item["query"],
                "relevant_ids": item["relevant_ids"],
                "retrieved_ids": retrieved_ids,
                "top1_correct": bool(retrieved_ids) and retrieved_ids[0] in item["relevant_ids"],
                "latency_seconds": result["latency_seconds"],
            }
        )
        eval_queries.append(
            {
                "retrieved_ids": retrieved_ids,
                "relevant_ids": item["relevant_ids"],
                "latency": result["latency_seconds"],
            }
        )

    runner = BenchmarkRunner()
    metrics_results = runner.run_retrieval_benchmark(eval_queries, indexing_time=index_stats["indexing_time_seconds"])
    metrics_results["use_reranker"] = args.use_reranker
    metrics_results["per_query"] = per_query_results
    metrics_results["notes"] = [
        "Ground-truth relevant_ids were hand-labeled against outputs/samples/structured/*.json "
        "content, not sourced from a labeled dataset (the page-stream datasets provide boundary "
        "labels only, no retrieval query/evidence pairs).",
        f"{len(GROUND_TRUTH)} queries across all 3 sample document types (invoice, resume, passport).",
    ]
    runner.save_results("retrieval_real", metrics_results)

    print(json.dumps(metrics_results, indent=2, default=str))
    print(f"\nResults saved to {runner.output_dir / 'retrieval_real.json'}")

    correct = sum(1 for r in per_query_results if r["top1_correct"])
    print(f"\nTop-1 accuracy: {correct}/{len(per_query_results)}")
    for r in per_query_results:
        status = "OK" if r["top1_correct"] else "MISS"
        top1 = r["retrieved_ids"][0] if r["retrieved_ids"] else "(none)"
        print(f"  [{status}] {r['query']!r} -> top1={top1}")


if __name__ == "__main__":
    main()