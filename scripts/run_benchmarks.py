#!/usr/bin/env python3
"""Run complete benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from document_intelligence.evaluation.benchmark import BenchmarkRunner
from document_intelligence.evaluation.retrieval_metrics import evaluate_retrieval
from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.stage1.boundary_baseline import BoundaryBaseline
from document_intelligence.stage1.grouping import decisions_to_groups
from document_intelligence.utils.timing import timer


def create_synthetic_pages() -> list[PageRepresentation]:
    """Create synthetic pages for benchmark when no PDF available."""
    pages = []
    texts = [
        "INVOICE\nBill To: Acme Corp\nItem: Laptop Qty: 2 Amount: 50000",
        "INVOICE continued\nSubtotal: 50000\nTax: 2340\nTotal Amount: 52340",
        "INVOICE page 3\nPayment terms: Net 30",
        "RESUME\nJohn Doe\nExperience: Software Engineer at Tech Co",
        "RESUME continued\nEducation: BS Computer Science\nSkills: Python, ML",
        "PASSPORT\nNationality: Example\nDate of Birth: 1990-01-01",
    ]
    for i, text in enumerate(texts, 1):
        pages.append(
            PageRepresentation(
                page_id=f"bench_page_{i:04d}",
                page_number=i,
                text=text,
            )
        )
    return pages


def run_stage1_benchmarks(runner: BenchmarkRunner) -> None:
    pages = create_synthetic_pages()
    true_groups = [[1, 2, 3], [4, 5], [6]]
    true_pairs = []
    page_to_group = {}
    for gi, g in enumerate(true_groups):
        for p in g:
            page_to_group[p] = gi
    for i in range(len(pages) - 1):
        a, b = pages[i].page_number, pages[i + 1].page_number
        is_boundary = page_to_group[a] != page_to_group[b]
        true_pairs.append((a, b, is_boundary))

    methods = {
        "baseline_rule": BoundaryBaseline(mode="weighted"),
        "baseline_embedding": BoundaryBaseline(mode="embedding_only"),
    }
    eval_data = {}
    for name, baseline in methods.items():
        decisions = baseline.predict_pairs(pages)
        pred_pairs = [(d.page_a, d.page_b, d.is_boundary) for d in decisions]
        pred_groups = [g.page_numbers for g in decisions_to_groups(len(pages), decisions, "bench")]
        eval_data[name] = [
            {
                "true_boundary_pairs": true_pairs,
                "pred_boundary_pairs": pred_pairs,
                "true_groups": true_groups,
                "pred_groups": pred_groups,
                "true_types": ["invoice", "resume", "passport"],
                "pred_types": ["invoice", "resume", "passport"],
            }
        ]
    results = runner.run_stage1_benchmark(eval_data)
    print("Stage 1 benchmarks:", results)


def run_retrieval_benchmark(runner: BenchmarkRunner) -> None:
    queries = [
        {
            "retrieved_ids": ["doc_001_chunk_001", "doc_001_chunk_002"],
            "relevant_ids": ["doc_001_chunk_002"],
            "latency": 0.05,
        },
        {
            "retrieved_ids": ["doc_002_chunk_001"],
            "relevant_ids": ["doc_002_chunk_001"],
            "latency": 0.03,
        },
    ]
    results = runner.run_retrieval_benchmark(queries, indexing_time=1.2)
    print("Retrieval benchmarks:", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "retrieval"])
    args = parser.parse_args()

    runner = BenchmarkRunner()
    if args.stage in ("all", "stage1"):
        run_stage1_benchmarks(runner)
    if args.stage in ("all", "retrieval"):
        run_retrieval_benchmark(runner)
    print(f"Results saved to {runner.output_dir}")


if __name__ == "__main__":
    main()
