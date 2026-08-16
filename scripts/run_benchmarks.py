#!/usr/bin/env python3
"""Run benchmarks on real page-stream datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings
from document_intelligence.dataset.adapter import DocSplitAdapter
from document_intelligence.evaluation.benchmark import BenchmarkRunner
from document_intelligence.evaluation.dataset_eval import (
    build_training_matrix,
    evaluate_boundary_method,
    iter_stream_samples,
)
from document_intelligence.evaluation.resource_metrics import measure_resources
from document_intelligence.stage1.boundary_classifier import BoundaryClassifier
from document_intelligence.stage1.page_features import PageFeatureBuilder
from document_intelligence.stage3.embeddings import EmbeddingModel
from document_intelligence.utils.timing import timer
from run_stage2_benchmark import run_stage2_sample_benchmark
from run_stage3_benchmark import run_stage3_real_benchmark


def load_rows(adapter: DocSplitAdapter, split: str) -> list[dict]:
    ds = adapter.load_dataset(split)
    return [ds[i] for i in range(len(ds))]


def run_dataset_stage1_benchmarks(runner: BenchmarkRunner, max_train_streams: int | None = None) -> dict:
    settings = get_settings()
    results: dict = {
        "train_dataset": settings.dataset_name,
        "train_config": settings.dataset_config,
        "test_dataset": settings.test_dataset_name,
        "test_config": settings.test_dataset_config,
        "methods": {},
        "notes": [
            "Train/dev source: nutrientdocs/openpss-mirror (OpenPSS community mirror).",
            "Test/eval source: nutrientdocs/doc-split-benchmark (official evaluation slice).",
            "nutrientdocs/doc-split-v2 is referenced by the assignment but was not accessible on HuggingFace Hub.",
            "Embedding model instance is reused across Stage 1 benchmark phases to avoid repeated model initialization.",
            "Classifier scaler is fit after the train/validation split using training features only.",
            "Document-type classification accuracy is not reported because these datasets provide boundary labels only.",
        ],
    }

    train_adapter = DocSplitAdapter(
        dataset_name=settings.dataset_name,
        dataset_config=settings.dataset_config,
    )
    test_adapter = DocSplitAdapter(
        dataset_name=settings.test_dataset_name,
        dataset_config=settings.test_dataset_config,
    )

    with timer("load_train") as load_train_t:
        train_rows = load_rows(train_adapter, "train")
    with timer("load_test") as load_test_t:
        test_rows = load_rows(test_adapter, "test")

    test_samples = list(iter_stream_samples(test_adapter, test_rows))
    shared_embedder = EmbeddingModel()
    test_feature_builder = PageFeatureBuilder(embedder=shared_embedder)
    train_feature_builder = PageFeatureBuilder(embedder=shared_embedder)
    results["dataset_stats"] = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_streams": len(test_samples),
        "load_train_seconds": load_train_t.elapsed_seconds,
        "load_test_seconds": load_test_t.elapsed_seconds,
    }

    with timer("baseline_rule") as t_rule:
        results["methods"]["baseline_rule"] = evaluate_boundary_method(
            test_samples,
            "baseline_rule",
            feature_builder=test_feature_builder,
        )
    results["methods"]["baseline_rule"]["latency_seconds"] = t_rule.elapsed_seconds

    with timer("baseline_embedding") as t_emb:
        results["methods"]["baseline_embedding"] = evaluate_boundary_method(
            test_samples,
            "baseline_embedding",
            feature_builder=test_feature_builder,
        )
    results["methods"]["baseline_embedding"]["latency_seconds"] = t_emb.elapsed_seconds

    train_stream_rows = train_rows
    if max_train_streams is not None:
        grouped: dict[str, list[dict]] = {}
        for row in train_rows:
            grouped.setdefault(str(row["stream_id"]), []).append(row)
        selected = list(grouped.keys())[:max_train_streams]
        train_stream_rows = [row for sid in selected for row in grouped[sid]]
        results["dataset_stats"]["train_rows_used_for_classifier"] = len(train_stream_rows)
        results["dataset_stats"]["train_streams_used_for_classifier"] = len(selected)

    with timer("train_classifier") as t_train:
        X, y = build_training_matrix(train_adapter, train_stream_rows, feature_builder=train_feature_builder)
        classifier = BoundaryClassifier()
        train_result = classifier.train(X, y, val_size=0.2)
        model_path = settings.models_dir / "boundary_classifier.joblib"
        classifier.save(model_path)
    results["classifier_training"] = {
        "train_size": train_result.train_size,
        "val_size": train_result.val_size,
        "model_path": str(model_path),
        "feature_importance": classifier.feature_importance(),
        "latency_seconds": t_train.elapsed_seconds,
    }

    with timer("learned_classifier") as t_learned:
        results["methods"]["learned_classifier"] = evaluate_boundary_method(
            test_samples,
            "learned",
            classifier=classifier,
            feature_builder=test_feature_builder,
        )
    results["methods"]["learned_classifier"]["latency_seconds"] = t_learned.elapsed_seconds

    results["resources"] = asdict(
        measure_resources(
            t_rule.elapsed_seconds + t_emb.elapsed_seconds + t_train.elapsed_seconds + t_learned.elapsed_seconds,
            model_cache_dir=settings.models_dir,
        )
    )
    runner.save_results("stage1_dataset", results)
    return results


def _stage3_modes(stage3_mode: str, use_reranker: bool | None) -> list[bool | None]:
    if use_reranker is not None:
        return [use_reranker]
    if stage3_mode == "both":
        return [False, True]
    if stage3_mode == "vector":
        return [False]
    if stage3_mode == "reranker":
        return [True]
    return [None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-backed benchmarks")
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "stage2", "stage3", "retrieval"])
    parser.add_argument("--max-train-streams", type=int, default=None, help="Limit train streams for faster runs")
    parser.add_argument("--top-k", type=int, default=5, help="Stage 3 retrieval depth")
    parser.add_argument(
        "--stage3-mode",
        default="both",
        choices=["config", "vector", "reranker", "both"],
        help="Stage 3 reranker mode. Ignored when --use-reranker/--no-reranker is set.",
    )
    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=None,
        help="Force the Stage 3 reranker on",
    )
    parser.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Force the Stage 3 reranker off",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner()
    summary: dict[str, dict] = {}
    if args.stage in ("all", "stage1"):
        summary["stage1"] = run_dataset_stage1_benchmarks(runner, max_train_streams=args.max_train_streams)
    if args.stage in ("all", "stage2"):
        summary["stage2"] = run_stage2_sample_benchmark(runner)
    if args.stage in ("all", "stage3", "retrieval"):
        stage3_results: dict[str, dict] = {}
        for mode in _stage3_modes(args.stage3_mode, args.use_reranker):
            result = run_stage3_real_benchmark(runner, use_reranker=mode, top_k=args.top_k)
            stage3_results[result["output_name"]] = result
        summary["stage3"] = stage3_results
    print(json.dumps(summary, indent=2, default=str))
    print(f"Results saved to {runner.output_dir}")


if __name__ == "__main__":
    main()
