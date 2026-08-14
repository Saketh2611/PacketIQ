"""Complete benchmark runner."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from document_intelligence.config.settings import get_settings
from document_intelligence.evaluation.resource_metrics import measure_resources
from document_intelligence.evaluation.retrieval_metrics import evaluate_retrieval
from document_intelligence.evaluation.stage1_metrics import evaluate_stage1
from document_intelligence.evaluation.stage2_metrics import evaluate_stage2
from document_intelligence.utils.logging import get_logger
from document_intelligence.utils.timing import timer

logger = get_logger(__name__)


class BenchmarkRunner:
    """Run and persist benchmark results."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.outputs_dir / "benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, name: str, results: dict[str, Any]) -> Path:
        path = self.output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved benchmark", path=str(path))
        return path

    def run_stage1_benchmark(self, eval_data: list[dict]) -> dict[str, Any]:
        results: dict[str, Any] = {"methods": {}}
        with timer("stage1_benchmark") as t:
            for method_name, samples in eval_data.items():
                all_metrics = []
                for sample in samples:
                    m = evaluate_stage1(
                        sample["true_boundary_pairs"],
                        sample["pred_boundary_pairs"],
                        sample["true_groups"],
                        sample["pred_groups"],
                        sample.get("true_types", []),
                        sample.get("pred_types", []),
                    )
                    all_metrics.append(asdict(m))
                if all_metrics:
                    avg = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))}
                    results["methods"][method_name] = avg
        results["resources"] = asdict(measure_resources(t.elapsed_seconds))
        self.save_results("stage1", results)
        return results

    def run_stage2_benchmark(self, documents: list, processing_time: float) -> dict[str, Any]:
        with timer("stage2_benchmark") as t:
            metrics = evaluate_stage2(documents, processing_time)
        results = {"metrics": asdict(metrics), "resources": asdict(measure_resources(t.elapsed_seconds))}
        self.save_results("stage2", results)
        return results

    def run_retrieval_benchmark(self, queries: list[dict], indexing_time: float = 0.0) -> dict[str, Any]:
        with timer("retrieval_benchmark") as t:
            metrics = evaluate_retrieval(queries, indexing_time)
        results = {"metrics": asdict(metrics), "resources": asdict(measure_resources(t.elapsed_seconds))}
        self.save_results("retrieval", results)
        return results
