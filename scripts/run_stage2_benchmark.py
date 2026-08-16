#!/usr/bin/env python3
"""Run a real Stage 2 structuring benchmark against the sample PDF packet.

Unlike scripts/run_benchmarks.py (which only wires up Stage 1 and a fake
Stage 3 smoke test), this script actually calls evaluate_stage2() on the
real pipeline output, and reports the real Stage2Metrics dataclass.

There is no labeled Stage 2 ground-truth dataset (the page-stream datasets
only provide boundary labels), so this runs against the bundled sample
packet — same scope as the "Stage 2" section already in docs/benchmark_report.md,
just now backed by an actual metrics computation instead of hand-written numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import asdict

from document_intelligence.config.settings import get_settings
from document_intelligence.evaluation.benchmark import BenchmarkRunner
from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.utils.timing import timer


def main() -> None:
    settings = get_settings()
    pdf_path = settings.outputs_dir / "samples" / "sample_packet.pdf"
    if not pdf_path.exists():
        print(f"Sample packet not found at {pdf_path}. Run scripts/generate_sample_outputs.py first.")
        sys.exit(1)

    pipeline = DocumentIntelligencePipeline()

    print(f"Running Stage 1 on {pdf_path} ...")
    stage1 = pipeline.run_stage1(pdf_path)

    print(f"Running Stage 2 on {len(stage1['documents'])} detected documents ...")
    with timer("stage2_structuring") as t:
        structured_docs = pipeline.run_stage2(stage1, pdf_path)

    runner = BenchmarkRunner()
    results = runner.run_stage2_benchmark(structured_docs, t.elapsed_seconds)
    results["source_pdf"] = str(pdf_path)
    results["documents"] = [
        {
            "document_id": d.document_id,
            "document_type": d.document_type,
            "confidence": d.confidence,
            "page_start": d.source.page_start,
            "page_end": d.source.page_end,
            "sections": len(d.content.sections),
        }
        for d in structured_docs
    ]
    runner.save_results("stage2", results)

    print(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {runner.output_dir / 'stage2.json'}")


if __name__ == "__main__":
    main()