#!/usr/bin/env python3
"""Run Stage 1 boundary detection and classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 analysis")
    parser.add_argument("--input", required=True, help="Path to PDF packet")
    parser.add_argument("--output", default="outputs/stage1.json", help="Output JSON path")
    parser.add_argument("--method", default="baseline", choices=["baseline", "embedding", "learned"])
    args = parser.parse_args()

    pipeline = DocumentIntelligencePipeline()
    result = pipeline.run_stage1(args.input, method=args.method)
    write_json(args.output, result)
    print(f"Stage 1 complete: {len(result['documents'])} documents detected")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
