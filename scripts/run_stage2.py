#!/usr/bin/env python3
"""Run Stage 2 structured document extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.pipeline import DocumentIntelligencePipeline
from document_intelligence.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 structuring")
    parser.add_argument("--stage1", required=True, help="Stage 1 output JSON")
    parser.add_argument("--pdf", required=True, help="Original PDF path")
    parser.add_argument("--output", default="outputs/structured/", help="Output directory")
    args = parser.parse_args()

    pipeline = DocumentIntelligencePipeline()
    stage1 = read_json(args.stage1)
    docs = pipeline.run_stage2(stage1, args.pdf)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        write_json(out_dir / f"{doc.document_id}.json", doc.model_dump())
    print(f"Stage 2 complete: {len(docs)} structured documents saved to {out_dir}")


if __name__ == "__main__":
    main()
