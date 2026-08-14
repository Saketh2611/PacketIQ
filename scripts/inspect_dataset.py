#!/usr/bin/env python3
"""Inspect the configured page-stream dataset schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings
from document_intelligence.dataset.adapter import DocSplitAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configured page-stream dataset")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset name")
    parser.add_argument("--config", default=None, help="Optional Hugging Face dataset config")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    settings = get_settings()
    adapter = DocSplitAdapter(dataset_name=args.dataset, dataset_config=args.config)
    try:
        ds = adapter.load_dataset(args.split)
        schema = adapter.inspect_schema()
        print("=== Page-Stream Dataset Inspection ===")
        print(f"Dataset: {adapter.dataset_name}")
        print(f"Config: {adapter.dataset_config or '(default)'}")
        print(f"Split: {args.split}")
        print(f"Samples: {schema['num_samples']}")
        print(f"Fields: {schema['fields']}")
        print(f"Field types: {schema['example_keys']}")
        sample = ds[0]
        print("\nExample sample (truncated):")
        truncated = {k: (str(v)[:200] + "..." if len(str(v)) > 200 else v) for k, v in sample.items()}
        print(json.dumps(truncated, indent=2, default=str))

        if {"stream_id", "position"}.issubset(sample) and ({"boundary"} <= set(sample) or {"label"} <= set(sample)):
            stream_id = sample["stream_id"]
            rows = []
            for row in ds:
                if row["stream_id"] == stream_id:
                    rows.append(row)
                elif rows:
                    break
            normalized = adapter.normalize_stream_rows(rows)
            ann = adapter.stream_rows_to_annotation(rows)
            print(f"\nParsed first stream: {stream_id}")
            print("Normalized fields: stream_id, page_number, original_position, text, starts_new_document, source")
            print(
                "First normalized page: "
                f"page_number={normalized[0].page_number}, "
                f"original_position={normalized[0].original_position}, "
                f"starts_new_document={normalized[0].starts_new_document}, "
                f"text_len={len(normalized[0].text)}, "
                f"source={normalized[0].source or '(none)'}"
            )
        else:
            ann = adapter.sample_to_annotation(sample, 0)
        print(f"\nParsed groups: {ann.groups}")
        print(f"Document types: {ann.document_types}")
    except Exception as exc:
        print(f"Could not load dataset: {exc}")
        manifest = settings.processed_dir / "dataset_manifest.json"
        if manifest.exists():
            print(f"\nExisting manifest:\n{manifest.read_text()}")


if __name__ == "__main__":
    main()
