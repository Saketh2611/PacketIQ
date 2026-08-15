#!/usr/bin/env python3
"""Inspect train and test page-stream datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings
from document_intelligence.dataset.adapter import DocSplitAdapter


def inspect_one(label: str, adapter: DocSplitAdapter, split: str) -> None:
    ds = adapter.load_dataset(split)
    schema = adapter.inspect_schema()
    print(f"\n=== {label} ===")
    print(f"Dataset: {adapter.dataset_name}")
    print(f"Config: {adapter.dataset_config or '(default)'}")
    print(f"Split: {split}")
    print(f"Rows: {schema['num_samples']}")
    print(f"Fields: {schema['fields']}")
    print(f"Field types: {schema['example_keys']}")

    sample = ds[0]
    preview = {}
    for k, v in sample.items():
        if k == "image":
            preview[k] = str(type(v).__name__)
        else:
            preview[k] = str(v)[:160]
    print("\nExample row:")
    print(json.dumps(preview, indent=2))

    grouped = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        grouped[row["stream_id"]].append(row)
    first_stream = next(iter(grouped.values()))
    ann = adapter.stream_rows_to_annotation(first_stream)
    print(f"\nFirst stream pages: {len(first_stream)}")
    print(f"Parsed groups: {ann.groups}")
    print(f"Document types: {ann.document_types}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configured page-stream datasets")
    parser.add_argument("--split", default=None, help="Split override for single-dataset mode")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = get_settings()

    if args.dataset:
        adapter = DocSplitAdapter(dataset_name=args.dataset, dataset_config=args.config)
        inspect_one("Custom dataset", adapter, args.split or "train")
        return

    train_adapter = DocSplitAdapter(
        dataset_name=settings.dataset_name,
        dataset_config=settings.dataset_config,
    )
    test_adapter = DocSplitAdapter(
        dataset_name=settings.test_dataset_name,
        dataset_config=settings.test_dataset_config,
    )

    print("Dataset roles:")
    print("- openpss-mirror: OpenPSS community mirror used for train/dev experimentation")
    print("- doc-split-benchmark: official evaluation slice (our200 config)")
    print("- doc-split-v2: assignment reference name; verify separately if accessible")

    inspect_one("Train/Dev dataset", train_adapter, "train")
    inspect_one("Test/Eval dataset", test_adapter, "test")


if __name__ == "__main__":
    main()
