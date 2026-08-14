#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset
from document_intelligence.config.settings import get_settings

settings = get_settings()
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=settings.dataset_name)
parser.add_argument("--config", default=settings.dataset_config)
args = parser.parse_args()

if args.config:
    full = load_dataset(args.dataset, args.config)
else:
    full = load_dataset(args.dataset)
print("Dataset:", full)
for split in full:
    ds = full[split]
    print(f"\n=== Split: {split} ({len(ds)} samples) ===")
    print("Fields:", ds.column_names)
    sample = ds[0]
    info = {}
    for k, v in sample.items():
        if isinstance(v, (bytes, bytearray)):
            info[k] = {"type": "bytes", "len": len(v)}
        elif isinstance(v, list):
            info[k] = {"type": "list", "len": len(v), "preview": v[:5]}
        elif isinstance(v, dict):
            info[k] = {"type": "dict", "keys": list(v.keys())}
        else:
            info[k] = {"type": type(v).__name__, "value": str(v)[:200]}
    print(json.dumps(info, indent=2))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from document_intelligence.dataset.adapter import DocSplitAdapter

    adapter = DocSplitAdapter(dataset_name=args.dataset, dataset_config=args.config)
    adapter._dataset = ds
    if {"stream_id", "position"}.issubset(sample) and ({"boundary"} <= set(sample) or {"label"} <= set(sample)):
        rows = []
        for row in ds:
            if row["stream_id"] == sample["stream_id"]:
                rows.append(row)
            elif rows:
                break
        ann = adapter.stream_rows_to_annotation(rows)
    else:
        ann = adapter.sample_to_annotation(sample, 0)
    print("Parsed annotation:", ann)
