#!/usr/bin/env python3
"""Compare dataset cards, splits, and schemas."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import dataset_info

from document_intelligence.config.settings import get_settings

settings = get_settings()

DATASETS = [
    ("train/dev", settings.dataset_name, settings.dataset_config),
    ("test/eval", settings.test_dataset_name, settings.test_dataset_config),
    ("doc-split-v2 (reference)", "nutrientdocs/doc-split-v2", None),
]


def summarize_rows(ds, n: int = 3) -> dict:
    sample = ds[0]
    fields = {k: type(v).__name__ for k, v in sample.items()}
    streams = defaultdict(int)
    for row in ds:
        streams[row.get("stream_id", "unknown")] += 1
    lengths = list(streams.values())
    return {
        "rows": len(ds),
        "unique_streams": len(streams),
        "avg_pages_per_stream": sum(lengths) / len(lengths) if lengths else 0,
        "fields": fields,
        "sample_keys": list(sample.keys()),
    }


def main() -> None:
    report = {"datasets": {}}
    for role, name, config in DATASETS:
        entry = {"role": role, "name": name, "config": config}
        try:
            info = dataset_info(name)
            entry["description"] = (info.description or "")[:500]
            entry["configs"] = info.config_names or []
            entry["splits_in_card"] = list((info.splits or {}).keys()) if info.splits else []
        except Exception as exc:
            entry["card_error"] = str(exc)

        try:
            configs = get_dataset_config_names(name)
            entry["available_configs"] = configs
        except Exception as exc:
            entry["config_list_error"] = str(exc)

        try:
            if config:
                full = load_dataset(name, config)
            else:
                full = load_dataset(name)
            entry["loaded_splits"] = {k: summarize_rows(v) for k, v in full.items()}
        except Exception as exc:
            entry["load_error"] = str(exc)

        report["datasets"][name] = entry
        print(f"\n=== {role}: {name} (config={config}) ===")
        print(json.dumps(entry, indent=2, default=str))

    out = settings.processed_dir / "dataset_relationship_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved relationship report to {out}")


if __name__ == "__main__":
    main()
