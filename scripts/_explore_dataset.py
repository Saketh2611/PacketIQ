#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from datasets import load_dataset

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings

settings = get_settings()
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=settings.dataset_name)
parser.add_argument("--config", default=settings.dataset_config)
parser.add_argument("--split", default="train")
args = parser.parse_args()

if args.config:
    ds = load_dataset(args.dataset, args.config, split=args.split)
else:
    ds = load_dataset(args.dataset, split=args.split)
print("Total rows:", len(ds))

# Group by stream_id
streams = defaultdict(list)
for row in ds:
    streams[row["stream_id"]].append(row)

print("Unique streams:", len(streams))
lengths = [len(v) for v in streams.values()]
print("Pages per stream: min", min(lengths), "max", max(lengths), "avg", sum(lengths)/len(lengths))

# Check boundary values
boundary_field = "boundary" if "boundary" in ds.column_names else "label"
text_field = "page_text" if "page_text" in ds.column_names else "text"
boundaries = Counter(r[boundary_field] for r in ds)
print("Boundary values:", boundaries)

# Sample one stream
sid = list(streams.keys())[0]
rows = sorted(streams[sid], key=lambda r: r["position"])
print(f"\nSample stream {sid} ({len(rows)} pages):")
for r in rows[:8]:
    source = r.get("source", args.dataset)
    print(f"  pos={r['position']} boundary={r[boundary_field]} source={source} text={r[text_field][:60]!r}...")

# How many boundaries per stream
bounds_per_stream = []
for sid, rows in streams.items():
    rows = sorted(rows, key=lambda r: r["position"])
    bcount = sum(1 for r in rows if r[boundary_field] == 1)
    bounds_per_stream.append(bcount)
print("\nBoundaries (boundary=1) per stream: min", min(bounds_per_stream), "max", max(bounds_per_stream))

# Check if boundary=1 always at position 0
first_page_boundary = sum(1 for sid, rows in streams.items() if sorted(rows, key=lambda r: r["position"])[0][boundary_field] == 1)
print("First page has boundary=1:", first_page_boundary, "/", len(streams))
