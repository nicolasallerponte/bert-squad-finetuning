#!/usr/bin/env python3
"""Print a Markdown table with every result in results/*.json, ready to paste into the report.

    python summarize.py results
"""
import glob
import json
import os
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for path in glob.glob(os.path.join(folder, "*.json")):
    with open(path) as f:
        rows.append(json.load(f))
rows.sort(key=lambda r: r.get("date", ""))
latest = {}                      # keep only the most recent result of each run name
for r in rows:
    latest[r["run"]] = r
rows = list(latest.values())
if not rows:
    sys.exit(f"No results in {folder}/")

base = next((r for r in rows if r.get("samples_per_s")), None)
cols = ["run", "precision", "batch_size", "workers", "pin_memory", "compile",
        "samples_per_s", "speedup", "step_time_ms", "mfu_pct", "peak_mem_gb", "wall_time_s", "val_loss"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    if base and r.get("samples_per_s"):
        r["speedup"] = f"{r['samples_per_s'] / base['samples_per_s']:.2f}x"
    print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
