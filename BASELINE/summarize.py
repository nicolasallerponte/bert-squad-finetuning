#!/usr/bin/env python3
"""Print a Markdown table with every result in results/*.json, ready to paste into the report.

Repetitions of the same configuration (run names ending in _r1, _r2...) are grouped: the table
gives the median throughput, its range (min-max) and the number of repetitions.

    python summarize.py results
"""
import glob
import json
import os
import re
import statistics
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
if not latest:
    sys.exit(f"No results in {folder}/")

groups = {}                      # configuration name -> its repetitions, in order of appearance
for r in latest.values():
    groups.setdefault(re.sub(r"_r\d+$", "", r["run"]), []).append(r)


def median(reps, key):
    values = [r[key] for r in reps if r.get(key) is not None]
    return round(statistics.median(values), 4) if values else ""


table = []
for name, reps in groups.items():
    sps = [r["samples_per_s"] for r in reps if r.get("samples_per_s")]
    first = reps[0]
    table.append(dict(
        run=name, precision=first.get("precision"), batch_size=first.get("batch_size"),
        workers=first.get("workers"), compile=first.get("compile"), fused_adam=first.get("fused_adam", False),
        reps=len(reps), samples_per_s=median(reps, "samples_per_s"),
        range=f"{min(sps)}-{max(sps)}" if len(sps) > 1 else "",
        step_time_ms=median(reps, "step_time_ms"), mfu_pct=median(reps, "mfu_pct"),
        peak_mem_gb=median(reps, "peak_mem_gb"), wall_time_s=median(reps, "wall_time_s"),
        avg_samples_per_s=median(reps, "avg_samples_per_s"), val_loss=median(reps, "val_loss"),
        exact_match=median(reps, "exact_match"), f1=median(reps, "f1")))

base = next((t for t in table if t["samples_per_s"]), None)
for t in table:
    if base and t["samples_per_s"]:
        t["speedup"] = f"{t['samples_per_s'] / base['samples_per_s']:.2f}x"

cols = ["run", "precision", "batch_size", "workers", "compile", "fused_adam", "reps", "samples_per_s", "range",
        "speedup", "step_time_ms", "mfu_pct", "peak_mem_gb", "wall_time_s", "avg_samples_per_s", "val_loss",
        "exact_match", "f1"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for t in table:
    print("| " + " | ".join(str(t.get(c, "")) for c in cols) + " |")
