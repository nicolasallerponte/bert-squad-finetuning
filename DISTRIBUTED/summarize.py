#!/usr/bin/env python3
"""Print a Markdown table with every result in results/*.json, ready to paste into the report.

Repetitions of the same configuration (run names ending in _r1, _r2...) are grouped: the table
gives the median throughput, its range (min-max) and the number of repetitions. Speedup and
parallel efficiency are relative to weak_g1, the same script on one GPU:

    speedup = samples/s / samples/s on 1 GPU        efficiency = speedup / number of GPUs

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
latest = {r["run"]: r for r in rows}           # keep only the most recent result of each run name
latest.pop("nccl_check", None)                 # the NCCL log run is too short to measure anything
if not latest:
    sys.exit(f"No results in {folder}/")

groups = {}                                    # configuration name -> its repetitions
for r in latest.values():
    groups.setdefault(re.sub(r"_r\d+$", "", r["run"]), []).append(r)


def median(reps, key):
    values = [r[key] for r in reps if r.get(key) is not None]
    return round(statistics.median(values), 4) if values else ""


def comm_share(reps):
    """Share of the GPU busy time spent in NCCL kernels, from the profiled repetition."""
    for r in reps:
        p = r.get("profile_ms_per_sample")
        if p and sum(p.values()):
            return f"{100 * p.get('communication', 0) / sum(p.values()):.0f} %"
    return ""


table = []
for name, reps in groups.items():
    sps = [r["samples_per_s"] for r in reps if r.get("samples_per_s")]
    first = reps[0]
    table.append(dict(
        run=name, gpus=first["world_size"], nodes=first["nodes"], strategy=first["strategy"],
        batch_per_gpu=first["batch_size"], global_batch=first["global_batch_size"], reps=len(reps),
        samples_per_s=median(reps, "samples_per_s"), range=f"{min(sps)}-{max(sps)}" if len(sps) > 1 else "",
        per_gpu=median(reps, "samples_per_s_per_gpu"), step_time_ms=median(reps, "step_time_ms"),
        mfu_pct=median(reps, "mfu_pct"), comm_share=comm_share(reps), peak_mem_gb=median(reps, "peak_mem_gb"),
        wall_time_s=median(reps, "wall_time_s"), val_loss=median(reps, "val_loss"),
        exact_match=median(reps, "exact_match"), f1=median(reps, "f1")))

base = next((t for t in table if t["run"] == "weak_g1" and t["samples_per_s"]), None)
for t in table:
    if base and t["samples_per_s"]:
        s = t["samples_per_s"] / base["samples_per_s"]
        t["speedup"] = f"{s:.2f}x"
        t["efficiency"] = f"{100 * s / t['gpus']:.0f} %"
table.sort(key=lambda t: (not t["run"].startswith("weak"), not t["run"].startswith("strong"),
                          t["run"].startswith("full"), t["gpus"], t["nodes"], t["run"]))

cols = ["run", "gpus", "nodes", "strategy", "batch_per_gpu", "global_batch", "reps", "samples_per_s", "range",
        "speedup", "efficiency", "per_gpu", "step_time_ms", "mfu_pct", "comm_share", "peak_mem_gb",
        "wall_time_s", "val_loss", "exact_match", "f1"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for t in table:
    print("| " + " | ".join(str(t.get(c, "")) for c in cols) + " |")
