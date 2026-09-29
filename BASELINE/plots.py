#!/usr/bin/env python3
"""Figures of the report, from results/*.json:

    figures/throughput.png   samples/s of each optimization step (median and range of the repetitions)
    figures/profile.png      share of the GPU time of each kernel group, per step (torch.profiler)
    figures/loss.png         training loss against time, baseline and optimized full runs

    python plots.py results
"""
import glob
import gzip
import json
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
out_dir = "figures"

LABELS = {
    "s0_baseline": "Baseline: FP32, batch 16",
    "s1_pipeline": "+ 8 workers, pinned memory",
    "s2_tf32": "+ TF32",
    "s3_bf16": "+ BF16 mixed precision",
    "s4_batch32": "+ batch 32",
    "s5_compile": "+ torch.compile",
    "s6_fused_adam": "+ fused AdamW",
    "s7_batch64": "+ batch 64",
    "s8_batch128": "+ batch 128",
}
GROUPS = ["matmul", "attention", "element-wise and other", "optimizer", "memory copies"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]    # fixed order, one per group
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
    "ytick.color": TEXT, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

rows = []
for path in glob.glob(os.path.join(folder, "*.json")):
    with open(path) as f:
        rows.append(json.load(f))
rows.sort(key=lambda r: r.get("date", ""))
latest = {r["run"]: r for r in rows}                  # most recent result of each run name
steps = {}                                            # step name -> its repetitions
for r in latest.values():
    name = re.sub(r"_r\d+$", "", r["run"])
    if re.match(r"s\d+_", name):
        steps.setdefault(name, []).append(r)
order = sorted(steps, key=lambda n: int(re.match(r"s(\d+)_", n).group(1)))
os.makedirs(out_dir, exist_ok=True)


def save(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("written", path)


# ---------------------------------------------------------------- 1. throughput per step
if order:
    med = [statistics.median(r["samples_per_s"] for r in steps[n]) for n in order]
    lo = [min(r["samples_per_s"] for r in steps[n]) for n in order]
    hi = [max(r["samples_per_s"] for r in steps[n]) for n in order]
    y = range(len(order))[::-1]                       # step 0 at the top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.2))
    ax.barh(y, med, height=0.6, color=COLORS[0])
    if any(h > l for l, h in zip(lo, hi)):
        ax.errorbar(med, y, xerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]],
                    fmt="none", ecolor=TEXT2, elinewidth=1, capsize=3)
    for yi, m, h in zip(y, med, hi):
        ax.text(h + max(med) * 0.01, yi, f"{m / med[0]:.2f}x", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(list(y), [LABELS.get(n, n) for n in order])
    ax.set_xlabel("training throughput (samples/s)")
    ax.set_xlim(0, max(hi) * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title("Throughput after each optimization, applied cumulatively", loc="left", fontsize=11)
    save(fig, "throughput.png")

# ---------------------------------------------------------------- 2. profiler breakdown per step
def trace_breakdown(run):
    """Share of the GPU time of each kernel group, from the torch.profiler trace of a run.

    Only GPU kernels and memory operations are counted (annotations such as ProfilerStep# span
    them). A kernel launched inside Optimizer.step counts as optimizer whatever its name: without
    --fused-adam, AdamW runs as generic element-wise kernels. Shares, not absolute times, because
    the profiled steps are the first ones of the run and are slower than the measured window."""
    from train_qa import kernel_group
    paths = glob.glob(os.path.join("profiles", run["run"], "*.pt.trace.json*"))
    if not paths:
        return None
    opener = gzip.open if paths[0].endswith(".gz") else open
    with opener(paths[0], "rt") as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    opt_spans = [(e["ts"], e["ts"] + e["dur"]) for e in events
                 if e.get("cat") == "user_annotation" and e["name"].startswith("Optimizer.step")]
    opt_launches = {e["args"].get("correlation") for e in events if e.get("cat") == "cuda_runtime"
                    and any(t0 <= e["ts"] <= t1 for t0, t1 in opt_spans)}
    groups = {}
    for e in events:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            g = "optimizer" if e["args"].get("correlation") in opt_launches else kernel_group(e["name"])
            groups[g] = groups.get(g, 0.0) + e["dur"]
    total = sum(groups.values())
    return {g: 100 * t / total for g, t in groups.items()}


prof = []
for n in order:
    first = next((r for r in steps[n] if r["run"].endswith("_r1")), steps[n][0])
    b = trace_breakdown(first)
    if b:
        prof.append((n, b))
if prof:
    print("GPU time share (%): step | " + " | ".join(GROUPS))
    for n, b in prof:
        print(f"  {n} | " + " | ".join(f"{b.get(g, 0.0):.1f}" for g in GROUPS))
    y = range(len(prof))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(prof) + 1.6))
    left = [0.0] * len(prof)
    for g, color in zip(GROUPS, COLORS):
        vals = [p.get(g, 0.0) for _, p in prof]
        if max(vals) < 0.5:                               # invisible at this scale (memory copies)
            continue
        ax.barh(y, vals, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2, label=g)
        if g in ("matmul", "attention"):                  # the two shares the discussion refers to
            for yi, l0, v in zip(y, left, vals):
                if v >= 9:
                    ax.text(l0 + v / 2, yi, f"{v:.0f} %", ha="center", va="center", color="white", fontsize=8.5)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(list(y), [LABELS.get(n, n) for n, _ in prof])
    ax.set_xlabel("share of the GPU time (%), torch.profiler")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=8.5)
    ax.set_title("Where the GPU time goes", loc="left", fontsize=11, pad=24)
    save(fig, "profile.png")

# ---------------------------------------------------------------- 3. loss against time
full = [(name, latest.get(name)) for name in ("baseline_full", "final_full")]
full = [(n, r) for n, r in full if r and r.get("loss_curve")]
if full:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = {"baseline_full": "Baseline (FP32, batch 16)", "final_full": "Optimized (BF16, batch 64, compiled)"}
    for (n, r), color in zip(full, COLORS):
        _, t, loss = zip(*r["loss_curve"])
        minutes = [s / 60 for s in t]
        ax.plot(minutes, loss, color=color, linewidth=2, label=names.get(n, n))
        ax.annotate(f"{r['wall_time_s'] / 60:.1f} min", (minutes[-1], loss[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("training time (minutes)")
    ax.set_ylabel("training loss (mean of 100 steps)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Same training, 2 epochs: loss against wall-clock time", loc="left", fontsize=11)
    save(fig, "loss.png")
