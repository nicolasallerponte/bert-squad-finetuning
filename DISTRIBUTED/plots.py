#!/usr/bin/env python3
"""Figures of the report, from results/*.json:

    scaling.png   samples/s against the number of GPUs, weak and strong scaling, with the ideal line
    loss.png      training loss against time of the two full runs, and of BASELINE's optimized run

    python plots.py results
"""
import glob
import json
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
SURFACE, TEXT, TEXT2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3dd"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def load(folder):
    rows = []
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: r.get("date", ""))
    return {r["run"]: r for r in rows}


latest = load(folder)
groups = {}
for r in latest.values():
    groups.setdefault(re.sub(r"_r\d+$", "", r["run"]), []).append(r)


def med(name):
    v = [r["samples_per_s"] for r in groups.get(name, []) if r.get("samples_per_s")]
    return statistics.median(v) if v else None


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)


os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- scaling
one = med("weak_g1")
if one:
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax)
    gpus = [1, 2, 4]
    ax.plot(gpus, [one * g for g in gpus], "--", color=TEXT2, label="ideal (linear)")
    series = [("weak: 64 per GPU", BLUE, ["weak_g1", "weak_g2_1node", "weak_g4"]),
              ("strong: global batch 64", ORANGE, ["weak_g1", "strong_g2", "strong_g4"])]
    for label, color, names in series:
        pts = [(g, med(n)) for g, n in zip(gpus, names) if med(n)]
        if pts:
            ax.plot(*zip(*pts), "o-", color=color, label=label)
            for g, v in pts:
                ax.annotate(f"{v:.0f}", (g, v), textcoords="offset points", xytext=(6, -12), color=color)
    if med("weak_g2_2nodes"):
        ax.plot([2], [med("weak_g2_2nodes")], "s", color=GREEN, label="weak, 2 GPU on 2 nodes")
    ax.set_xticks(gpus, ["1 GPU", "2 GPU\n(1 node)", "4 GPU\n(2 nodes)"])
    ax.set_ylabel("samples/s (all GPUs)", color=TEXT)
    ax.set_title("Training throughput, BERT-base on SQuAD, A100 40 GB", color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "scaling.png"), dpi=200)
    print("figures/scaling.png")

# ---------------------------------------------------------------- loss
curves = [("full_g4_global64", "4 GPU, global batch 64", BLUE),
          ("full_g4_global256", "4 GPU, global batch 256", ORANGE)]
base = load(os.path.join("..", "BASELINE", folder)).get("final_full")
if any(n in latest for n, _, _ in curves):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax)
    if base:
        t, l = zip(*[(s / 60, v) for _, s, v in base["loss_curve"]])
        ax.plot(t, l, color=TEXT2, label="BASELINE optimized, 1 GPU")
    for name, label, color in curves:
        if name in latest and latest[name]["loss_curve"]:
            t, l = zip(*[(s / 60, v) for _, s, v in latest[name]["loss_curve"]])
            ax.plot(t, l, color=color, label=label)
    ax.set_xlabel("minutes", color=TEXT)
    ax.set_ylabel("training loss", color=TEXT)
    ax.set_title("Training loss against time, 2 epochs", color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "loss.png"), dpi=200)
    print("figures/loss.png")
