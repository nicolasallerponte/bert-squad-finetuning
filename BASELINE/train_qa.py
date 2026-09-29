#!/usr/bin/env python3
"""
Fine-tuning BERT-base on SQuAD v1.1 on a single GPU, with explicit PyTorch training loop.

Every optimization from the HPC-for-AI lab is exposed as a flag, so the *same* script
produces the baseline and each improvement, and they can be compared one by one:

    --workers N --pin-memory     input pipeline
    --precision {fp32,tf32,bf16,fp16}
    --batch-size N
    --compile                    torch.compile
    --fused-adam                 AdamW as a single fused CUDA kernel

Throughput is measured over --measure-steps steps AFTER --warmup-steps steps, because the
first iterations pay for CUDA context creation, memory-pool growth and compilation.

Usage:
    python train_qa.py --prepare-only                 # download + tokenize once (login node)
    python train_qa.py --max-steps 300                # quick throughput measurement
    python train_qa.py --epochs 1                     # full training run
"""

import argparse
import json
import math
import os
import random
import re
import socket
import string
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("STORE", "."), "lab-ai-data"),
                   help="where the tokenized dataset is cached")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full epochs)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)

    # the optimizations
    p.add_argument("--workers", type=int, default=0, help="DataLoader num_workers")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--precision", choices=["fp32", "tf32", "bf16", "fp16"], default="fp32")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--fused-adam", action="store_true", help="fused CUDA implementation of AdamW")

    # measurement
    p.add_argument("--warmup-steps", type=int, default=20, help="steps excluded from throughput")
    p.add_argument("--measure-steps", type=int, default=100, help="steps used to measure throughput")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--profile", action="store_true", help="run torch.profiler for a few steps")
    p.add_argument("--eval-samples", type=int, default=1600,
                   help="validation samples evaluated at the end, always the same ones (0 = skip)")
    p.add_argument("--squad-eval", action="store_true",
                   help="after training, exact match and F1 on the whole SQuAD validation split")

    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--prepare-only", action="store_true", help="download + tokenize, then exit")
    return p.parse_args()


# --------------------------------------------------------------------------- data
def prepare_features(examples, tokenizer, max_len, stride):
    """Standard SQuAD preprocessing: long contexts are split into overlapping windows, and the
    answer span is mapped from characters to token positions (CLS if the answer is not inside)."""
    questions = [q.lstrip() for q in examples["question"]]
    tok = tokenizer(
        questions, examples["context"],
        truncation="only_second", max_length=max_len, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",          # static shapes: required for torch.compile without recompiles
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets_all = tok.pop("offset_mapping")
    starts, ends = [], []
    for i, offsets in enumerate(offsets_all):
        input_ids = tok["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        seq_ids = tok.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        idx = ctx_start
        while idx <= ctx_end and offsets[idx][0] <= start_char:
            idx += 1
        starts.append(idx - 1)
        idx = ctx_end
        while idx >= ctx_start and offsets[idx][1] >= end_char:
            idx -= 1
        ends.append(idx + 1)
    tok["start_positions"] = starts
    tok["end_positions"] = ends
    return tok


def load_tokenized(args, tokenizer):
    from datasets import load_dataset, load_from_disk

    path = os.path.join(args.data_dir, f"squad_{args.model.replace('/', '_')}_{args.max_len}_{args.stride}")
    if os.path.isdir(path):
        return load_from_disk(path)

    raw = load_dataset("rajpurkar/squad")
    cols = raw["train"].column_names
    tokenized = raw.map(
        lambda ex: prepare_features(ex, tokenizer, args.max_len, args.stride),
        batched=True, remove_columns=cols, num_proc=8, desc="tokenizing",
    )
    os.makedirs(args.data_dir, exist_ok=True)
    tokenized.save_to_disk(path)
    return tokenized


# --------------------------------------------------------------------------- helpers
PEAK_TFLOPS = {"fp32": 19.5, "tf32": 156.0, "bf16": 312.0, "fp16": 312.0}   # A100 dense, no sparsity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_ctx(precision):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


KERNEL_GROUPS = (                                  # first match wins
    ("optimizer", ("adam", "multi_tensor_apply", "foreach")),
    ("attention", ("fmha", "flash", "attention", "attn")),
    ("matmul", ("gemm", "cutlass", "cublas", "aten::mm", "aten::addmm", "aten::bmm")),
    ("memory copies", ("memcpy", "memset")),
)


def kernel_group(name):
    name = name.lower()
    for group, keys in KERNEL_GROUPS:
        if any(k in name for k in keys):
            return group
    return "element-wise and other"


def run_profiler(model, loader, optimizer, scaler, args, device, run_name):
    """5 steps under torch.profiler, after 1 skipped and 3 warmup steps. Prints the top operations,
    saves a trace for TensorBoard and returns the GPU time of each kernel group."""
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    model.train()
    it = iter(loader)
    active = 5
    sched = schedule(wait=1, warmup=3, active=active, repeat=1)
    trace_dir = os.path.join("profiles", run_name)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=sched,
                 on_trace_ready=tensorboard_trace_handler(trace_dir, use_gzip=True),
                 record_shapes=False) as prof:
        for _ in range(9):
            batch = next(it)
            batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
            with autocast_ctx(args.precision):
                loss = model(**batch).loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            prof.step()
    averages = prof.key_averages()
    print("\n===== torch.profiler: top 15 by CUDA time =====")
    print(averages.table(sort_by="cuda_time_total", row_limit=15))

    # GPU time grouped by what it computes: from the GPU kernel rows when the profiler records
    # them (CUDA), otherwise from the self GPU time of each operation
    def device_time(evt):
        t = getattr(evt, "self_device_time_total", None)
        return (evt.self_cuda_time_total if t is None else t) / 1000     # us -> ms

    groups = {}
    try:
        # device rows are kernels and copies, plus annotations that span them (ProfilerStep#,
        # Optimizer.step#...): those would count the same time twice
        kernels = [e for e in averages if str(e.device_type).endswith("CUDA")
                   and not getattr(e, "is_user_annotation", False)
                   and not e.key.startswith(("ProfilerStep", "Optimizer."))]
        rows = kernels or [e for e in averages if device_time(e) > 0]
        for evt in rows:
            g = kernel_group(evt.key)
            groups[g] = groups.get(g, 0.0) + device_time(evt)
    except Exception as e:                            # never lose a run because of the summary
        print(f"WARNING: could not group the GPU time: {e}")
    samples = active * args.batch_size
    summary = {g: round(ms / samples, 3) for g, ms in sorted(groups.items())}   # ms per sample
    print(f"GPU time per sample by kernel group (ms): {summary} | trace in {trace_dir}")
    print("===== end profiler =====\n", flush=True)
    return summary


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()                         # dropout off; forgetting this silently changes the model
    # A fixed number of SAMPLES, not of batches: with shuffle=False the first N validation
    # samples are the same whatever the batch size, so runs with different batches are comparable.
    n_batches = args.eval_samples // args.batch_size
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
        with autocast_ctx(args.precision):
            total += model(**batch).loss.float().item()
        n += 1
    model.train()
    return total / max(n, 1)


# --------------------------------------------------------------------------- SQuAD metrics
def normalize_answer(s):
    """Normalization of the official SQuAD v1.1 script: lower case, no punctuation, no articles."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, truth):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(truth).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def squad_metrics(predictions, answers):
    """Exact match and F1 (in %), each prediction scored against its best reference answer."""
    em = f1 = 0.0
    for pred, refs in zip(predictions, answers):
        em += max(float(normalize_answer(pred) == normalize_answer(r)) for r in refs)
        f1 += max(f1_score(pred, r) for r in refs)
    return 100 * em / len(answers), 100 * f1 / len(answers)


def extract_answers(start_logits, end_logits, feats, contexts, n_best=20, max_answer_len=30):
    """For every example, the highest scoring valid span over all its windows (standard HF
    post-processing): start and end inside the context, end >= start, at most max_answer_len tokens."""
    best = [(-float("inf"), "") for _ in contexts]
    starts = torch.topk(start_logits, n_best, dim=1)
    ends = torch.topk(end_logits, n_best, dim=1)
    for i, ex in enumerate(feats["overflow_to_sample_mapping"]):
        offsets, seq_ids = feats["offset_mapping"][i], feats.sequence_ids(i)
        for s_score, s in zip(starts.values[i].tolist(), starts.indices[i].tolist()):
            if seq_ids[s] != 1:
                continue
            for e_score, e in zip(ends.values[i].tolist(), ends.indices[i].tolist()):
                if seq_ids[e] != 1 or e < s or e - s + 1 > max_answer_len:
                    continue
                if s_score + e_score > best[ex][0]:
                    best[ex] = (s_score + e_score, contexts[ex][offsets[s][0]:offsets[e][1]])
    return [text for _, text in best]


@torch.no_grad()
def squad_evaluate(model, tokenizer, args, device):
    """Exact match and F1 on the whole validation split, as reported for SQuAD v1.1."""
    from datasets import load_dataset

    raw = load_dataset("rajpurkar/squad", split="validation")
    contexts = list(raw["context"])                 # recent datasets return a Column, not a list
    feats = tokenizer([q.lstrip() for q in raw["question"]], contexts,
                      truncation="only_second", max_length=args.max_len, stride=args.stride,
                      return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    model = getattr(model, "_orig_mod", model)      # the eager module: the last batch is smaller
    model.eval()
    keys = ("input_ids", "token_type_ids", "attention_mask")
    starts, ends = [], []
    for b in range(0, len(feats["input_ids"]), args.batch_size):
        batch = {k: torch.tensor(feats[k][b:b + args.batch_size], device=device) for k in keys}
        with autocast_ctx(args.precision):
            out = model(**batch)
        starts.append(out.start_logits.float().cpu())
        ends.append(out.end_logits.float().cpu())
    model.train()
    preds = extract_answers(torch.cat(starts), torch.cat(ends), feats, contexts)
    em, f1 = squad_metrics(preds, [a["text"] for a in raw["answers"]])
    return dict(exact_match=round(em, 2), f1=round(f1, 2), squad_examples=len(raw))


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = load_tokenized(args, tokenizer)
    if args.prepare_only:
        AutoModelForQuestionAnswering.from_pretrained(args.model)     # warm the model cache too
        print(f"Prepared: train={len(data['train'])} features, validation={len(data['validation'])} features")
        return

    assert torch.cuda.is_available(), "No GPU visible. Run this on a GPU node, not the login node."
    device = torch.device("cuda")
    set_seed(args.seed)

    # TF32 only when explicitly requested, so that 'fp32' is a true FP32 baseline
    torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = args.precision == "tf32"

    data.set_format("torch")
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_memory,
                     drop_last=True)            # equal-size batches: no recompilation with --compile
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(data["train"], shuffle=True, **loader_kw)
    val_loader = DataLoader(data["validation"], shuffle=False, **loader_kw)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # FLOPs per token (Kaplan et al., 2020): 6 x non-embedding params + attention term.
    # Embedding lookups do no matrix multiplication, so they are excluded.
    n_emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
    cfg = model.config
    flops_per_token = 6 * (n_params - n_emb) + 6 * cfg.num_hidden_layers * args.max_len * cfg.hidden_size
    flops_per_sample = flops_per_token * args.max_len
    if args.compile:
        model = torch.compile(model)

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=args.fused_adam)
    warm = max(1, int(0.1 * total_steps))     # linear warmup (10 %) then linear decay
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    run_name = args.run_name or (f"bs{args.batch_size}_{args.precision}_w{args.workers}"
                                 f"{'_pin' if args.pin_memory else ''}{'_compile' if args.compile else ''}")
    print(f"Run: {run_name} | host {socket.gethostname()} | {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | params {n_params/1e6:.1f} M "
          f"({(n_params - n_emb)/1e6:.1f} M non-embedding) | {flops_per_sample/1e9:.0f} GFLOP/sample")
    print(f"train features {len(data['train'])} | steps/epoch {steps_per_epoch} | total steps {total_steps}",
          flush=True)

    profile_ms = None
    if args.profile:
        profile_ms = run_profiler(model, train_loader, optimizer, scaler, args, device, run_name)

    # ---------------- training loop
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step, measure_t0, measured_steps = 0, None, 0
    loss_curve = []                             # (step, seconds, mean loss) every --log-every steps
    running = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    done = False
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
        for batch in train_loader:
            if step == args.warmup_steps:
                torch.cuda.synchronize()
                measure_t0 = time.perf_counter()

            batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
            with autocast_ctx(args.precision):
                loss = model(**batch).loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_sched.step()
            running += loss.detach().float()      # no .item() here: it would force a sync every step
            step += 1

            if measure_t0 is not None and measured_steps == 0 and step == args.warmup_steps + args.measure_steps:
                torch.cuda.synchronize()
                measured_time = time.perf_counter() - measure_t0
                measured_steps = args.measure_steps
            if step % args.log_every == 0:
                mean_loss = (running / args.log_every).item()
                loss_curve.append((step, round(time.perf_counter() - t_start, 2), round(mean_loss, 4)))
                print(f"  step {step:>6}/{total_steps}  epoch {epoch}  loss {mean_loss:.4f}", flush=True)
                running.zero_()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    # ---------------- results
    config = {k: v for k, v in vars(args).items() if k not in ("data_dir", "results_dir")}   # no local paths
    res = dict(run=run_name, date=datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__, **config,
               total_steps=step, wall_time_s=round(wall, 2),
               avg_samples_per_s=round(step * args.batch_size / wall, 1),   # whole run, not a window
               params=n_params,
               non_embedding_params=n_params - n_emb, gflops_per_sample=round(flops_per_sample / 1e9, 1),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    if measured_steps:
        sps = measured_steps * args.batch_size / measured_time
        achieved_tflops = sps * flops_per_sample / 1e12
        res.update(samples_per_s=round(sps, 1), step_time_ms=round(1000 * measured_time / measured_steps, 1),
                   achieved_tflops=round(achieved_tflops, 1),
                   mfu_pct=round(100 * achieved_tflops / PEAK_TFLOPS[args.precision], 1),
                   peak_tflops_used=PEAK_TFLOPS[args.precision])
    else:
        print("WARNING: not enough steps to measure throughput (increase --max-steps).")
    res["loss_curve"] = loss_curve
    if profile_ms:
        res["profile_ms_per_sample"] = profile_ms
    if args.eval_samples:
        res["val_loss"] = round(evaluate(model, val_loader, args, device), 4)
    if args.squad_eval:
        res.update(squad_evaluate(model, tokenizer, args, device))

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"{run_name}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n===== RESULT =====")
    for k in ("run", "samples_per_s", "avg_samples_per_s", "step_time_ms", "achieved_tflops", "mfu_pct",
              "peak_mem_gb", "wall_time_s", "val_loss", "exact_match", "f1"):
        if k in res:
            print(f"  {k:<16} {res[k]}")
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()
