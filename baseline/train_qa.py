#!/usr/bin/env python3
"""
Fine-tuning BERT-base on SQuAD v1.1 on a single GPU, with explicit PyTorch training loop.

Every optimization from the HPC-for-AI lab is exposed as a flag, so the *same* script
produces the baseline and each improvement, and they can be compared one by one:

    --workers N --pin-memory     input pipeline
    --precision {fp32,tf32,bf16,fp16}
    --batch-size N
    --compile                    torch.compile

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
import socket
import time
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

    # measurement
    p.add_argument("--warmup-steps", type=int, default=20, help="steps excluded from throughput")
    p.add_argument("--measure-steps", type=int, default=100, help="steps used to measure throughput")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--profile", action="store_true", help="run torch.profiler for a few steps")
    p.add_argument("--eval-samples", type=int, default=1600,
                   help="validation samples evaluated at the end, always the same ones (0 = skip)")

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


def run_profiler(model, loader, optimizer, scaler, args, device):
    """A few steps under torch.profiler; prints the top CUDA kernels. Output goes to the log."""
    from torch.profiler import ProfilerActivity, profile, schedule

    model.train()
    it = iter(loader)
    sched = schedule(wait=1, warmup=3, active=5, repeat=1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched, record_shapes=False) as prof:
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
    print("\n===== torch.profiler: top 15 by CUDA time =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("===== end profiler =====\n", flush=True)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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

    if args.profile:
        run_profiler(model, train_loader, optimizer, scaler, args, device)

    # ---------------- training loop
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step, measure_t0, measured_steps = 0, None, 0
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
                print(f"  step {step:>6}/{total_steps}  epoch {epoch}  loss {(running / args.log_every).item():.4f}",
                      flush=True)
                running.zero_()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    # ---------------- results
    res = dict(run=run_name, date=datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__, **{k: v for k, v in vars(args).items()},
               total_steps=step, wall_time_s=round(wall, 2), params=n_params,
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
    if args.eval_samples:
        res["val_loss"] = round(evaluate(model, val_loader, args, device), 4)

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"{run_name}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n===== RESULT =====")
    for k in ("run", "samples_per_s", "step_time_ms", "achieved_tflops", "mfu_pct", "peak_mem_gb",
              "wall_time_s", "val_loss"):
        if k in res:
            print(f"  {k:<16} {res[k]}")
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()
