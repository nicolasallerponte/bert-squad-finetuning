#!/usr/bin/env python3
"""
Fine-tuning BERT-base on SQuAD v1.1 on several GPUs and nodes, with PyTorch DistributedDataParallel.

It is the training loop of BASELINE/train_qa.py with the changes that distribution needs, and
nothing else: the data, the model, the optimizations and the way throughput is measured are the
same, so the numbers of both deliverables can be compared directly. The shared code (data
preparation, evaluation, SQuAD metrics) is imported from BASELINE/train_qa.py.

One process per GPU. The script reads its rank from the environment, so it runs under srun
(one SLURM task per GPU, no torchrun) or under torchrun:

    srun python train_ddp.py --batch-size 64 ...              # inside an sbatch job
    torchrun --nproc-per-node 2 train_ddp.py ...              # one node, without SLURM

--batch-size is the batch of EACH GPU: the global batch is --batch-size x number of GPUs.

Strategies, besides plain DDP:
    --zero                     ZeRO stage 1: the AdamW state is sharded among the GPUs
    --grad-compress bf16       gradients cast to BF16 for the all-reduce (half the traffic)
"""

import argparse
import contextlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BASELINE"))
from train_qa import (PEAK_TFLOPS, autocast_ctx, evaluate, kernel_group,  # noqa: E402
                      load_tokenized, set_seed, squad_evaluate)


# --------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("STORE", "."), "lab-ai-data"),
                   help="where the tokenized dataset is cached (the same as in BASELINE)")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full epochs)")
    p.add_argument("--batch-size", type=int, default=64, help="batch of each GPU")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)

    # single-GPU optimizations, as in BASELINE
    p.add_argument("--workers", type=int, default=0, help="DataLoader num_workers, per process")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--precision", choices=["fp32", "tf32", "bf16", "fp16"], default="fp32")
    p.add_argument("--compile", action="store_true", help="torch.compile the DDP model")
    p.add_argument("--fused-adam", action="store_true", help="fused CUDA implementation of AdamW")

    # distribution
    p.add_argument("--zero", action="store_true", help="ZeRO stage 1 (ZeroRedundancyOptimizer)")
    p.add_argument("--grad-compress", choices=["none", "bf16"], default="none",
                   help="DDP communication hook that compresses the gradients before the all-reduce")
    p.add_argument("--bucket-mb", type=int, default=25, help="size of the DDP gradient buckets")

    # measurement
    p.add_argument("--warmup-steps", type=int, default=20, help="steps excluded from throughput")
    p.add_argument("--measure-steps", type=int, default=100, help="steps used to measure throughput")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--profile", action="store_true", help="run torch.profiler for a few steps (rank 0)")
    p.add_argument("--eval-samples", type=int, default=1600,
                   help="validation samples evaluated at the end by rank 0, always the same ones (0 = skip)")
    p.add_argument("--squad-eval", action="store_true",
                   help="after training, exact match and F1 on the whole SQuAD validation split (rank 0)")

    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


# --------------------------------------------------------------------------- distributed setup
def first_host(nodelist):
    """First host name of a SLURM node list such as 'c207-[9-10]'."""
    out = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def init_distributed():
    """Rank, world size and local rank from torchrun or from srun, then the process group.

    Under srun the rendezvous address is the first node of the job step (rank 0 runs there),
    and the port depends on the job and step ids, so that consecutive steps of the same job
    never reuse a port that may still be in TIME_WAIT."""
    env = os.environ
    if "RANK" in env and "WORLD_SIZE" in env:                       # torchrun
        rank, world, local = int(env["RANK"]), int(env["WORLD_SIZE"]), int(env.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in env and "SLURM_STEP_ID" in env:          # srun, one task per GPU
        rank, world, local = int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"]), int(env["SLURM_LOCALID"])
        if "MASTER_ADDR" not in env:
            env["MASTER_ADDR"] = first_host(env.get("SLURM_STEP_NODELIST", env["SLURM_JOB_NODELIST"]))
        if "MASTER_PORT" not in env:
            env["MASTER_PORT"] = str(20000 + (int(env["SLURM_JOB_ID"]) * 7 + int(env["SLURM_STEP_ID"])) % 20000)
    else:                                                           # plain python: one process
        rank, world, local = 0, 1, 0
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")

    if torch.cuda.is_available():
        # srun may show each task both GPUs of the node or only its own: the modulo covers both
        torch.cuda.set_device(local % torch.cuda.device_count())
        device, backend = torch.device("cuda", torch.cuda.current_device()), "nccl"
    else:                                   # CPU only, to check the script on a login node
        device, backend = torch.device("cpu"), "gloo"
    # a long timeout: rank 0 evaluates alone at the end while the others wait in a barrier
    dist.init_process_group(backend, rank=rank, world_size=world, timeout=timedelta(minutes=30))
    return rank, world, local, device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- profiler
def comm_or_kernel_group(name):
    """The kernel groups of BASELINE, plus the NCCL kernels that do the all-reduce."""
    return "communication" if "nccl" in name.lower() else kernel_group(name)


def run_profiler(model, loader, optimizer, scaler, args, device, run_name, rank):
    """9 steps on every rank (the all-reduce needs all of them); rank 0 records the last 5 under
    torch.profiler, after 1 skipped and 3 warmup steps, saves a trace for TensorBoard and returns
    the GPU time of each kernel group per sample of that GPU.

    The communication kernels run on their own CUDA stream, overlapped with the backward pass,
    so the groups add up to more than the step time: their sum is GPU busy time, not wall time."""
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    model.train()
    it = iter(loader)
    active = 5
    trace_dir = os.path.join("profiles", run_name)
    if rank == 0:
        prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           schedule=schedule(wait=1, warmup=3, active=active, repeat=1),
                           on_trace_ready=tensorboard_trace_handler(trace_dir, use_gzip=True),
                           record_shapes=False)
    else:
        prof_ctx = contextlib.nullcontext()
    with prof_ctx as prof:
        for _ in range(9):
            batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in next(it).items()}
            train_step(model, batch, optimizer, scaler, args)
            if prof is not None:
                prof.step()
    if rank != 0:
        return None

    averages = prof.key_averages()
    print("\n===== torch.profiler (rank 0): top 15 by CUDA time =====")
    print(averages.table(sort_by="cuda_time_total", row_limit=15))

    def device_time(evt):
        t = getattr(evt, "self_device_time_total", None)
        return (evt.self_cuda_time_total if t is None else t) / 1000     # us -> ms

    groups = {}
    try:
        kernels = [e for e in averages if str(e.device_type).endswith("CUDA")
                   and not getattr(e, "is_user_annotation", False)
                   and not e.key.startswith(("ProfilerStep", "Optimizer."))]
        rows = kernels or [e for e in averages if device_time(e) > 0]
        for evt in rows:
            g = comm_or_kernel_group(evt.key)
            groups[g] = groups.get(g, 0.0) + device_time(evt)
    except Exception as e:                            # never lose a run because of the summary
        print(f"WARNING: could not group the GPU time: {e}")
    samples = active * args.batch_size
    summary = {g: round(ms / samples, 3) for g, ms in sorted(groups.items())}   # ms per sample
    print(f"GPU time per sample by kernel group (ms): {summary} | trace in {trace_dir}")
    print("===== end profiler =====\n", flush=True)
    return summary


# --------------------------------------------------------------------------- training
def train_step(model, batch, optimizer, scaler, args):
    """One step. DDP starts the all-reduce of each gradient bucket during backward(), as soon as
    the bucket is ready, so most of the communication overlaps with the computation."""
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
    return loss


def main():
    args = parse_args()
    rank, world, local, device = init_distributed()
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    log = print if rank == 0 else (lambda *a, **k: None)
    # same seed on every rank: identical initial weights (DDP broadcasts rank 0's anyway), and
    # the DistributedSampler draws the same permutation everywhere and gives each rank its part
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = args.precision == "tf32"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = load_tokenized(args, tokenizer)       # prepared once by BASELINE/prepare_data.sh
    data.set_format("torch")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_memory,
                     drop_last=True)
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    # each rank reads a different 1/world of the training set, every epoch reshuffled
    sampler = DistributedSampler(data["train"], num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True)
    train_loader = DataLoader(data["train"], sampler=sampler, **loader_kw)
    val_loader = DataLoader(data["validation"], shuffle=False, **loader_kw)    # rank 0 only

    raw_model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    n_emb = sum(p.numel() for n, p in raw_model.named_parameters() if "embeddings" in n)
    cfg = raw_model.config
    flops_per_token = 6 * (n_params - n_emb) + 6 * cfg.num_hidden_layers * args.max_len * cfg.hidden_size
    flops_per_sample = flops_per_token * args.max_len

    model = DDP(raw_model, device_ids=[device.index] if device.type == "cuda" else None,
                bucket_cap_mb=args.bucket_mb, gradient_as_bucket_view=True)
    if args.grad_compress == "bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    if args.compile:
        model = torch.compile(model)            # DDP first, then compile: graph breaks at the buckets

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    if args.zero:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(raw_model.parameters(), optimizer_class=torch.optim.AdamW,
                                            lr=args.lr, fused=args.fused_adam)
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, fused=args.fused_adam)
    warm = max(1, int(0.1 * total_steps))     # linear warmup (10 %) then linear decay, as in BASELINE
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    nodes = len(set(hosts))
    strategy = "zero1" if args.zero else "ddp"
    if args.grad_compress != "none":
        strategy += f"+{args.grad_compress}comm"
    global_batch = args.batch_size * world
    run_name = args.run_name or f"{strategy}_g{world}_n{nodes}_bs{args.batch_size}_{args.precision}"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    log(f"Run: {run_name} | {world} processes on {nodes} node(s): {', '.join(sorted(set(hosts)))} | {gpu_name}")
    log(f"torch {torch.__version__} (CUDA {torch.version.cuda}, NCCL "
        f"{'.'.join(map(str, torch.cuda.nccl.version())) if device.type == 'cuda' else '-'}) | "
        f"params {n_params/1e6:.1f} M | {flops_per_sample/1e9:.0f} GFLOP/sample")
    log(f"strategy {strategy} | batch {args.batch_size} per GPU, {global_batch} global | "
        f"steps/epoch {steps_per_epoch} | total steps {total_steps}", flush=True)

    profile_ms = None
    if args.profile:
        profile_ms = run_profiler(model, train_loader, optimizer, scaler, args, device, run_name, rank)

    # ---------------- training loop (the one of BASELINE, plus set_epoch and the loss reduction)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step, measure_t0, measured_steps, measured_time = 0, None, 0, 0.0
    loss_curve = []                             # (step, seconds, mean loss over all GPUs)
    running = torch.zeros((), device=device)
    sync(device)
    t_start = time.perf_counter()

    done = False
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
        sampler.set_epoch(epoch)
        for batch in train_loader:
            if step == args.warmup_steps:
                sync(device)
                measure_t0 = time.perf_counter()

            batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, scaler, args)
            lr_sched.step()
            running += loss.detach().float()
            step += 1

            if measure_t0 is not None and measured_steps == 0 and step == args.warmup_steps + args.measure_steps:
                sync(device)
                measured_time = time.perf_counter() - measure_t0
                measured_steps = args.measure_steps
            if step % args.log_every == 0:
                dist.all_reduce(running)          # one small collective every --log-every steps
                mean_loss = (running / (args.log_every * world)).item()
                loss_curve.append((step, round(time.perf_counter() - t_start, 2), round(mean_loss, 4)))
                log(f"  step {step:>6}/{total_steps}  epoch {epoch}  loss {mean_loss:.4f}", flush=True)
                running.zero_()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    sync(device)
    wall = time.perf_counter() - t_start

    # the slowest rank sets the pace: times and memory are the maximum over the GPUs
    stats = torch.tensor([wall, measured_time,
                          torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0],
                         device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    wall, measured_time, peak_mem = stats.tolist()

    # ---------------- results (rank 0)
    if rank == 0:
        config = {k: v for k, v in vars(args).items() if k not in ("data_dir", "results_dir")}
        res = dict(run=run_name, date=datetime.now().isoformat(timespec="seconds"),
                   hosts=sorted(set(hosts)), gpu=gpu_name, torch=torch.__version__,
                   nccl=".".join(map(str, torch.cuda.nccl.version())) if device.type == "cuda" else None,
                   **config, world_size=world, nodes=nodes, gpus_per_node=world // nodes, strategy=strategy,
                   global_batch_size=global_batch,
                   total_steps=step, wall_time_s=round(wall, 2),
                   avg_samples_per_s=round(step * global_batch / wall, 1),
                   params=n_params, non_embedding_params=n_params - n_emb,
                   gflops_per_sample=round(flops_per_sample / 1e9, 1), peak_mem_gb=round(peak_mem, 2))
        if measured_steps:
            sps = measured_steps * global_batch / measured_time          # all the GPUs together
            achieved_tflops = sps * flops_per_sample / 1e12
            res.update(samples_per_s=round(sps, 1), samples_per_s_per_gpu=round(sps / world, 1),
                       step_time_ms=round(1000 * measured_time / measured_steps, 1),
                       achieved_tflops=round(achieved_tflops, 1),
                       mfu_pct=round(100 * achieved_tflops / (PEAK_TFLOPS[args.precision] * world), 1),
                       peak_tflops_used=PEAK_TFLOPS[args.precision])
        else:
            print("WARNING: not enough steps to measure throughput (increase --max-steps).")
        res["loss_curve"] = loss_curve
        if profile_ms:
            res["profile_ms_per_sample"] = profile_ms
        # evaluation on rank 0 with the plain model: the weights are the same on every GPU
        if args.eval_samples:
            res["val_loss"] = round(evaluate(raw_model, val_loader, args, device), 4)
        if args.squad_eval:
            res.update(squad_evaluate(raw_model, tokenizer, args, device))

        os.makedirs(args.results_dir, exist_ok=True)
        out = os.path.join(args.results_dir, f"{run_name}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print("\n===== RESULT =====")
        for k in ("run", "world_size", "nodes", "global_batch_size", "samples_per_s", "samples_per_s_per_gpu",
                  "step_time_ms", "mfu_pct", "peak_mem_gb", "wall_time_s", "val_loss", "exact_match", "f1"):
            if k in res:
                print(f"  {k:<22} {res[k]}")
        print(f"  saved to {out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
