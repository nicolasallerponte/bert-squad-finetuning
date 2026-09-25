# Deliverable 1: BERT-base on SQuAD, single A100

> **TODO:** fill every `TODO` with the numbers from your runs. `python summarize.py results`
> prints the results table ready to paste.

## 1. Task

Fine-tune **BERT-base (uncased, 110 M parameters)** for extractive question answering on
**SQuAD v1.1** on a single **NVIDIA A100-PCIE-40GB** of FinisTerrae III, measure the training
time, and improve it with the single-GPU optimizations of the course, applied one at a time.

## 2. Files

| File | What it does |
|---|---|
| `train_qa.py` | The training script. Explicit PyTorch loop; every optimization is a command-line flag |
| `prepare_data.sh` | SLURM job (CPU): downloads the model and SQuAD and tokenizes the dataset once |
| `run_baseline.sh` | SLURM job: full epoch with the baseline configuration, with profiler |
| `run_improvements.sh` | SLURM job: applies the optimizations cumulatively and measures each step |
| `run_final.sh` | SLURM job: full epoch with every optimization |
| `summarize.py` | Builds the Markdown results table from `results/*.json` |
| `results/*.json` | Raw measurements of every run |

How to reproduce:

```bash
mkdir -p logs
sbatch prepare_data.sh          # once
sbatch run_improvements.sh      # step-by-step throughput
sbatch run_baseline.sh          # full epoch, baseline
sbatch run_final.sh             # full epoch, optimized
python summarize.py results
```

## 3. Methodology

- **Throughput** = samples per second over **100 training steps**, measured **after a warmup**
  (30 steps; 60 with `torch.compile`), because the first iterations pay for CUDA context
  creation, memory-pool growth and compilation. `torch.cuda.synchronize()` brackets the window.
- The loss is accumulated on the GPU and only read every 100 steps: calling `.item()` every
  step would force a CPU-GPU synchronization and distort the measurement.
- **MFU** = achieved FLOP/s ÷ peak FLOP/s of the precision in use, with the usual estimate of
  training cost `6 × parameters × tokens` (384 tokens per sample). A100 dense peaks: 19.5 TFLOP/s
  FP32, 156 TF32, 312 BF16/FP16.
- **FP32 is a true FP32 baseline**: TF32 is disabled unless requested.
- All batches have the same size (`drop_last=True`, padding to 384): static shapes, so
  `torch.compile` does not recompile.
- Every job gets exactly **32 CPU cores per A100**, which FT3's scheduler enforces.

## 4. Results

### 4.1 Optimizations, one at a time

TODO: paste the table from `python summarize.py results` (runs `s0` to `s4`).

| Step | Change | samples/s | Speedup | MFU | Peak memory |
|---|---|---|---|---|---|
| 0 | Baseline: FP32, batch 16, no workers | TODO | 1.00× | TODO | TODO |
| 1 | + 8 DataLoader workers, pinned memory | TODO | TODO | TODO | TODO |
| 2 | + BF16 mixed precision (`autocast`) | TODO | TODO | TODO | TODO |
| 3 | + batch size 32 | TODO | TODO | TODO | TODO |
| 4 | + `torch.compile` | TODO | TODO | TODO | TODO |

### 4.2 Full training (1 epoch)

| Configuration | Training time | Final validation loss |
|---|---|---|
| Baseline | TODO | TODO |
| Optimized | TODO | TODO |
| **Speedup** | **TODO** | |

### 4.3 Profiler

TODO: paste the `torch.profiler` table printed in `logs/bert-improvements_*.out` for `s0`
(baseline) and `s4` (optimized), and comment which kernels dominate before and after.

## 5. Discussion

TODO. Questions worth answering:

- Which single optimization gave the largest gain, and why? (Expected: BF16, the *«biggest single
  lever»*, because it moves the GEMMs to the tensor cores.)
- Did the input pipeline matter? If the tokenized dataset is already in memory, workers and
  pinned memory may add little; say so if that is what you measured.
- Why is the MFU far from 100 %? Element-wise operations (LayerNorm, GELU, dropout, softmax) are
  memory-bound; `torch.compile` fuses them.
- Why BF16 and not FP16? Same exponent range as FP32, so no `GradScaler` is needed on Ampere.
