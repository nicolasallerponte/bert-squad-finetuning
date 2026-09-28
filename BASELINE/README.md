# Deliverable 1: BERT-base on SQuAD, single A100

## 1. Task

Fine-tune **BERT-base (uncased)** for extractive question answering on **SQuAD v1.1** on a
single **NVIDIA A100-PCIE-40GB** of FinisTerrae III, measure the training time, and improve it
with the single-GPU optimizations of the course, applied one at a time.

| | |
|---|---|
| Model | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) (loaded by its short name `bert-base-uncased`): 108.9 M parameters, 85.1 M of them outside the embeddings |
| Data | SQuAD v1.1, windows of 384 tokens with a stride of 128: **88,492 training** and 10,753 validation features |
| Training | AdamW, learning rate 3e-5 with 10 % linear warmup and linear decay, **2 epochs** |
| Hardware | 1 x A100-PCIE-40GB (driver 570.86.15, CUDA 12.8), 32 CPU cores |
| Software | PyTorch 2.11.0+cu128, Python 3.10.8 |

## 2. Files

| File | What it does |
|---|---|
| `train_qa.py` | Training script. Explicit PyTorch loop; every optimization is a command-line flag |
| `prepare_data.sh` | SLURM job (CPU only): downloads the model and SQuAD and tokenizes the dataset once |
| `run_improvements.sh` | SLURM job: applies the optimizations cumulatively and measures each step |
| `run_baseline.sh` | SLURM job: full training (2 epochs) with the baseline configuration |
| `run_final.sh` | SLURM job: full training (2 epochs) with every optimization |
| `summarize.py` | Builds the results table from `results/*.json` |
| `results/` | Raw measurements of every run (JSON); the runs reported below are in `results/2026-09-24-first-round/` |

To reproduce:

```bash
mkdir -p logs
sbatch prepare_data.sh          # once
sbatch run_improvements.sh      # step-by-step throughput
sbatch run_baseline.sh          # full training, baseline
sbatch run_final.sh             # full training, optimized
python summarize.py results
```

## 3. Methodology

- **Throughput** is measured in samples per second over **100 training steps**, **after a
  warmup** of 30 steps (60 with `torch.compile`), because the first iterations pay for CUDA
  context creation, memory pool growth and compilation. `torch.cuda.synchronize()` brackets the
  measured window.
- The loss is accumulated on the GPU and read only every 100 steps: calling `.item()` at every
  step would force a CPU-GPU synchronization and distort the measurement.
- **Training cost** follows Kaplan et al. (2020): `6 x N + 6 x n_layers x n_ctx x d_model`
  FLOPs per token, with `N` the **non-embedding** parameters (embedding lookups perform no matrix
  multiplication). For BERT-base at 384 tokens this is **204 GFLOP per training sample**.
- **MFU** = achieved FLOP/s divided by the A100 dense peak of the precision in use: 19.5 TFLOP/s
  FP32, 312 TFLOP/s BF16.
- **FP32 is a true FP32 baseline**: TF32 is explicitly disabled.
- Every batch has the same shape (padding to 384, `drop_last=True`), so `torch.compile` never
  recompiles.
- The **validation loss** is always computed on the **same 1,600 validation samples** (the first
  ones of the split, no shuffling), whatever the batch size, so the full runs are comparable.

## 4. Deployment on FinisTerrae III

Most of the work was getting the environment right on the cluster. What we found:

| Issue | Symptom | Solution |
|---|---|---|
| **CUDA version of the driver** | `pip install torch` installs the CUDA 13 build, but the A100 driver (570.86.15) supports **up to CUDA 12.8**: PyTorch would not see the GPU | Install from the CUDA 12.8 index: `pip install torch --index-url https://download.pytorch.org/whl/cu128`. Checked on a GPU node: `torch.cuda.is_available()` returns `True` |
| **Python module** | `python/3.10.10` is not available under `cesga/system` | `module load cesga/2020 python/3.10.8`, loaded both when creating the virtual environment and in every job |
| **CPUs per GPU** | Requesting 4 cores with one A100 is rejected: *«CPUs requested should be (nodes \* gpus/node \* 32)»* | Exactly **32 CPU cores per A100** in every GPU job |
| **Quota** | `$HOME` has 20 GB, and PyTorch with its CUDA libraries takes several GB | Virtual environment and Hugging Face cache (`HF_HOME`) in `$STORE` |
| **Data preparation time** | Download and tokenization would be counted as training time | A separate CPU job tokenizes once and saves the dataset to `$STORE`; training jobs load it from disk |
| **Queue** | Long jobs wait more | Every job stays under 6 h and runs in the `short` QoS, the one with the highest priority |

## 5. Results

### 5.1 Optimizations, one at a time

300 steps per configuration, throughput over 100 steps after warmup (job 9980447).

| Step | Change | samples/s | Speedup | TFLOP/s | MFU | Peak memory |
|---|---|---|---|---|---|---|
| 0 | Baseline: FP32, batch 16, no DataLoader workers | 64.5 | 1.00x | 13.2 | 67.5 % | 5.34 GB |
| 1 | + 8 DataLoader workers, pinned memory | 63.9 | 0.99x | 13.0 | 66.9 % | 5.34 GB |
| 2 | + BF16 mixed precision (`autocast`) | 297.3 | **4.61x** | 60.7 | 19.5 % | 4.10 GB |
| 3 | + batch size 32 | 346.1 | 5.37x | 70.6 | 22.6 % | 6.64 GB |
| 4 | + `torch.compile` | 406.5 | 6.30x | 83.0 | 26.6 % | 5.98 GB |
| 5 | + batch size 64 | 442.3 | 6.86x | 90.3 | 28.9 % | 10.43 GB |
| 6 | + batch size 128 | 460.0 | 7.13x | 93.9 | 30.1 % | 19.33 GB |

The validation losses of these short runs are not comparable with each other: after 300 steps,
larger batches have simply seen more samples. Quality is compared in 5.2.

### 5.2 Full training (2 epochs)

| Configuration | Batch | Throughput | Training time | Validation loss |
|---|---|---|---|---|
| Baseline (FP32, no workers) | 16 | 65.9 samples/s | **2,739.8 s (45.7 min)** | 0.961 |
| Optimized (BF16, workers, `torch.compile`) | 64 | 463.1 samples/s | **446.1 s (7.4 min)** | 0.990 |
| **Speedup** | | **7.03x** | **6.14x** | +3.0 % |

The baseline ran 11,060 steps (5,530 per epoch) and the optimized run 2,764 (1,382 per epoch).
Both evaluated the same 1,600 validation samples.

### 5.3 Profiler

5 profiled steps with `torch.profiler`, top CUDA operations.

Matrix products are compared at operation level (`aten::mm` in the backward pass plus
`aten::addmm` in the linear layers of the forward pass); the kernel names show which hardware
units run them. Only the top 15 operations are listed by the profiler.

**Baseline (FP32, batch 16)**

| Operation | CUDA time | Share of GPU time |
|---|---|---|
| Matrix products (`aten::mm` + `aten::addmm`) | 993.9 ms | **69.9 %** |
| ... run by `ampere_sgemm_*` kernels (FP32, regular CUDA cores) | | |
| Attention backward (`fmha_cutlassB_f32`) | 86.8 ms | 6.1 % |
| **Total CUDA time** | **1,423 ms** | **17.8 ms per sample** |

**Optimized (BF16, batch 32, `torch.compile`)**

| Operation | CUDA time | Share of GPU time |
|---|---|---|
| Matrix products (`aten::mm` + `aten::addmm`) | 157.2 ms | **32.9 %** |
| ... run by `ampere_bf16_s16816gemm_*` kernels (Tensor Cores) | | |
| Attention backward (`fmha_cutlassB_bf16`) | 93.8 ms | **19.7 %** |
| **Total CUDA time** | **477 ms** | **3.0 ms per sample** |

Per sample, the profiled GPU time goes from 17.8 ms to 3.0 ms (5.96x), consistent with the
6.30x throughput gain of step 4.

## 6. Discussion

**The input pipeline was not a bottleneck (step 1, 0.99x).** The dataset is tokenized once and
kept in memory, so building a batch costs almost nothing compared with a 248 ms FP32 training
step: the GPU never waits for data. Workers and pinned memory are the right tools when the data
has to be read and decoded during training; here there was nothing for them to hide.

**BF16 mixed precision is the largest single gain (4.61x).** The profiler shows why. In FP32,
70 % of the GPU time goes to matrix products, run by `ampere_sgemm_*` kernels on the regular
CUDA cores. With `autocast` to BF16 the same products run on
`ampere_bf16_s16816gemm_*` kernels, which use the **Tensor Cores** (the `16816` is the shape of
the Tensor Core MMA instruction). The dense peak goes from 19.5 to 312 TFLOP/s. BF16 keeps the
8-bit exponent of FP32, so no `GradScaler` is needed, unlike FP16. Peak memory also drops
(5.34 to 4.10 GB), because activations are stored in 16 bits.

**Why the MFU drops while the speed grows.** MFU is relative to the peak of the precision in use:
the FP32 baseline reaches 67.5 % of a 19.5 TFLOP/s peak, while BF16 reaches 19.5 % to 30.1 % of a
peak 16 times higher. In absolute terms the GPU does 13.2 TFLOP/s in the baseline and 93.9 in the
best configuration.

**Larger batches help, with diminishing returns (steps 3, 5, 6).** Larger matrix products use
the Tensor Cores better, and the fixed cost of every step (optimizer update, kernel launches,
Python overhead) is spread over more samples. Going from batch 64 to 128 gains only 4 % for twice
the memory. The final run uses **batch 64**: batch 128 would also mean 8 times fewer optimizer
updates than the baseline at the same learning rate, which could degrade the model for a very
small speed gain.

**`torch.compile` adds 17 % on top of BF16 (step 4).** A transformer is not only matrix products:
LayerNorm, GELU, dropout, softmax and residual additions are element-wise and memory-bound.
Compilation fuses them into fewer kernels, which also lowers peak memory (6.64 to 5.98 GB at
batch 32) because fewer intermediate tensors are kept.

**Amdahl in the profile.** Once the matrix products run on Tensor Cores, attention becomes a much
larger share of the time: the attention backward kernel goes from 6.1 % to 19.7 % of the GPU time.
Speeding up one part makes the rest weigh more; attention is now the next target.

**Training time improves less than throughput (6.14x against 7.03x).** Multiplying the steps by
the measured time per step gives 2,685 s for the baseline and 382 s for the optimized run; the
rest of the wall time (54 s and 64 s) are fixed costs: start-up, warmup and, above all,
compilation. They are about the same in both runs, but they weigh far more in a 7-minute run than
in a 45-minute one. This is Amdahl's law again: the part that was not accelerated limits the
total speedup, and it would matter less in a longer training.

**The model quality is preserved.** On the same 1,600 validation samples the loss is 0.990,
3 % above the baseline (0.961). The optimized run makes 4 times fewer optimizer updates (batch 64
against 16) with the same learning rate; scaling the learning rate with the batch size would
likely close the gap, but tuning hyperparameters is outside the scope of this deliverable.

**Why the MFU stays around 30 %.** Matrix products are fast, but the rest of the step is not:
attention, memory-bound element-wise operations, the optimizer update and the per-step CPU
overhead. This matches the typical 20-40 % for a full training job, even when individual matrix
kernels are close to their peak.
