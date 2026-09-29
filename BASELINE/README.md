# Deliverable 1: BERT-base on SQuAD, single A100

We fine-tune BERT-base for question answering on SQuAD v1.1 on one A100 of FinisTerrae III,
measure the training time, and then apply the single-GPU optimizations of the course one at a
time, profiling each of them.

**Result:** two epochs take **45.8 min** in the FP32 baseline and **7.2 min** once optimized, a
**6.4x** speedup, at the cost of one point of F1 (87.97 to 86.97).

The same report, as a PDF: [`report/report.pdf`](report/report.pdf).

## 1. Setup and method

| | |
|---|---|
| Model | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased), 109 M parameters |
| Data | SQuAD v1.1, 384-token windows with a 128-token stride: 88,492 training and 10,753 validation features |
| Training | 2 epochs, AdamW, learning rate 3e-5, 10 % linear warmup then linear decay |
| Software | PyTorch 2.11.0+cu128, Python 3.10.8, virtual environment in `$STORE` |

- **Throughput** (samples/s) is measured over 100 steps after a warmup of 30 steps (60 with
  `torch.compile`), between two `torch.cuda.synchronize()` calls.
- Each configuration runs **3 times**; we report the **median** and the range.
- `torch.profiler` records 5 steps per configuration. We report the **share** of GPU time per
  kernel group, which is comparable across configurations; absolute times are not, since the
  profiled steps are the first ones of each run.
- The baseline is **true FP32** (TF32 disabled). Quality is measured with the official SQuAD
  **exact match (EM) and F1** on the full validation set.

## 2. Files

| File | Purpose |
|---|---|
| `train_qa.py` | Training script: an explicit PyTorch loop in which every optimization is a flag |
| `prepare_data.sh` | CPU job that downloads the model and tokenizes SQuAD once |
| `run_improvements.sh` | Applies the optimizations cumulatively, 3 repetitions each, profiling the first |
| `run_baseline.sh`, `run_final.sh` | Full 2-epoch trainings, baseline and optimized, with EM and F1 |
| `summarize.py`, `plots.py` | Table and figures of this report, from `results/*.json` |
| `results/`, `profiles/`, `figures/` | Raw measurements, `torch.profiler` traces and figures |
| `report/` | This report in LaTeX, and its PDF |

```bash
sbatch prepare_data.sh                                            # once
sbatch run_improvements.sh; sbatch run_baseline.sh; sbatch run_final.sh
python summarize.py results; python plots.py results
tensorboard --logdir profiles                                     # needs torch-tb-profiler
```

## 3. Running on FinisTerrae III

### Resources requested

| Job | GPU | Cores | Memory | Time limit | Actual time |
|---|---|---|---|---|---|
| `prepare_data.sh` | none | 8 | 4 GB | 0.5 h | < 0.5 h |
| `run_improvements.sh` | 1 A100 | 32 | 64 GB | 3 h | 31 min |
| `run_baseline.sh` | 1 A100 | 32 | 64 GB | 3 h | 48 min |
| `run_final.sh` | 1 A100 | 32 | 64 GB | 2 h | 10 min |

- **One A100 per job**, as the deliverable requires: the training script uses a single GPU, so
  more GPUs would sit idle.
- **32 cores**, because FT3 rejects GPU jobs with any other count per A100; eight of them run
  the DataLoader workers.
- **64 GB of memory**, a generous margin for the tokenized dataset held in memory and the eight
  worker processes.
- **Time limits** of two to three times the expected duration, so that a slow node does not
  kill a run, and always below 6 h, which places the jobs in the `short` QoS, the one with the
  highest priority.
- **Tokenization runs in a CPU-only job** (8 processes, 0.5 GB peak), so no GPU time is spent
  on it and it is not counted as training time.
- The three GPU jobs are independent, so they were submitted at once on three GPUs: the whole
  set of measurements took 53 minutes.

### Deployment issues

| Issue | Solution |
|---|---|
| The default PyTorch wheel targets CUDA 13, but the A100 driver (570.86.15) supports up to 12.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `python/3.10.10` is not available | `module load cesga/2020 python/3.10.8`, in the venv and in every job |
| A GPU job with 4 cores is rejected: *«CPUs requested should be (nodes \* gpus/node \* 32)»* | Exactly 32 cores per A100 |
| `$HOME` quota is 20 GB | venv and Hugging Face cache (`HF_HOME`) in `$STORE` |

## 4. Results

### 4.1 One optimization at a time

The optimizations are applied cumulatively, in the order seen in class. The three repetitions
agree within 1.5 %, except one baseline run 7 % slower, which is why we report the median.

![Throughput after each optimization](figures/throughput.png)

| Step | Change | samples/s, median [range] | Speedup | Peak memory |
|---|---|---|---|---|
| 0 | Baseline: FP32, batch 16, no workers | 64.3 [60.0, 64.7] | 1.00x | 5.3 GB |
| 1 | + 8 workers, pinned memory | 64.4 [64.4, 65.2] | 1.00x | 5.3 GB |
| 2 | + TF32 | 194.6 [194.6, 195.4] | 3.03x | 5.3 GB |
| 3 | + BF16 mixed precision | 289.0 [287.5, 291.5] | 4.49x | 4.1 GB |
| 4 | + batch 32 | 342.2 [342.2, 347.7] | 5.32x | 6.6 GB |
| 5 | + `torch.compile` | 397.0 [396.1, 402.4] | 6.17x | 6.0 GB |
| 6 | + fused AdamW | 430.1 [430.0, 436.2] | 6.69x | 6.0 GB |
| 7 | + batch 64 | 460.2 [460.0, 468.3] | 7.16x | 10.4 GB |
| 8 | + batch 128 | 470.4 [470.2, 477.5] | 7.32x | 19.3 GB |

**Input pipeline (step 1): no gain.** The dataset is tokenized beforehand and kept in memory, so
building a batch costs almost nothing next to a 249 ms training step. The GPU never waits for
data, and workers and pinned memory have nothing to hide.

**Precision (steps 2 and 3): the largest gains.** TF32 alone triples throughput without touching
the model, because it runs the FP32 matrix products on the Tensor Cores (peak 19.5 to 156
TFLOP/s). BF16 mixed precision adds another 1.5x: it also halves the bytes moved, and lowers
peak memory from 5.3 to 4.1 GB. Since BF16 keeps the FP32 exponent range, no loss scaling is
needed.

**Batch size (steps 4, 7 and 8): diminishing returns.** Larger batches feed larger matrices to
the Tensor Cores and spread the fixed cost of each step over more samples. The gain shrinks
quickly: +18 % for batch 32, +7 % for 64 and only +2 % for 128, which doubles the memory. The
final run therefore uses batch 64.

**Compilation and fused optimizer (steps 5 and 6).** `torch.compile` adds 16 % and fused AdamW
another 8 %. The profile explains why.

### 4.2 Where the GPU time goes

![Share of GPU time per kernel group](figures/profile.png)

Kernels launched inside the optimizer step count as optimizer; memory copies stay below 0.5 %.

In FP32, matrix products take 82 % of the GPU time. Once they run on the Tensor Cores, their
share falls to about 40 %, and kernels that were negligible become visible: speeding up one part
makes the rest weigh more, as Amdahl's law predicts.

Each later optimization targets exactly that remainder. `torch.compile` fuses the element-wise
operations (GELU, dropout, residual additions, LayerNorm) into fewer kernels, cutting their share
from 34 to 23 %. The optimizer then accounts for 11 %, because the standard AdamW launches many
small kernels; the fused version brings it down to 4 %, and larger batches amortize it further
(1 % at batch 128).

In the final configuration, **attention is the largest cost (32 %)**. PyTorch selects the
memory-efficient attention kernel rather than FlashAttention, because BERT passes a padding mask,
and its backward pass alone takes 23 % of the GPU time. This is where the next improvement would
have to come from.

### 4.3 Full training

![Training loss against wall-clock time](figures/loss.png)

| | Baseline (step 0) | Optimized (step 7) |
|---|---|---|
| Training time, 2 epochs | 2,747 s (45.8 min) | **431 s (7.2 min)** |
| Mean throughput | 64.4 samples/s | 410.1 samples/s |
| EM / F1 | 80.34 / 87.97 | 79.13 / 86.97 |
| Peak memory | 5.3 GB | 10.4 GB |

**Time improves 6.4x, less than throughput (7.3x).** The optimized run spends about 56 s outside
the steady state, mostly compiling: its curve starts a minute late. This fixed cost weighs more
in a 7-minute run than it would in a longer one. Both curves drop at the start of the second
epoch, when the model sees the training data again.

**Quality is preserved.** The baseline matches the published BERT-base results (80.8 EM,
88.5 F1). The optimized run loses about one point, plausibly because batch 64 makes four times
fewer optimizer updates at the same learning rate; tuning it is outside the scope of this
deliverable.

**Utilization.** Counting 204 GFLOP per training sample (Kaplan et al., 2020), the baseline
reaches 13 TFLOP/s, 67 % of the FP32 peak, and the optimized run 96 TFLOP/s, 31 % of the BF16
peak: a lower fraction of a peak that is 16 times higher.

## 5. Conclusions

- Precision is the decisive lever: it accounts for 4.5x of the 7.3x gain in throughput.
- Each optimization exposes the next bottleneck, and profiling after every step shows which
  one: now it is attention.
- The code was the easy part; most of the work was deploying it correctly on FinisTerrae III.
