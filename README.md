# HPC Tools Lab AI: accelerating BERT-base training on FinisTerrae III

Deliverables for the *HPC for AI* block of HPC Tools (Máster HPC, UDC/USC, 2026-27).

| Folder | Deliverable | Git tag |
|---|---|---|
| [`BASELINE/`](BASELINE/) | 1: single-GPU baseline and its optimization | `BASELINE` |
| `DISTRIBUTED/` | 2: multi-node, multi-GPU training | `DISTRIBUTED` |

## Repository structure

Each deliverable lives in its own folder, and the commit tagged with the folder's name contains
everything needed to reproduce it: the Python code, the SLURM jobs, the raw results and the report.

```
.
├── README.md               this file: structure and environment
├── requirements.txt        Python dependencies (PyTorch is installed apart, see below)
└── BASELINE/               deliverable 1, tag BASELINE
    ├── README.md           the report
    ├── report/             the same report in LaTeX, and its PDF
    ├── train_qa.py         training script; every optimization is a command-line flag
    ├── prepare_data.sh     SLURM job (CPU): downloads the model and tokenizes SQuAD once
    ├── run_improvements.sh SLURM job: applies the optimizations one at a time, 3 runs each
    ├── run_baseline.sh     SLURM job: full training, baseline configuration
    ├── run_final.sh        SLURM job: full training, optimized configuration
    ├── summarize.py        builds the results table from results/
    ├── plots.py            draws the figures of the report from results/ and profiles/
    ├── results/            one JSON file per run: throughput, memory, loss curve, EM and F1
    │   └── 2026-09-24-first-round/   an earlier round, one run per step, not used in the report
    ├── profiles/           torch.profiler traces, one per optimization step (TensorBoard)
    └── figures/            figures of the report
```

The workflow is: prepare the data once, run the three GPU jobs (they are independent and can run
at the same time), and build the table and figures from their results. The exact commands are in
[`BASELINE/README.md`](BASELINE/README.md#2-files).

## Environment (FinisTerrae III)

```bash
module load cesga/2020 python/3.10.8
python3 -m venv $STORE/mypython
source $STORE/mypython/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8: FT3 driver 570.86.15
pip install -r requirements.txt
pip cache purge
```

Every job loads the same module and activates the same environment before running.
