# HPC Tools Lab AI: accelerating BERT-base training on FinisTerrae III

Deliverables for the *HPC for AI* block of HPC Tools (Máster HPC, UDC/USC, 2026-27).

| Folder | Deliverable | Git tag |
|---|---|---|
| [`baseline/`](baseline/) | 1: single-GPU baseline and its optimization | `baseline` |
| `distributed/` | 2: multi-node, multi-GPU training | `distributed` |

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
