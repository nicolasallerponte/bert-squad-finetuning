#!/bin/bash
# Full training run of the BASELINE (1 epoch) on one A100.
#   mkdir -p logs && sbatch run_baseline.sh
#SBATCH --job-name=bert-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # FT3 requires exactly 32 CPUs per A100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00             # < 6 h -> 'short' QoS, highest priority
#SBATCH --output=logs/%x_%j.out

set -eo pipefail          # no -u: Lmod and venv activate use unset variables
module load cesga/2020 python/3.10.8
source "$STORE/mypython/bin/activate"
export HF_HOME="$STORE/hf_cache"     # model + dataset cache out of $HOME (20 GB quota)
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Baseline: FP32, no DataLoader workers, no pinned memory, batch 16, no compilation.
python train_qa.py --epochs 1 --batch-size 16 --precision fp32 --workers 0 \
                   --run-name baseline_full --profile

echo "Finished at $(date)"
