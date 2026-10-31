#!/bin/bash
# Full training runs (2 epochs) on 2 nodes x 2 A100, with exact match and F1 at the end, to
# compare against BASELINE/run_final.sh (7.2 min on one A100, EM 79.13, F1 86.97).
#   mkdir -p logs && sbatch run_full.sh
#SBATCH --job-name=bert-full
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2         # one process per GPU
#SBATCH --cpus-per-task=32          # FT3 requires exactly 32 CPUs per A100
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail          # no -u: Lmod and venv activate use unset variables
module load cesga/2020 python/3.10.8
source "$STORE/mypython/bin/activate"
export HF_HOME="$STORE/hf_cache"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST at $(date)"
OPT="--epochs 2 --precision bf16 --workers 8 --pin-memory --compile --fused-adam
     --warmup-steps 60 --squad-eval"

# 1. The same optimization problem as BASELINE: global batch 64 (16 per GPU), learning rate 3e-5,
#    2764 steps. Only the number of GPUs changes, so EM and F1 must stay where they were.
srun python train_ddp.py $OPT --batch-size 16 --lr 3e-5 --run-name full_g4_global64 \
  || echo "FAILED: full_g4_global64"

# 2. 64 per GPU, global batch 256: each GPU works at its efficient batch, with 4 times fewer
#    steps (690). The learning rate grows with the square root of the batch (3e-5 x 2).
srun python train_ddp.py $OPT --batch-size 64 --lr 6e-5 --run-name full_g4_global256 \
  || echo "FAILED: full_g4_global256"

echo; python summarize.py results
echo "Finished at $(date)"
