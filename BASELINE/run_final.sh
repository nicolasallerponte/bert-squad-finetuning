#!/bin/bash
# Full training run (2 epochs) with ALL the optimizations, to compare against run_baseline.sh.
# Adjust the flags to the best configuration found with run_improvements.sh.
#   mkdir -p logs && sbatch run_final.sh
#SBATCH --job-name=bert-final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # FT3 requires exactly 32 CPUs per A100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail          # no -u: Lmod and venv activate use unset variables
module load cesga/2020 python/3.10.8
source "$STORE/mypython/bin/activate"
export HF_HOME="$STORE/hf_cache"
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Batch 64: 96 % of the throughput of batch 128 with half the memory (see run_improvements.sh).
OPT="--epochs 2 --precision bf16 --batch-size 64 --workers 8 --pin-memory --compile --fused-adam
     --warmup-steps 60 --squad-eval"
python train_qa.py $OPT --run-name final_full

echo "Finished at $(date)"
