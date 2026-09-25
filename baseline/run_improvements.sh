#!/bin/bash
# Apply the optimizations ONE AT A TIME, cumulatively, in the order recommended in class,
# and measure the throughput after each one (samples/s over 100 steps, after warmup).
#   mkdir -p logs && sbatch run_improvements.sh
#SBATCH --job-name=bert-improvements
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

STEPS="--max-steps 300 --warmup-steps 30 --measure-steps 100"

run () { echo; echo "################ $1"; shift; python train_qa.py $STEPS "$@"; }

run "0 baseline"                --run-name s0_baseline  --precision fp32 --batch-size 16 --workers 0 --profile
run "1 + input pipeline"        --run-name s1_pipeline  --precision fp32 --batch-size 16 --workers 8 --pin-memory
run "2 + bf16 mixed precision"  --run-name s2_bf16      --precision bf16 --batch-size 16 --workers 8 --pin-memory
run "3 + larger batch"          --run-name s3_batch32   --precision bf16 --batch-size 32 --workers 8 --pin-memory
run "4 + torch.compile"         --run-name s4_compile   --precision bf16 --batch-size 32 --workers 8 --pin-memory \
                                --compile --warmup-steps 60 --profile

echo; python summarize.py results
echo "Finished at $(date)"
