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
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail          # no -u: Lmod and venv activate use unset variables
module load cesga/2020 python/3.10.8
source "$STORE/mypython/bin/activate"
export HF_HOME="$STORE/hf_cache"
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

STEPS="--max-steps 300 --warmup-steps 30 --measure-steps 100"
PIPE="--workers 8 --pin-memory"
REPS=3                    # every step is measured 3 times; the report gives the median

run () { echo; echo "################ $1"; shift; python train_qa.py $STEPS "$@"; }

# The repetitions are interleaved (all the steps, then all of them again) so that any drift of
# the node during the job affects every step alike. The profiler runs in the first repetition.
for r in $(seq 1 $REPS); do
  P=""; [ "$r" -eq 1 ] && P="--profile"
  run "0 baseline, rep $r"         --run-name s0_baseline_r$r   --precision fp32 --batch-size 16 --workers 0 $P
  run "1 + input pipeline, rep $r" --run-name s1_pipeline_r$r   --precision fp32 --batch-size 16 $PIPE $P
  run "2 + TF32, rep $r"           --run-name s2_tf32_r$r       --precision tf32 --batch-size 16 $PIPE $P
  run "3 + BF16, rep $r"           --run-name s3_bf16_r$r       --precision bf16 --batch-size 16 $PIPE $P
  run "4 + batch 32, rep $r"       --run-name s4_batch32_r$r    --precision bf16 --batch-size 32 $PIPE $P
  run "5 + torch.compile, rep $r"  --run-name s5_compile_r$r    --precision bf16 --batch-size 32 $PIPE \
                                   --compile --warmup-steps 60 $P
  run "6 + fused AdamW, rep $r"    --run-name s6_fused_adam_r$r --precision bf16 --batch-size 32 $PIPE \
                                   --compile --fused-adam --warmup-steps 60 $P
  run "7 + batch 64, rep $r"       --run-name s7_batch64_r$r    --precision bf16 --batch-size 64 $PIPE \
                                   --compile --fused-adam --warmup-steps 60 $P
  run "8 + batch 128, rep $r"      --run-name s8_batch128_r$r   --precision bf16 --batch-size 128 $PIPE \
                                   --compile --fused-adam --warmup-steps 60 $P
done

echo; python summarize.py results
echo "Finished at $(date)"
