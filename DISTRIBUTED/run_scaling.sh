#!/bin/bash
# Scaling of the training on 1, 2 and 4 A100, inside one allocation of 2 nodes x 2 GPUs:
# each measurement is a job step (srun) that uses part or all of the allocation.
#   mkdir -p logs && sbatch run_scaling.sh
#SBATCH --job-name=bert-scaling
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2         # one process per GPU
#SBATCH --cpus-per-task=32          # FT3 requires exactly 32 CPUs per A100
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G                  # per node: two processes, 8 DataLoader workers each
#SBATCH --time=02:30:00             # < 6 h -> 'short' QoS, highest priority
#SBATCH --output=logs/%x_%j.out

set -eo pipefail          # no -u: Lmod and venv activate use unset variables
module load cesga/2020 python/3.10.8
source "$STORE/mypython/bin/activate"
export HF_HOME="$STORE/hf_cache"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK     # srun does not inherit --cpus-per-task
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST at $(date)"
# once per node; diagnostics only, so they never stop the job
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
srun --nodes=1 --ntasks=1 nvidia-smi topo -m || true

# The optimized single-GPU configuration of BASELINE (run_final.sh), for 160 steps: 60 of warmup
# (torch.compile) and 100 measured
OPT="--precision bf16 --workers 8 --pin-memory --compile --fused-adam
     --max-steps 200 --warmup-steps 60 --measure-steps 100 --eval-samples 0"
REPS=3

# run NAME NODES PROCESSES [training flags]: one job step; a failed step does not stop the job
run () {
  local name=$1 nodes=$2 tasks=$3; shift 3
  echo; echo "################ $name: $tasks GPU on $nodes node(s)"
  srun --nodes="$nodes" --ntasks="$tasks" --ntasks-per-node=$(( (tasks + nodes - 1) / nodes )) \
       python train_ddp.py $OPT --run-name "$name" "$@" || echo "FAILED: $name"
}

# The first 4-GPU run logs how NCCL connects the GPUs (inside a node and between the nodes)
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
run nccl_check 2 4 --batch-size 64 --max-steps 30 --warmup-steps 10 --measure-steps 10
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS

# Repetitions interleaved, as in BASELINE, so that any drift affects every configuration alike.
# The profiler runs in the first repetition.
for r in $(seq 1 $REPS); do
  P=""; [ "$r" -eq 1 ] && P="--profile"
  # weak scaling: 64 samples per GPU, the global batch grows with the GPUs
  run weak_g1_r$r         1 1 --batch-size 64 $P
  run weak_g2_1node_r$r   1 2 --batch-size 64 $P
  run weak_g2_2nodes_r$r  2 2 --batch-size 64 $P      # same 2 GPUs, but through the network
  run weak_g4_r$r         2 4 --batch-size 64 $P
  # strong scaling: global batch 64, as in BASELINE, split among the GPUs
  run strong_g2_r$r       1 2 --batch-size 32 $P
  run strong_g4_r$r       2 4 --batch-size 16 $P
  # two alternatives to plain DDP, on the 4 GPUs
  run zero1_g4_r$r        2 4 --batch-size 64 --zero $P
  run bf16comm_g4_r$r     2 4 --batch-size 64 --grad-compress bf16 $P
done

echo; python summarize.py results
echo "Finished at $(date)"
