#!/bin/bash
# Download BERT-base + SQuAD and tokenize the dataset ONCE, so that the download is not
# counted in any training time. Runs on a CPU node (no GPU needed).
#   mkdir -p logs && sbatch prepare_data.sh
#SBATCH --job-name=bert-prepare
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail          # no -u: Lmod and venv activate use unset variables
module load cesga/2020 python/3.10.8
source "$STORE/mypython/bin/activate"
export HF_HOME="$STORE/hf_cache"
cd "$SLURM_SUBMIT_DIR"

python train_qa.py --prepare-only
