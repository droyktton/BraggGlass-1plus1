#!/bin/bash
# Single run of coupled_chains_sim on one GPU.
#
# Usage:
#   sbatch slurm/run_single.slurm <seedD> <seedT> [params_file]
#   sbatch --gres=gpu:rtx3080:1 slurm/run_single.slurm 1234 1234   # pin GPU type
#
#SBATCH --job-name=bragg-glass
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if command -v conda >/dev/null 2>&1; then
    set +u  # conda's (de)activate hooks reference unset vars under `set -u`
    eval "$(conda shell.bash hook)"
    conda activate base
    set -u
fi

echo "Node: $(hostname)"
nvidia-smi -L || true

SEED_D=${1:?"usage: sbatch run_single.slurm <seedD> <seedT> [params_file]"}
SEED_T=${2:?"usage: sbatch run_single.slurm <seedD> <seedT> [params_file]"}
PARAMS=${3:-params.ini}

./coupled_chains_sim "$SEED_D" "$SEED_T" --params "$PARAMS"
