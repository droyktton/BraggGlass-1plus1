#!/bin/bash
# Disorder-seed scan, parallelized across the gpu partition (10 GPUs total:
# 4x RTX3080 + 3x RTX5070Ti + 2x RTX3080Ti + 1x A10). Each array task runs
# one disorder seed on one GPU in its own results/seed_<N>/ subdirectory —
# this isolation matters because every run writes the same filenames
# (correlation_replica_<T>.dat etc.), so tasks running concurrently in a
# shared directory would overwrite each other's output.
#
# Usage:
#   sbatch --array=0-9 slurm/run_array.slurm [seedT] [params_file]
#   sbatch --array=0-9%4 slurm/run_array.slurm   # cap at 4 concurrent tasks
#
#SBATCH --job-name=bragg-glass-scan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if command -v conda >/dev/null 2>&1; then
    set +u  # conda's (de)activate hooks reference unset vars under `set -u`
    eval "$(conda shell.bash hook)"
    conda activate base
    set -u
fi

echo "Node: $(hostname), array task: $SLURM_ARRAY_TASK_ID"
nvidia-smi -L || true

SEED_T=${1:-1234}
PARAMS=${2:-params.ini}
SEED_D=$((1234 + SLURM_ARRAY_TASK_ID))

RUNDIR="results/seed_${SEED_D}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

"$SLURM_SUBMIT_DIR/coupled_chains_sim" "$SEED_D" "$SEED_T" \
    --params "$SLURM_SUBMIT_DIR/$PARAMS"
