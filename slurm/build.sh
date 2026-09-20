#!/usr/bin/env bash
# One-time build for the cluster's "gpu" partition, which mixes Ampere
# (RTX 3080, RTX 3080Ti, A10 -> sm_86) and Blackwell (RTX 5070Ti -> sm_120)
# nodes. Building both targets into one fat binary means jobs don't need to
# care which node type they land on. Run this on the login node — no GPU
# needs to be present at compile time since the architectures are explicit.
#
# Usage:
#   ./slurm/build.sh
#   EXTRA_DEFS="-DLARKIN -DNREPLICAS=5" ./slurm/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The site "cuda/11.1.0" module predates Blackwell support; the CUDA 13.0
# toolkit in the conda base env is what has sm_120. Adjust here if your
# environment differs.
if command -v conda >/dev/null 2>&1; then
    set +u  # conda's (de)activate hooks reference unset vars under `set -u`
    eval "$(conda shell.bash hook)"
    conda activate base
    set -u
fi

nvcc -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_120,code=sm_120 \
    coupled_elastic_chains.cu \
    -DHARDCORE ${EXTRA_DEFS:-} \
    -o coupled_chains_sim -lcurand

echo "Built ./coupled_chains_sim for sm_86 + sm_120"
