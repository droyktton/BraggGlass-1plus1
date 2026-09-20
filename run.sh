#!/usr/bin/env bash
# Local batch runner: scans a range of disorder seeds through one thermal-noise seed.
# For cluster (SLURM) execution use the scripts in slurm/ instead — running many
# seeds in parallel from a shared working directory will clobber each other's
# output files, since every run writes the same filenames (see slurm/README.md).
set -euo pipefail
cd "$(dirname "$0")"

# Build (uncomment to (re)compile). Targets both Ampere (RTX 3080/3080Ti/A10)
# and Blackwell (RTX 5070Ti) GPUs in one binary — see slurm/build.sh for the
# cluster build used on the SLURM login node.
# nvcc -O3 \
#   -gencode arch=compute_86,code=sm_86 \
#   -gencode arch=compute_120,code=sm_120 \
#   coupled_elastic_chains.cu -DHARDCORE -o coupled_chains_sim -lcurand

n_seeds=${1:?"usage: ./run.sh <n_seeds> [seedT] [params_file]"}
seedT=${2:-1234}
params=${3:-params.ini}

last=$((1234 + n_seeds))
for ((seedD = 1234; seedD < last; seedD++)); do
    ./coupled_chains_sim "$seedD" "$seedT" --params "$params"
    for f in displacement_spectra_replica_*.dat correlation_replica_*.dat; do
        [ -e "$f" ] && mv "$f" "seed_${seedD}_${f}"
    done
done
