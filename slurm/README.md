# Running BraggGlass-1plus1 on this Cluster

This is a SLURM (23.11, OpenHPC/lmod) cluster. GPUs live in the `gpu`
partition only — the general CPU/KNL partitions have no GRES:

| Node | GPUs | Partition |
|---|---|---|
| `node210` | 4x RTX 3080 (`sm_86`) | `gpu` |
| `node211` | 3x RTX 5070 Ti (`sm_120`, Blackwell) | `gpu` |
| `node212` | 2x RTX 3080 Ti (`sm_86`) | `gpu` |
| `node213` | 1x A10 (`sm_86`) | `gpu` |

10 GPUs total. The partition has no default time limit — always pass
`--time` explicitly (the provided scripts default to 24h; lower it if your
run is shorter, since `MaxTime=UNLIMITED` means SLURM won't do it for you).

## Environment

There is no site-wide NVHPC SDK and the only CUDA **module** is
`cuda/11.1.0-zj6lnzj`, which predates Blackwell (`sm_120`) support. Instead,
use the CUDA 13.0 toolkit already installed in the `base` conda environment
(`nvcc`, `libcurand`) — it can target both `sm_86` and `sm_120`, and is what
the binary in this repo was linked against (`ldd coupled_chains_sim` shows
`libcurand.so.10` resolving into `~/miniconda3/lib`). All scripts here
`conda activate base` for this reason; adjust if your setup differs.

Verify a GPU is actually reachable on whichever node you land on with:
```bash
srun --partition=gpu --gres=gpu:1 --pty nvidia-smi
```
The provided `.slurm` scripts also print `nvidia-smi -L` at the top of every
job log for the same reason.

## Build

Run once from the login node (no GPU needed at compile time — the target
architectures are given explicitly so it builds a fat binary covering all
four node types):
```bash
./slurm/build.sh
# or, e.g. for the Larkin regime with a 5-replica ladder:
EXTRA_DEFS="-DLARKIN -DNREPLICAS=5" ./slurm/build.sh
```

## Submit a Single Run

```bash
sbatch slurm/run_single.slurm <seedD> <seedT> [params_file]
# pin a specific GPU type if you need reproducible performance:
sbatch --gres=gpu:rtx3080:1 slurm/run_single.slurm 1234 1234 params.ini
```

## Submit a Disorder-Seed Scan

```bash
sbatch --array=0-9 slurm/run_array.slurm [seedT] [params_file]
# throttle concurrency (e.g. leave GPUs for other users):
sbatch --array=0-9%4 slurm/run_array.slurm
```
Each array task writes into its own `results/seed_<seedD>/` directory (task
`i` uses disorder seed `1234+i`). Aggregate afterwards, e.g.:
```bash
gnuplot -e "..." ver.gnu   # after adjusting ver.gnu's glob to results/seed_*/*
```

**Do not** run many seeds concurrently from a shared directory — every run
writes filenames like `correlation_replica_<T>.dat` with no seed in the
name, so parallel tasks in the same directory will clobber each other. The
array script's per-task `results/seed_<N>/` directory exists specifically to
avoid this; `run.sh`'s local sequential loop avoids it by never running two
seeds at once.

## Monitoring & Cleanup

```bash
squeue -u $USER              # your queued/running jobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after completion
scancel <jobid>               # cancel a job
scancel -u $USER --name=bragg-glass-scan   # cancel a whole array
```
Logs land in `slurm/logs/`.

## Storage Note

`$HOME` is NFS-mounted (`nas-0-0:/export/data2/...`) and shared across all
cluster users; at last check it was already ~89% full. Each replica/seed
writes a handful of small `.dat` files, but a large seed scan (`--array`
with many tasks × `-DNREPLICAS=N`) multiplies that quickly — keep an eye on
`df -h $HOME` and clean up `results/` between scans you don't need to keep.
