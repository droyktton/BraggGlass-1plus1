# BraggGlass-1plus1

A high-performance CUDA C++ simulation framework designed to study the equilibrium configurations, phase transitions, and structural properties of elastically coupled continuous chains ($1+1$ dimensional objects) in a rugged, disordered pinning landscape. 

This repository is optimized to explore the physics of the **Bragg Glass phase**, random manifolds, and the crossover into the **Larkin regime** using massive GPU parallelism.

## Physical Background

The code simulates the overdamped Langevin dynamics of a discrete displacement field $u(x, y)$ representing $N_x$ elastically coupled elastic chains, each of length $N_y$. The equation of motion for each point $(x, y)$ is governed by:

$$
\frac{\partial u(x,y)}{\partial t} = f_{\text{elastic}} + f_{\text{pinning}}(x, y, u) + \eta(x, y, t)
$$

Where:
* **Elastic Force ($f_{\text{elastic}}$):** Incorporates anisotropic elastic constants ($c_x$ coupling between adjacent chains, and $c_y$ tension along an individual chain):
  $$f_{\text{elastic}} = c_x [u(x-1,y) + u(x+1,y) - 2u(x,y)] + c_y [u(x,y-1) + u(x,y+1) - 2u(x,y)]$$
* **Pinning Force ($f_{\text{pinning}}$):** A quenched disordered landscape. Depending on compilation flags, it simulates either a periodic/random manifold landscape or a pure Larkin force.
* **Thermal Noise ($\eta$):** Gaussian white noise satisfying $\langle \eta(x,y,t)\eta(x',y',t') \rangle = 2 k_B T \delta_{x,x'} \delta_{y,y'} \delta(t-t')$.

To navigate the highly non-convex, rugged energy landscapes typical of glassy systems, this framework implements **Replica Exchange Langevin Dynamics (Parallel Tempering)**. Multiple copies (replicas) of the system run concurrently at a ladder of distinct temperatures, swapping configurations periodically according to a Metropolis-Hastings criterion to bypass high energy barriers.

## Key Features

- **Massive Parallel Execution:** Fully written in CUDA C++ utilizing a 2D thread block structure (`dim3 block(16, 16)`) to map spatial grids cleanly to physical GPU warps.
- **Counter-Based Quenched Random Fields:** Uses the stateless `Philox4_32_10_t` pseudo-random number generator from `cuRAND`. Pinning landscapes are computed deterministically on-the-fly based on coordinate/displacement seeds, removing the need to allocate large random grids in device memory.
- **On-GPU Energy Reductions:** Calculates global configurational energies using a custom layout kernel combined with fast `thrust::reduce` parallel reduction arrays on the device.
- **Advanced Diagnostics:** Exports the Fourier-space displacement spectrum $S_u(q_y)$, the real-space displacement correlation function $B(r) = \langle [u(r) - u(0)]^2 \rangle$, and the exact density (Bragg-peak) structure factor $S_\rho(q_y)$ computed directly from the raw configuration (no harmonic/Gaussian approximation) — see [Output Files](#output-files) for exact definitions.

## Installation & Compilation

### Prerequisites
* NVIDIA GPU (Compute Capability 6.0+)
* CUDA Toolkit (v11.0 or newer; v12.8+ required to target Blackwell/`sm_120` GPUs)
* Host C++ compiler supporting C++14 or higher (e.g., `g++`)

The code depends only on `curand_kernel.h` (header-only device API) and Thrust
(`thrust::reduce`), both shipped with any CUDA Toolkit — no separate NVHPC SDK
or standalone Thrust install is required.

### Compile-Time Flags

| Flag | Effect |
|---|---|
| *(none)* | Piecewise-constant "periodic manifold" pinning landscape |
| `-DLARKIN` | Replace it with an uncorrelated (Larkin) random-force landscape |
| `-DHARDCORE` | Enforce a hardcore minimum spacing (0.9) between neighboring chains |
| `-DNREPLICAS=N` | Run `N` parallel-tempering replicas instead of a single system (default 1) |

### Compilation Commands

Basic build (single GPU architecture, e.g. a workstation with an Ampere card):
```bash
nvcc -O3 -arch=native coupled_elastic_chains.cu -DHARDCORE -o coupled_chains_sim -lcurand
```

Bragg Glass with a 5-replica temperature ladder:
```bash
nvcc -O3 -arch=native coupled_elastic_chains.cu -DHARDCORE -DNREPLICAS=5 -o coupled_chains_sim -lcurand
```

Larkin regime instead of the periodic manifold:
```bash
nvcc -O3 -arch=native coupled_elastic_chains.cu -DHARDCORE -DLARKIN -o coupled_chains_sim -lcurand
```

`-arch=native` requires a GPU to be visible at compile time. To build a single
binary that runs across a mix of GPU generations (as needed on a shared
cluster), list the target architectures explicitly instead — see
[`slurm/build.sh`](slurm/build.sh).

## Usage

```bash
./coupled_chains_sim <seedD> <seedT> [--params <file.ini>]
```

* `seedD` — seed for the quenched disorder (pinning landscape). Keep this
  fixed across replicas of the *same* parallel-tempering ladder — they must
  share one disorder realization for replica exchange to be valid — and vary
  it across independent disorder realizations (e.g. the seed scans in
  `run.sh`).
* `seedT` — seed for the thermal noise / initial random phase field. Safe to
  vary independently of `seedD`.
* `--params <file.ini>` — optional key/value file overriding the compiled-in
  defaults in `SimParams` (see `params.ini` for the format and comments).
  Unknown keys or malformed lines abort with an error.

### Parameters File

| Key | Meaning | Default |
|---|---|---|
| `Nx` | Number of coupled chains | 32 |
| `Ny` | Length of each chain | 512 |
| `cx` | Elastic coupling between adjacent chains | 1.0 |
| `cy` | Elastic tension along a chain | 1.0 |
| `V0` | Pinning strength | 0.1 |
| `dt` | Langevin time step | 0.01 |
| `rf` | Disorder correlation length (piecewise-constant pinning only, ignored with `-DLARKIN`) | 10.0 |
| `kBT` | Temperature of the base (`seedD`/`seedT`) replica; with `-DNREPLICAS=N>1` this is the low end of a ladder that runs up to `kBT=1.0` | 0.5 |
| `seedD`, `seedT` | Overridden by the CLI arguments if both are given | 42 |

The number of Langevin steps (`n_steps = 1,000,000`) and the temperature
ladder's upper bound (`T_max = 1.0`) are currently compiled in (`main()` in
`coupled_elastic_chains.cu`), not configurable via `params.ini`.

### Output Files

At the end of the run, each replica's final displacement field `u(x,y)` is
copied to the host and reduced to two diagnostics. Per replica (files named
by that replica's *final* temperature `T`, after any parallel-tempering
swaps — see `compute_and_save_displacement_spectra` / `_correlation` in
`coupled_elastic_chains.cu`):

* **`displacement_spectra_replica_<T>.dat`** — chain-averaged power spectrum
  of the displacement field along `y`. For each chain `x` the code takes the
  discrete Fourier transform $\hat u_x(q_y) = \sum_{y=0}^{N_y-1} u(x,y)\,e^{-iq_y y}$
  at $q_y = 2\pi k/N_y$, then averages over chains:
  $$S_u(q_y) = \frac{1}{N_x}\sum_{x=0}^{N_x-1}\frac{|\hat u_x(q_y)|^2}{N_y^2}$$
  Columns: `k`, `qy`, `S_u(qy)`, for `k = 1 .. Ny/2` (the `k=0` / zero-mode
  row is skipped).

* **`correlation_replica_<T>.dat`** — real-space displacement correlations,
  averaged over the whole grid, separately along a chain and across chains:
  $$B_y(r) = \frac{1}{N_xN_y}\sum_{x,y}\big[u(x,y)-u(x,y{+}r \bmod N_y)\big]^2,\qquad
    B_x(r) = \frac{1}{N_xN_y}\sum_{x,y}\big[u(x,y)-u(x{+}r \bmod N_x,y)\big]^2$$
  Columns: `r`, `B_y(r)`, `B_x(r)`, for `r = 0 .. max(Nx,Ny)/2 - 1`; a column
  reads `nan` past its own axis' range (e.g. `B_x` once `r >= Nx/2`).

* **`structure_factor_replica_<T>.dat`** — the density (Bragg-peak)
  structure factor along `y`, evaluated *exactly* from the raw
  configuration (no harmonic/Debye–Waller approximation): for each chain
  `x`, sum the local phase factor $e^{i(q_y y + 2\pi u(x,y))}$ over `y`
  (the $2\pi u$ term is the density modulation at the fundamental Bragg
  wavevector $Q=2\pi$, since `u` is in units of the lattice periodicity),
  take $|\cdot|^2$, then average over chains:
  $$S_\rho(q_y) = \frac{1}{N_x}\sum_{x=0}^{N_x-1}\frac{1}{N_y}\Big|\sum_{y=0}^{N_y-1} e^{i(q_y y + 2\pi u(x,y))}\Big|^2$$
  Columns: `k`, `qy`, `S(qy)`, for `k = 0 .. Ny/2` (unlike the displacement
  spectrum, the `k=0` row *is* included here — it's the total Bragg
  intensity, not a divergent zero mode). This is the quantity that decays
  from $N_y$ (perfect order) as disorder washes out Bragg peaks.

  $S_\rho$ and $S_u$/$B(r)$ carry different information and are not
  linearly related — density is a nonlinear (periodic) function of $u$. If
  you only need a cheap estimate rather than this exact export, the
  harmonic/Debye–Waller approximation gives $S_\rho(q) \approx \sum_r e^{-iqr}\,e^{-2\pi^2 B(r)}$
  from the already-exported $B(r)$, but it only holds while $u$ stays
  Gaussian-distributed (breaks down with proliferating dislocations, strong
  pinning, or `-DHARDCORE`) — this file avoids that assumption entirely.

Once per run:
* `simulation_parameters.txt` — echoes the parameters actually used.

Not exported to a file: `compute_configurational_energy()` evaluates the
same elastic + pinning energy per site (forward-difference bonds, so each
bond is counted once) and reduces it on the GPU via `thrust::reduce` — this
is used only internally, to decide parallel-tempering swaps every 100 steps.

### Running a Seed Scan Locally

```bash
./run.sh <n_seeds> [seedT] [params_file]
```

Runs disorder seeds `1234 .. 1234+n_seeds-1` sequentially against one
`seedT`, renaming each run's output with a `seed_<seedD>_` prefix so
successive runs don't overwrite each other. This is fine for a workstation;
for a cluster, parallelize seeds across jobs instead (see below) rather than
looping — but note that concurrent runs in the *same* directory will still
collide on filenames, so each parallel job needs its own working directory
(handled by `slurm/run_array.slurm`).

### Visualization

`ver.gnu` gives a quick three-panel look (`$S_u$`, `$S_\rho$`, `$B(r)$`)
against the thermal/Larkin scaling predictions, aggregating whatever
`seed_*_*_replica_*.dat` files sit in the current directory:
```bash
gnuplot ver.gnu
```
Edit the `Nx`/`Ny` variables at the top of `ver.gnu` to match the grid used
for the run before plotting.

For anything beyond that quick look — saved PNGs, correctly grouping by
temperature when `-DNREPLICAS>1` instead of lumping every replica together,
or comparing the exact $S_\rho$ against the Debye–Waller estimate from
$B(r)$ — use the scripts in [`plotting/`](plotting/) (same conda `base` env
as the rest of the toolchain; needs `numpy` + `matplotlib`, already present
there):
```bash
python plotting/plot_all.py --ny 64 --dir . --out plots/diagnostics.png   # all three panels
python plotting/plot_spectrum.py --ny 64 --dir .                          # S_u(qy) only
python plotting/plot_structure_factor.py --ny 64 --dir .                  # S_rho(qy) + DW estimate
python plotting/plot_correlation.py --dir .                               # B(r) only
```
`--dir` is searched recursively, so it works equally on a flat `run.sh`
seed scan, a single run's output, or a `slurm/run_array.slurm` `results/`
tree. Files are grouped and averaged **by temperature** (not lumped
together across a `-DNREPLICAS>1` ladder), and each script's `--out` accepts
any extension matplotlib supports (`.png`, `.pdf`, `.svg`, ...).

## Running on a Cluster (SLURM)

See [`slurm/README.md`](slurm/README.md) and the build/submission scripts in
[`slurm/`](slurm/) for cluster-specific setup, GPU node layout, and job
templates.
