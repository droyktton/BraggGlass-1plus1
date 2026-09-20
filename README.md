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

### Geometry: what `x`, `y`, and `u` actually mean

`u(x,y)` is a **transverse** displacement, in the same direction/units as
`x` itself: chain `x` sits at nominal (vortex-lattice) position `x` and its
actual transverse position is $x+u(x,y)$ at height `y`. Two pieces of the
code fix this unambiguously:
* The `-DHARDCORE` constraint compares `u(x,y)` against `u(x{-}1,y)` and
  `u(x{+}1,y)` — its **x-neighbors at the same `y`** — to keep chains from
  crossing, which only makes sense if `u` and `x` share units.
* The pinning force depends on `(x + u)/rf` (`coupled_elastic_chains.cu`),
  i.e. the chain feels a random potential as a function of its *actual
  transverse position*, not an abstract combination.

So `x` indexes the $N_x$ chains, laid out on a 1D lattice of unit-spaced
transverse positions (this is the vortex/flux-line lattice); `y` is just
the internal coordinate along each line's length. `c_x` is the lattice's
shear modulus (resists neighboring chains from moving apart transversely);
`c_y` is the line-tension modulus (resists a single chain from bending
along its own length) — standard flux-line-lattice elasticity (Blatter et
al., *Rev. Mod. Phys.* 66, 1125). This matters for which diagnostics answer
which question:
* **Along-chain diagnostics** ($S_u(q_y)$, $B_y(r)$, and the `_replica_`
  structure-factor files without `transverse_` in the name) measure how
  much a *single* line wanders along its own length — the "random
  manifold" roughness problem.
* **Transverse diagnostics** ($B_x(r)$, and the `transverse_*` files)
  measure whether the $N_x$ chains stay on a regular lattice — the actual
  **Bragg glass** translational-order question. These are the ones that
  answer "is this configuration still a crystal, or has it molten."

See [Output Files](#output-files) for exact formulas for both sets.

## Key Features

- **Massive Parallel Execution:** Fully written in CUDA C++ utilizing a 2D thread block structure (`dim3 block(16, 16)`) to map spatial grids cleanly to physical GPU warps.
- **Counter-Based Quenched Random Fields:** Uses the stateless `Philox4_32_10_t` pseudo-random number generator from `cuRAND`. Pinning landscapes are computed deterministically on-the-fly based on coordinate/displacement seeds, removing the need to allocate large random grids in device memory.
- **On-GPU Energy Reductions:** Calculates global configurational energies using a custom layout kernel combined with fast `thrust::reduce` parallel reduction arrays on the device.
- **Advanced Diagnostics:** Exports both along-chain (single-line roughness: $S_u(q_y)$, $B_y(r)$) and transverse (vortex-lattice translational order: $B_x(r)$, and the exact density structure factor near the Bragg peak $S_\rho(Q{+}q_x)$, computed directly from the raw configuration with no harmonic/Gaussian approximation) diagnostics; the $Q{=}0$ branch of $S_\rho$ for either axis is also available as an exact derived quantity via `plotting/` — see [Geometry](#geometry-what-x-y-and-u-actually-mean) for which is which and [Output Files](#output-files) for exact definitions.

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
copied to the host and reduced to diagnostics along **both** axes (see
[Geometry](#geometry-what-x-y-and-u-actually-mean) for why they measure
different physics). Per replica, files are named by that replica's *final*
temperature `T`, after any parallel-tempering swaps.

#### Along-chain (single-line roughness along `y`)

* **`displacement_spectra_replica_<T>.dat`** — chain-averaged power spectrum
  of the displacement field along `y` (`compute_and_save_displacement_spectra`).
  For each chain `x`: $\hat u_x(q_y) = \sum_{y=0}^{N_y-1} u(x,y)\,e^{-iq_y y}$
  at $q_y = 2\pi k/N_y$, averaged over chains:
  $$S_u(q_y) = \frac{1}{N_x}\sum_{x=0}^{N_x-1}\frac{|\hat u_x(q_y)|^2}{N_y^2}$$
  Columns: `k`, `qy`, `S_u(qy)`, for `k = 1 .. Ny/2` (`k=0` skipped).

* **`structure_factor_replica_<T>.dat`** — phase coherence of a single
  chain's own wandering along its length, evaluated *exactly*
  (`compute_and_save_structure_factor`): $\sum_y e^{i(q_y y + 2\pi u(x,y))}$,
  $|\cdot|^2$, averaged over chains. **This is not the vortex-lattice Bragg
  peak** — see the transverse version below for that. Columns: `k`, `qy`,
  `S(qy)`, `k = 0 .. Ny/2`.

#### Transverse (vortex-lattice translational order, across chains along `x`)

These answer the actual "Bragg glass" question: do the $N_x$ chains stay on
a regular lattice, or has it molten?

* **`transverse_spectrum_replica_<T>.dat`** — spectrum of `u` along `x`,
  averaged over `y` (`compute_and_save_transverse_spectrum`), exactly
  mirroring `displacement_spectra` with `x`/`Nx` swapped for `y`/`Ny`:
  $$S_u^{(x)}(q_x) = \frac{1}{N_y}\sum_{y=0}^{N_y-1}\frac{|\hat u_y(q_x)|^2}{N_x^2}, \qquad \hat u_y(q_x) = \sum_{x=0}^{N_x-1} u(x,y)\,e^{-iq_x x}$$
  Columns: `k`, `qx`, `S_u(qx)`, for `k = 1 .. Nx/2`.

* **`transverse_structure_factor_replica_<T>.dat`** — the actual vortex-lattice
  density (Bragg-peak) structure factor, evaluated *exactly* from the raw
  configuration across the **whole first Brillouin zone**
  $q\in[0,2\pi]$ in one consistent calculation — no small-$q$ or
  near-$2\pi$ approximation, no harmonic/Gaussian approximation, and (an
  earlier version of this file got this wrong) no dropped cross-term.
  `compute_and_save_transverse_structure_factor`: for each `y`, sum
  $e^{iq(x+u(x,y))}$ over `x` — the *literal* transverse position
  $x+u(x,y)$ of chain `x` — take $|\cdot|^2$, average over `y`:
  $$S_\rho(q) = \frac{1}{N_y}\sum_{y=0}^{N_y-1}\frac{1}{N_x}\Big|\sum_{x=0}^{N_x-1} e^{iq(x+u(x,y))}\Big|^2$$
  Columns: `k`, `qx`, `S(qx)`, for `k = 0 .. Nx` inclusive (i.e. `qx` runs
  `0, 2π/Nx, 2·2π/Nx, ..., 2π`) — note the resolution here is set by `Nx`,
  often the *smaller* grid dimension.

  **$q=0$ and $q=2\pi$ are genuinely different points, not the same one
  revisited.** $S_\rho(q{=}0)$ is *always* exactly $N_x$ regardless of the
  configuration — pure particle-number conservation, $\sum_x e^{i\cdot 0}=N_x$
  — while $S_\rho(q{=}2\pi)$ is the informative quantity that decays from
  $N_x$ (perfect lattice order) as disorder/thermal fluctuations melt the
  lattice. It's tempting to assume $S_\rho$ is $2\pi$-periodic (a discrete
  Fourier transform on integer `x` alone would be), but the `u`-dependent
  phase breaks that: $\hat\rho(q{+}2\pi,y) = \hat\rho(q,y)\cdot e^{-i2\pi u(x,y)} \ne \hat\rho(q,y)$
  whenever `u` isn't an integer. An earlier version of this file exploited
  a false version of this periodicity to stitch two different
  approximations together at $q=\pi$ and produced a spurious discontinuity
  there — the fix was to compute the one exact quantity across the whole
  zone directly, as above, rather than switching formulas partway through.
  A real jump between the $q=0$ and $q=2\pi$ values is expected physics
  (trivial peak vs. actual Bragg peak); a jump *elsewhere*, or a
  discontinuity in an otherwise-smooth region, would indicate a bug.

  $S_\rho$ and $S_u^{(x)}$/$B_x(r)$ are not linearly related near the
  $q=2\pi$ Bragg peak (density is a nonlinear/periodic function of `u`). A
  cheap estimate instead of this exact export: the harmonic/Debye–Waller
  approximation $S_\rho(q) \approx \sum_r e^{-iqr}\,e^{-2\pi^2 B_x(r)}$ from
  the already-exported $B_x(r)$ — valid only while `u` stays
  Gaussian-distributed (breaks down with proliferating dislocations, strong
  pinning, or `-DHARDCORE`).

  Near $q=0$ (excluding the trivial $q{=}0$ point itself), density
  fluctuations reduce to the compressional term
  $\delta\rho(x,y) = u(x{+}1,y)-u(x,y)$, exactly linear in `u`. Its
  structure factor is an exact rescaling of $S_u^{(x)}$ — but matching
  *this* file's normalization convention (dividing by $N_x^1$, not
  $S_u^{(x)}$'s $N_x^2$) needs an extra factor of $N_x$, from the discrete
  identity $\sum_x e^{iqx}=0$ for $q\ne 0 \bmod 2\pi$:
  $$S_\rho^{(Q=0)}(q) = 4N_x\sin^2(q/2)\,S_u^{(x)}(q) \;\xrightarrow{q\to 0}\; N_x\,q^2 S_u^{(x)}(q)$$
  `plotting/plot_transverse_spectrum_q0.py` computes this directly from
  `transverse_spectrum_replica_<T>.dat`, and
  `plotting/plot_transverse_structure_factor_full.py` overlays it on the
  exact curve near $q\to0$ as a consistency check (it should track the
  exact curve there and depart from it as $q$ grows — both are shown, not
  stitched). It's smooth and self-averaging (unlike the exact curve near
  $q=2\pi$), being linear rather than exponentially sensitive in `u`.
  The along-chain analogues (`plot_structure_factor_q0.py` /
  `plot_structure_factor.py` / `plot_structure_factor_full.py`) exist too,
  for the (different!) single-line-roughness question — that one's $Q=0$
  branch needs the analogous $N_y$ factor,
  $S_\rho^{(Q=0)}(q) = 4N_y\sin^2(q/2)\,S_u(q)$, matching `ver.gnu`'s
  original `q^2*S_u(q)*Ny` convention on the Displacement spectrum panel.

#### Both axes together

* **`correlation_replica_<T>.dat`** — real-space displacement correlations,
  averaged over the whole grid, along a chain and across chains:
  $$B_y(r) = \frac{1}{N_xN_y}\sum_{x,y}\big[u(x,y)-u(x,y{+}r \bmod N_y)\big]^2,\qquad
    B_x(r) = \frac{1}{N_xN_y}\sum_{x,y}\big[u(x,y)-u(x{+}r \bmod N_x,y)\big]^2$$
  Columns: `r`, `B_y(r)`, `B_x(r)`, for `r = 0 .. max(Nx,Ny)/2 - 1`; a column
  reads `nan` past its own axis' range (e.g. `B_x` once `r >= Nx/2`). $B_x(r)$
  is the transverse (vortex-lattice) correlation function feeding the
  Debye–Waller estimate above; $B_y(r)$ is the along-chain one.

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

`ver.gnu` gives a quick three-panel look (along-chain `$S_u(q_y)$`, the
full-zone **vortex-lattice** `$S_\rho(q)$`, `$B(r)$`) against the
thermal/Larkin scaling predictions, aggregating whatever
`seed_*_*_replica_*.dat` files sit in the current directory:
```bash
gnuplot ver.gnu
```
Edit the `Nx`/`Ny` variables at the top of `ver.gnu` to match the grid used
for the run before plotting.

For anything beyond that quick look — saved PNGs, correctly grouping by
temperature when `-DNREPLICAS>1` instead of lumping every replica together,
isolating one branch of $S_\rho$ on its own axes, or the along-chain
(rather than transverse) variants — use the scripts in
[`plotting/`](plotting/) (same conda `base` env as the rest of the
toolchain; needs `numpy` + `matplotlib`, already present there):
```bash
# All three panels (S_u along-chain, transverse/vortex-lattice S_rho full zone, B(r)):
python plotting/plot_all.py --nx 32 --ny 64 --dir . --out plots/diagnostics.png

# Along-chain (single-line roughness, y-axis):
python plotting/plot_spectrum.py --ny 64 --dir .                          # S_u(qy) only
python plotting/plot_structure_factor_full.py --ny 64 --dir .             # S_rho(q), full zone (Q=0 approx -> Bragg peak)
python plotting/plot_structure_factor_q0.py --ny 64 --dir .               # S_rho near Q=0 only (approx, from S_u)
python plotting/plot_structure_factor.py --ny 64 --dir .                  # S_rho near Bragg peak only (exact) + DW estimate

# Transverse (vortex-lattice translational order, x-axis -- the actual Bragg-glass question):
python plotting/plot_transverse_structure_factor_full.py --nx 32 --dir .  # S_rho(q), full zone (exact) + both approximations
python plotting/plot_transverse_spectrum_q0.py --nx 32 --dir .            # S_rho near Q=0 only (approx, from S_u^(x))
python plotting/plot_transverse_structure_factor.py --nx 32 --dir .       # S_rho(q), full zone (exact) + DW estimate

python plotting/plot_correlation.py --dir .                               # B(r), both axes
```
`--dir` is searched recursively, so it works equally on a flat `run.sh`
seed scan, a single run's output, or a `slurm/run_array.slurm` `results/`
tree. Files are grouped and averaged **by temperature** (not lumped
together across a `-DNREPLICAS>1` ladder), and each script's `--out` accepts
any extension matplotlib supports (`.png`, `.pdf`, `.svg`, ...).

See [`plotting/examples/`](plotting/examples/) for sample output from a real
cluster run (5-replica ladder, 6 disorder seeds).

## Running on a Cluster (SLURM)

See [`slurm/README.md`](slurm/README.md) and the build/submission scripts in
[`slurm/`](slurm/) for cluster-specific setup, GPU node layout, and job
templates.
