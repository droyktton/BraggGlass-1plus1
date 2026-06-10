# Coupled Elastic Chains — GPU Langevin Dynamics

A CUDA implementation of Langevin dynamics for a 2D array of elastically coupled
chains subject to quenched disorder and thermal noise. The model is relevant to the
statistical mechanics of elastic manifolds in random media, including flux-line
lattices, charge-density waves, and domain walls in disordered magnets.

## Physical Model

The system is an `Nx × Ny` grid of scalar displacements `u(x, y)`, where `x` labels
the chain index and `y` the position along each chain. Each degree of freedom evolves
according to the overdamped Langevin equation:

```
∂u/∂t = cx·∇²_x u + cy·∇²_y u + f_pin(x, y, u) + η(t)
```

- **Elastic term** — discrete Laplacian with coupling constants `cx` (across chains)
  and `cy` (along chains), with periodic boundary conditions in both directions.
- **Pinning force** `f_pin` — quenched (time-independent) random disorder, available
  in two modes selected at compile time (see below).
- **Thermal noise** `η(t)` — Gaussian white noise with variance `2 kBT / dt`,
  generated on-device via cuRAND (Philox 4×32-10 generator).

### Disorder modes

| Compile flag | Disorder type | Description |
|---|---|---|
| *(default)* | Piecewise-constant | Force is constant within blocks of size `rf` along the displacement axis; transitions between blocks are generated on-the-fly with Philox RNG, giving a finite correlation length `rf`. |
| `-DLARKIN` | Larkin random force | Fully uncorrelated quenched random force at every lattice site, corresponding to the Larkin–Ovchinnikov model. |

### Exclusion (optional)

Compiling with `-DHARDCORE` enforces a minimum spacing of 0.9 between neighbouring
chains along `x`, clipping proposed moves that would violate this constraint. This
implements a soft hardcore repulsion relevant for non-crossing manifolds.

## Requirements

- CUDA Toolkit ≥ 11.0 (tested with 12.x)
- A CUDA-capable GPU (compute capability ≥ 6.0 recommended)
- C++14 or later

No external libraries beyond the CUDA Toolkit are required.

## Building

```bash
# Default build (piecewise-constant pinning, no exclusion)
nvcc -O3 -std=c++14 -o simulate coupled_elastic_chains.cu -lcurand

# Larkin random-force disorder
nvcc -O3 -std=c++14 -DLARKIN -o simulate_larkin coupled_elastic_chains.cu -lcurand

# Piecewise pinning + hardcore exclusion
nvcc -O3 -std=c++14 -DHARDCORE -o simulate_hc coupled_elastic_chains.cu -lcurand

# Larkin + hardcore
nvcc -O3 -std=c++14 -DLARKIN -DHARDCORE -o simulate_larkin_hc coupled_elastic_chains.cu -lcurand
```

## Usage

```bash
# Run with default parameters, random seed 42
./simulate 42

# Run with parameters loaded from a file; seed is still taken from the CLI
./simulate 42 --params params.ini

# Quick parameter sweep over seeds
for seed in 1 2 3 4 5; do
    ./simulate $seed --params params.ini
done
```

The seed on the command line always takes precedence over any `seed` entry in the
parameter file, making it easy to run independent realisations of the same physical
setup.

## Parameter file

All physical and numerical parameters can be set in a plain-text `key = value` file.
Lines beginning with `#` and blank lines are ignored. Any key omitted from the file
retains its compiled-in default. An unrecognised key is treated as an error.

```ini
# params.ini — example parameter file

Nx   = 32      # number of chains
Ny   = 512     # sites per chain

cx   = 1.0     # elastic coupling across chains
cy   = 1.0     # elastic coupling along chains

V0   = 0.1     # pinning strength
rf   = 10.0    # disorder correlation length (piecewise pinning only)

dt   = 0.01    # Langevin time step
kBT  = 0.5     # temperature

seed = 42      # overridden by the CLI argument
```

See `params.ini` in this repository for a ready-to-use template.

### Default values

| Parameter | Default | Meaning |
|---|---|---|
| `Nx` | 32 | Number of chains |
| `Ny` | 512 | Sites per chain |
| `cx` | 1.0 | Inter-chain elastic constant |
| `cy` | 1.0 | Intra-chain elastic constant |
| `V0` | 0.1 | Pinning amplitude |
| `rf` | 10.0 | Disorder correlation length |
| `dt` | 0.01 | Time step |
| `kBT` | 0.5 | Temperature |
| `seed` | 42 | RNG seed (set via CLI) |

## Output files

All output files are written to the working directory after the simulation completes.

| File | Contents |
|---|---|
| `simulation_parameters.txt` | Record of all parameters used in the run |
| `bragg_glass_thermal_output.dat` | Displacement correlator `B(r) = ⟨[u(r) − u(0)]²⟩`, computed separately along (`B_y`) and across (`B_x`) chains |
| `displacement_spectra.dat` | Power spectrum of the displacement field `S_u(q_y)`, averaged over all chains |
| `structure_factor_output.dat` | Density structure factor `S(q_y)` (computed only if uncommented in `main`) |

All `.dat` files are whitespace-delimited ASCII with a `#`-prefixed header line,
readable directly by NumPy, gnuplot, or any standard analysis tool.

## Code structure

```
coupled_elastic_chains.cu
├── Device helpers          get_piecewise_pinning_force(), quenched_random_force()
├── CUDA kernels            init_rand_kernel(), update_displacements_thermal_kernel()
├── SimParams struct        parameter storage + fromFile() factory method
├── CoupledElasticChains    class owning all GPU state; exposes step() / copyToHost()
├── Analysis functions      compute_and_save_correlation/spectra/structure_factor()
└── main()                  CLI parsing, simulation loop, output
```

The `CoupledElasticChains` class manages GPU memory via RAII: all device buffers are
allocated in the constructor and freed in the destructor, so there are no leaks even
if an exception is thrown mid-run.
