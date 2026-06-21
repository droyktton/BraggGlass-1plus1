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
- **Advanced Diagnostics:** Exports Fourier-space displacement spectra $S_u(q_y)$, density structure factors $S(q_y)$, and real-space displacement correlation functions $B(r) = \langle [u(r) - u(0)]^2 \rangle$.

## Installation & Compilation

### Prerequisites
* NVIDIA GPU (Compute Capability 6.0+)
* CUDA Toolkit (v11.0 or newer recommended)
* Host C++ compiler supporting C++14 or higher (e.g., `g++`)

### Compilation Commands

To compile the standard **Periodic Pinning / Bragg Glass** simulation layout:
```bash
nvcc -O3 -arch=sm_70 coupled_elastic_chains.cu -o coupled_chains_sim -lthrust
