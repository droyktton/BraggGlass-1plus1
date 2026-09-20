#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <complex>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(err) \
    do { \
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ─────────────────────────── DEVICE HELPERS ──────────────────────────────────

#define PHILOX_OFFSET 10000000

__device__ double get_piecewise_pinning_force(unsigned int idx, unsigned int u_block,
                                              double V0, unsigned int seedD) {
    curandStatePhilox4_32_10_t state;
    curand_init(seedD, idx, u_block, &state);
    return -V0 + 2.0 * V0 * curand_uniform_double(&state);
}

__device__ double quenched_random_force(unsigned int x, unsigned int y,
                                        double V0, unsigned int seedD) {
    curandStatePhilox4_32_10_t state;
    curand_init(seedD, y, x, &state);
    return -V0 + 2.0 * V0 * curand_uniform_double(&state);
}

// ─────────────────────────────── KERNELS ─────────────────────────────────────

// FIX: use Philox throughout — update_displacements_thermal_kernel reads
// rand_states as curandStatePhilox4_32_10_t*, so init must use the same type.
// Mixing XORWOW (default curandState) and Philox pointers causes silent memory
// corruption in the kernel → "unknown error" on cudaDeviceSynchronize.
__global__ void init_rand_kernel(curandStatePhilox4_32_10_t* rand_states, int Nx, int Ny,
                                  unsigned long long seedT) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;
        curand_init(seedT, idx, 0, &rand_states[idx]);
    }
}

__global__ void update_displacements_thermal_kernel(
        const double* __restrict__ d_u,
        double*       __restrict__ d_u_next,
        const double* __restrict__ d_phi,
        curandStatePhilox4_32_10_t* rand_states,
        unsigned int seedD,
        int Nx, int Ny,
        double cx, double cy,
        double V0, double rf,
        double dt, double noise_scale) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;

        // Periodic boundary conditions
        int left_x  = (x - 1 + Nx) % Nx;
        int right_x = (x + 1) % Nx;
        int up_y    = (y - 1 + Ny) % Ny;
        int down_y  = (y + 1) % Ny;

        double u_curr  = d_u[idx];
        double u_left  = d_u[left_x  * Ny + y];
        double u_right = d_u[right_x * Ny + y];
        double u_up    = d_u[x * Ny + up_y];
        double u_down  = d_u[x * Ny + down_y];

        // 1. Elastic force
        double f_elastic = cx * (u_left + u_right - 2.0 * u_curr)
                         + cy * (u_up   + u_down  - 2.0 * u_curr);

        // 2. Pinning force
        int u_block_signed      = __double2int_rd((static_cast<double>(x) + u_curr) / rf);
        unsigned int u_block_ph = (unsigned int)(u_block_signed + PHILOX_OFFSET);

#ifndef LARKIN
        double f_pinning = get_piecewise_pinning_force(y, u_block_ph, V0, seedD);
#else
        double f_pinning = quenched_random_force(x, y, V0, seedD);
#endif

        // 3. Thermal noise (Gaussian via cuRAND)
        curandStatePhilox4_32_10_t local_state = rand_states[idx];
        double gaussian_noise    = curand_normal_double(&local_state);
        rand_states[idx]        = local_state;

        // 4. Langevin update
        double u_prop = u_curr + dt * (f_elastic + f_pinning) + noise_scale * gaussian_noise;

#ifdef HARDCORE
        double max_desplazamiento_izq = u_prop - u_left;
        double max_desplazamiento_der = u_right - u_prop;
        if (max_desplazamiento_izq < -0.9) u_prop = u_left  - 0.9;
        if (max_desplazamiento_der >  0.9) u_prop = u_right - 0.9;
#endif

        d_u_next[idx] = u_prop;
    }
}

__global__ void compute_configurational_energy_kernel(
        const double* __restrict__ d_u,
        unsigned int seedD,
        int Nx, int Ny,
        double cx, double cy,
        double V0, double rf,
        double* __restrict__ d_local_energies) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;

        // Forward neighbors for periodic boundary conditions to avoid double-counting bonds
        int right_x = (x + 1) % Nx;
        int down_y  = (y + 1) % Ny;

        double u_curr  = d_u[idx];
        double u_right = d_u[right_x * Ny + y];
        double u_down  = d_u[x * Ny + down_y];

        // 1. Forward Elastic Energy (1/2 * k * dx^2)
        double e_elastic = 0.5 * cx * (u_right - u_curr) * (u_right - u_curr)
                         + 0.5 * cy * (u_down  - u_curr) * (u_down  - u_curr);

        // 2. Pinning Potential Energy
        double e_pinning = 0.0;

#ifndef LARKIN
        int u_block_signed      = __double2int_rd((static_cast<double>(x) + u_curr) / rf);
        unsigned int u_block_ph = (unsigned int)(u_block_signed + PHILOX_OFFSET);
        double f_pinning = get_piecewise_pinning_force(y, u_block_ph, V0, seedD);
        e_pinning = -f_pinning * u_curr;
#else
        double f_pinning = quenched_random_force(x, y, V0, seedD);
        e_pinning = -f_pinning * u_curr;
#endif

        d_local_energies[idx] = e_elastic + e_pinning;
    }
}

// ──────────────────────── PHYSICAL SYSTEM CLASS ──────────────────────────────

struct SimParams {
    int          Nx   = 32;
    int          Ny   = 512;
    double       cx   = 1.0;
    double       cy   = 1.0;
    double       V0   = 0.1;
    double       dt   = 0.01;
    double       rf   = 10.0;
    double       kBT  = 0.5;
    unsigned int seedD = 42u;
    unsigned int seedT = 42u;

    /// Load parameters from a key = value file.
    /// Lines beginning with '#' and blank lines are ignored.
    /// Any key not present in the file retains its default value.
    /// Throws std::runtime_error on unrecognised keys or bad values.
    static SimParams fromFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open parameter file: " + path);

        SimParams p;
        std::string line;
        int line_no = 0;

        while (std::getline(f, line)) {
            ++line_no;

            // Strip comments and skip blank lines
            auto comment_pos = line.find('#');
            if (comment_pos != std::string::npos)
                line = line.substr(0, comment_pos);
            if (line.find_first_not_of(" \t\r\n") == std::string::npos)
                continue;

            std::istringstream ss(line);
            std::string key, eq;
            if (!(ss >> key >> eq) || eq != "=")
                throw std::runtime_error("Parse error on line " + std::to_string(line_no)
                                         + ": expected 'key = value'");

            try {
                if      (key == "Nx"   ) { int          v; ss >> v; p.Nx    = v; }
                else if (key == "Ny"   ) { int          v; ss >> v; p.Ny    = v; }
                else if (key == "cx"   ) { double       v; ss >> v; p.cx    = v; }
                else if (key == "cy"   ) { double       v; ss >> v; p.cy    = v; }
                else if (key == "V0"   ) { double       v; ss >> v; p.V0    = v; }
                else if (key == "dt"   ) { double       v; ss >> v; p.dt    = v; }
                else if (key == "rf"   ) { double       v; ss >> v; p.rf    = v; }
                else if (key == "kBT"  ) { double       v; ss >> v; p.kBT   = v; }
                else if (key == "seedD") { unsigned int v; ss >> v; p.seedD  = v; }
                else if (key == "seedT") { unsigned int v; ss >> v; p.seedT  = v; }
                else
                    throw std::runtime_error("Unknown parameter '" + key
                                             + "' on line " + std::to_string(line_no));
            } catch (const std::runtime_error&) {
                throw;
            } catch (...) {
                throw std::runtime_error("Bad value for '" + key
                                         + "' on line " + std::to_string(line_no));
            }
        }
        return p;
    }
};

class CoupledElasticChains {
public:
    // ── Construction / Destruction ────────────────────────────────────────────
    explicit CoupledElasticChains(const SimParams& p)
        : params_(p),
          grid_size_(p.Nx * p.Ny),
          noise_scale_(std::sqrt(2.0 * p.kBT * p.dt)),
          current_buf_(0)
    {
        const size_t bytes      = grid_size_ * sizeof(double);
        const size_t state_bytes = grid_size_ * sizeof(curandStatePhilox4_32_10_t);

        // ── GPU buffers via cudaMalloc ────────────────────────────────────────
        // FIX: use cudaMalloc for all buffers instead of thrust::device_vector.
        //   - d_u_buf_[0/1]: double-buffer for displacements; swapped via current_buf_
        //     index — raw pointer swap removed entirely.
        //   - d_rand_states_: curandState is not trivially copyable; Thrust would
        //     attempt element-wise construction/copy on resize which is UB and
        //     causes the std::bad_alloc observed at runtime.
        CUDA_CHECK(cudaMalloc(&d_u_buf_[0],      bytes));
        CUDA_CHECK(cudaMalloc(&d_u_buf_[1],      bytes));
        CUDA_CHECK(cudaMalloc(&d_phi_,           bytes));
        CUDA_CHECK(cudaMalloc(&d_rand_states_,   state_bytes));
        CUDA_CHECK(cudaMalloc(&d_local_energies_, bytes));

        // Initialise displacement field to zero
        CUDA_CHECK(cudaMemset(d_u_buf_[0], 0, bytes));
        CUDA_CHECK(cudaMemset(d_u_buf_[1], 0, bytes));

        // Initialise random phase field on host, then copy to device
        std::vector<double> h_phi(grid_size_);
        std::mt19937 prng(params_.seedT);
        std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * M_PI);
        for (auto& v : h_phi) v = dist_phi(prng);
        CUDA_CHECK(cudaMemcpy(d_phi_, h_phi.data(), bytes, cudaMemcpyHostToDevice));

        // Set up CUDA launch config.
        // 8x8 = 64 threads/block: curandState usa ~48 bytes en registros/local memory;
        // bloques de 256 threads (16x16) pueden saturar los registros disponibles
        // por SM en GPUs con compute capability < 8.0, causando "unknown error" en
        // el kernel de inicialización de cuRAND.
        threads_ = dim3(8, 8);
        blocks_  = dim3((params_.Nx + threads_.x - 1) / threads_.x,
                        (params_.Ny + threads_.y - 1) / threads_.y);

        // Initialise cuRAND states on the GPU
        // Diagnostic: print launch config so we can catch invalid grids
        std::cout << "[DEBUG] init_rand_kernel: grid=(" << blocks_.x << "," << blocks_.y
                  << ") block=(" << threads_.x << "," << threads_.y
                  << ") Nx=" << params_.Nx << " Ny=" << params_.Ny
                  << " seedT=" << p.seedT << "\n";
        std::cout.flush();

        init_rand_kernel<<<blocks_, threads_>>>(d_rand_states_, params_.Nx, params_.Ny,
                                                (unsigned long long)p.seedT);

        // Check launch error before sync — gives a different message than sync error
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cerr << "[FATAL] init_rand_kernel launch failed: "
                      << cudaGetErrorString(launch_err) << "\n";
            exit(EXIT_FAILURE);
        }

        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::cerr << "[FATAL] init_rand_kernel execution failed: "
                      << cudaGetErrorString(sync_err) << "\n";
            // Consume sticky error so subsequent calls show their own errors
            cudaGetLastError();
            exit(EXIT_FAILURE);
        }
    }

    ~CoupledElasticChains() {
        cudaFree(d_u_buf_[0]);
        cudaFree(d_u_buf_[1]);
        cudaFree(d_phi_);
        cudaFree(d_rand_states_);
        cudaFree(d_local_energies_);
    }

    // Non-copyable, non-movable
    CoupledElasticChains(const CoupledElasticChains&)            = delete;
    CoupledElasticChains& operator=(const CoupledElasticChains&) = delete;

    // ── Simulation interface ──────────────────────────────────────────────────

    /// Advance one Langevin time step on the GPU.
    void step() {
        // FIX: instead of swapping raw pointers (which broke ownership with
        //   thrust::device_vector), we index into d_u_buf_[0/1] directly.
        double* src = d_u_buf_[current_buf_];
        double* dst = d_u_buf_[current_buf_ ^ 1];

        update_displacements_thermal_kernel<<<blocks_, threads_>>>(
            src, dst, d_phi_, d_rand_states_,
            params_.seedD, params_.Nx, params_.Ny,
            params_.cx, params_.cy,
            params_.V0, params_.rf,
            params_.dt, noise_scale_
        );
        CUDA_CHECK(cudaGetLastError());

        current_buf_ ^= 1;  // flip active buffer
    }

    /// Copy the current displacement field from device to the provided host vector.
    void copyToHost(std::vector<double>& h_u) const {
        h_u.resize(grid_size_);
        CUDA_CHECK(cudaMemcpy(h_u.data(), d_u_buf_[current_buf_],
                              grid_size_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    double compute_configurational_energy() {
        compute_configurational_energy_kernel<<<blocks_, threads_>>>(
            d_u_buf_[current_buf_],
            params_.seedD,
            params_.Nx, params_.Ny,
            params_.cx, params_.cy,
            params_.V0, params_.rf,
            d_local_energies_
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Parallel reduction on GPU via Thrust (safe: d_local_energies_ is plain double*)
        thrust::device_ptr<double> dev_ptr(d_local_energies_);
        double total_energy = thrust::reduce(dev_ptr, dev_ptr + grid_size_,
                                             0.0, thrust::plus<double>());
        return total_energy;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    const SimParams& params()     const { return params_; }
    double           noiseScale() const { return noise_scale_; }
    int              gridSize()   const { return grid_size_; }

    void set_kBT(double new_kBT) {
        params_.kBT  = new_kBT;
        noise_scale_ = std::sqrt(2.0 * params_.dt * params_.kBT);
    }

    double get_kBT() const { return params_.kBT; }

private:
    SimParams    params_;
    int          grid_size_;
    double       noise_scale_;
    int          current_buf_;   // 0 or 1 — index into d_u_buf_

    // FIX: double-buffer as a plain array of two pointers; ownership is clear
    //   and no raw-pointer swap is needed.
    double*      d_u_buf_[2]      = {nullptr, nullptr};
    double*      d_phi_           = nullptr;
    curandStatePhilox4_32_10_t* d_rand_states_   = nullptr;  // cudaMalloc — Philox, same type used in all kernels
    double*      d_local_energies_ = nullptr;

    dim3 threads_;
    dim3 blocks_;
};


// ────────────────────────── ANALYSIS FUNCTIONS ───────────────────────────────
//
// Each diagnostic is split into a compute_* function (pure, returns the
// values for one snapshot) and a write_* function (formats an already
// -averaged vector to a file). main() calls compute_* repeatedly over many
// snapshots during the run, accumulates the results per temperature, and
// only calls write_* once at the end on the time-averaged result -- see
// the "TIME-AVERAGED SAMPLING" section of main() for why (a single
// snapshot at the end is a very noisy estimate of the equilibrium value at
// a given temperature).

std::vector<double> compute_displacement_spectrum(const std::vector<double>& h_u, int Nx, int Ny) {
    int num_modes = Ny / 2 + 1;
    std::vector<double> S_u(num_modes, 0.0);

    for (int x = 0; x < Nx; ++x) {
        for (int k = 0; k < num_modes; ++k) {
            std::complex<double> fourier_sum(0.0, 0.0);
            double qy = 2.0 * M_PI * k / Ny;
            for (int y = 0; y < Ny; ++y) {
                double phase = qy * y;
                fourier_sum += std::complex<double>(
                     h_u[x * Ny + y] * std::cos(phase),
                    -h_u[x * Ny + y] * std::sin(phase));
            }
            S_u[k] += std::norm(fourier_sum) / (Ny * Ny);
        }
    }
    for (auto& v : S_u) v /= Nx;
    return S_u;
}

void write_displacement_spectrum(const std::vector<double>& S_u, int Ny, std::ofstream& outfile) {
    outfile << "# k    qy    S_u(qy)\n";
    for (int k = 1; k < (int)S_u.size(); ++k) {
        double qy = 2.0 * M_PI * k / Ny;
        outfile << k << "    " << qy << "    " << S_u[k] << "\n";
    }
    outfile.close();
}

std::vector<double> compute_structure_factor(const std::vector<double>& h_u, int Nx, int Ny) {
    int num_modes = Ny / 2 + 1;
    std::vector<double> S_avg(num_modes, 0.0);

    for (int x = 0; x < Nx; ++x) {
        for (int k = 0; k < num_modes; ++k) {
            std::complex<double> fourier_sum(0.0, 0.0);
            double qy = 2.0 * M_PI * k / Ny;
            for (int y = 0; y < Ny; ++y) {
                double phase = qy * y + 2.0 * M_PI * h_u[x * Ny + y];
                fourier_sum += std::complex<double>(std::cos(phase), std::sin(phase));
            }
            S_avg[k] += std::norm(fourier_sum) / Ny;
        }
    }
    for (auto& v : S_avg) v /= Nx;
    return S_avg;
}

void write_structure_factor(const std::vector<double>& S, int Ny, std::ofstream& outfile) {
    outfile << "# k    qy    S(qy)\n";
    for (int k = 0; k < (int)S.size(); ++k) {
        double qy = 2.0 * M_PI * k / Ny;
        outfile << k << "    " << qy << "    " << S[k] << "\n";
    }
    outfile.close();
}

// The two functions above transform along y (the internal coordinate along
// a chain's own length) -- they measure single-chain roughness/coherence,
// not the vortex-lattice translational order. u(x,y) is a displacement in
// the SAME direction as x (see the HARDCORE constraint in
// update_displacements_thermal_kernel, which compares u against its
// x-neighbors at the same y to keep chains from crossing), so the actual
// vortex-lattice Bragg peak lives in the Fourier transform along x, at
// fixed y, averaged over y -- these two functions do exactly that.

std::vector<double> compute_transverse_spectrum(const std::vector<double>& h_u, int Nx, int Ny) {
    int num_modes = Nx / 2 + 1;
    std::vector<double> S_u(num_modes, 0.0);

    for (int y = 0; y < Ny; ++y) {
        for (int k = 0; k < num_modes; ++k) {
            std::complex<double> fourier_sum(0.0, 0.0);
            double qx = 2.0 * M_PI * k / Nx;
            for (int x = 0; x < Nx; ++x) {
                double phase = qx * x;
                fourier_sum += std::complex<double>(
                     h_u[x * Ny + y] * std::cos(phase),
                    -h_u[x * Ny + y] * std::sin(phase));
            }
            S_u[k] += std::norm(fourier_sum) / (Nx * Nx);
        }
    }
    for (auto& v : S_u) v /= Ny;
    return S_u;
}

void write_transverse_spectrum(const std::vector<double>& S_u, int Nx, std::ofstream& outfile) {
    outfile << "# k    qx    S_u(qx)\n";
    for (int k = 1; k < (int)S_u.size(); ++k) {
        double qx = 2.0 * M_PI * k / Nx;
        outfile << k << "    " << qx << "    " << S_u[k] << "\n";
    }
    outfile.close();
}

// Genuinely exact density structure factor, S_rho(q) = <|sum_x
// e^{iq(x+u(x,y))}|^2>/Nx, swept over the WHOLE first Brillouin zone
// q in [0, 2*pi] (k = 0..Nx inclusive) using the literal phase q*(x+u)
// -- not "q*x + 2*pi*u", which silently drops the q*u cross term and
// is only valid for q << 1/u. That approximation was an earlier, now
// -removed version of this function; dropping it is what fixes a
// spurious discontinuity between the q~0 and q~2*pi regions (they were
// being computed two different, inconsistent ways, stitched under a
// 2*pi-periodicity assumption on S_rho that is FALSE once u != 0:
// rho_hat(q+2*pi,y) = rho_hat(q,y) * e^{-i*2*pi*u(x,y)} != rho_hat(q,y)
// in general). q=0 and q=2*pi are genuinely different points -- q=0 is
// the trivial, disorder-independent peak (always exactly Nx, particle
// number conservation), q=2*pi is the informative Bragg peak -- so a
// real jump between them is expected physics, not a plotting bug.
std::vector<double> compute_transverse_structure_factor(const std::vector<double>& h_u, int Nx, int Ny) {
    int num_modes = Nx + 1;  // k = 0..Nx inclusive, closing the [0, 2*pi] interval
    std::vector<double> S_avg(num_modes, 0.0);

    for (int y = 0; y < Ny; ++y) {
        for (int k = 0; k < num_modes; ++k) {
            std::complex<double> fourier_sum(0.0, 0.0);
            double qx = 2.0 * M_PI * k / Nx;
            for (int x = 0; x < Nx; ++x) {
                double phase = qx * (x + h_u[x * Ny + y]);
                fourier_sum += std::complex<double>(std::cos(phase), std::sin(phase));
            }
            S_avg[k] += std::norm(fourier_sum) / Nx;
        }
    }
    for (auto& v : S_avg) v /= Ny;
    return S_avg;
}

void write_transverse_structure_factor(const std::vector<double>& S, int Nx, std::ofstream& outfile) {
    outfile << "# k    qx    S(qx)\n";
    for (int k = 0; k < (int)S.size(); ++k) {
        double qx = 2.0 * M_PI * k / Nx;
        outfile << k << "    " << qx << "    " << S[k] << "\n";
    }
    outfile.close();
}

std::pair<std::vector<double>, std::vector<double>>
compute_correlation(const std::vector<double>& h_u, int Nx, int Ny) {
    std::vector<double> B_y(Ny / 2, 0.0);
    std::vector<double> B_x(Nx / 2, 0.0);

    for (int dy = 0; dy < Ny / 2; ++dy) {
        double sum = 0.0;
        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y) {
                double diff = h_u[x * Ny + y] - h_u[x * Ny + ((y + dy) % Ny)];
                sum += diff * diff;
            }
        B_y[dy] = sum / (Nx * Ny);
    }

    for (int dx = 0; dx < Nx / 2; ++dx) {
        double sum = 0.0;
        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y) {
                double diff = h_u[x * Ny + y] - h_u[((x + dx) % Nx) * Ny + y];
                sum += diff * diff;
            }
        B_x[dx] = sum / (Nx * Ny);
    }

    return {B_y, B_x};
}

void write_correlation(const std::vector<double>& B_y, const std::vector<double>& B_x,
                        int Nx, int Ny, std::ofstream& outfile) {
    outfile << "# r    B_y(Along Chain)    B_x(Across Chains)\n";
    int max_r = std::max(Nx / 2, Ny / 2);
    for (int r = 0; r < max_r; ++r) {
        outfile << r << "    ";
        outfile << (r < (int)B_y.size() ? std::to_string(B_y[r]) : "nan") << "    ";
        outfile << (r < (int)B_x.size() ? std::to_string(B_x[r]) : "nan") << "\n";
    }
    outfile.close();
}

// ─────────────────────────────── MAIN ────────────────────────────────────────

#ifndef NREPLICAS
#define NREPLICAS 1
#endif

int main(int argc, char* argv[]) {
    // Usage:
    //   ./program <seedD> <seedT>
    //   ./program <seedD> <seedT> --params params.ini
    // FIX: require at least 3 arguments (seedD + seedT) to avoid argv[2] out-of-bounds UB.
    if (argc < 3) {
        std::cerr << "Usage: ./program <seedD> <seedT> [--params <file>]\n";
        return 1;
    }

    SimParams p;

    // Parse optional --params flag
    for (int i = 3; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--params") {
            try {
                p = SimParams::fromFile(argv[i + 1]);
                std::cout << "Loaded parameters from: " << argv[i + 1] << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error loading parameter file: " << e.what() << "\n";
                return 1;
            }
            break;
        }
    }

    // CLI seeds always win
    p.seedD = static_cast<unsigned int>(std::atoi(argv[1]));
    p.seedT = static_cast<unsigned int>(std::atoi(argv[2]));

    const int n_steps = 1000000;

    // Print simulation parameters
    std::cout << "Simulation Parameters:\n"
              << "  Grid Size  : " << p.Nx << " x " << p.Ny << "\n"
              << "  Time Steps : " << n_steps << "\n"
              << "  cx, cy     : " << p.cx << ", " << p.cy << "\n"
              << "  V0         : " << p.V0 << "\n"
              << "  dt         : " << p.dt << "\n"
              << "  kBT        : " << p.kBT << "\n"
              << "  rf         : " << p.rf << "\n"
              << "  seedD      : " << p.seedD << "\n"
              << "  seedT      : " << p.seedT << "\n";
#ifndef LARKIN
    std::cout << "  Disorder   : Piecewise-constant pinning (rf = " << p.rf << ")\n";
#else
    std::cout << "  Disorder   : Larkin random force\n";
#endif
#ifdef HARDCORE
    std::cout << "  Exclusion  : Hardcore (0.9 minimum spacing)\n";
#else
    std::cout << "  Exclusion  : None\n";
#endif
    std::cout << "  Replicas   : " << NREPLICAS << "\n";

    // ── Inicializar contexto CUDA explícitamente ──────────────────────────────
    // En algunos sistemas/drivers el runtime no crea el contexto de forma lazy;
    // cualquier cudaMalloc anterior a esto devuelve "unknown error".
    // cudaFree(nullptr) es el idiom estándar para forzar la creación del contexto.
    {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::cerr << "No CUDA devices found.\n";
            return 1;
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));  // fuerza inicialización del contexto
        std::cout << "CUDA context initialized on device 0.\n";
    }

    // ── Build replica ladder ──────────────────────────────────────────────────
    const int n_replicas = NREPLICAS;
    std::vector<std::unique_ptr<CoupledElasticChains>> replicas;
    replicas.reserve(n_replicas);

    const unsigned int seedD_orig = p.seedD;
    const unsigned int seedT_orig = p.seedT;
    const double       kBT_orig   = p.kBT;

    const double T_min = p.kBT;
    const double T_max = 1.0;
    // Geometric (log-uniform) ladder: T_i = T_min * (T_max/T_min)^(i/(N-1)).
    // Linear spacing makes the swap acceptance between adjacent replicas
    // collapse at low T -- Delta_beta = 1/T_i - 1/T_{i+1} ~ Delta_T/T^2
    // diverges as T -> 0 for fixed Delta_T -- exactly where a glassy system
    // needs replica exchange working best, and it also bunches most
    // replicas near T_max when T_min/T_max spans a decade or more (e.g.
    // T_min=0.01, T_max=1, N=5 linear gives 0.01, 0.26, 0.51, 0.75, 1 --
    // only one point below 0.5). Geometric spacing keeps
    // Delta_beta ~ (ratio-1)/T instead, roughly constant across the whole
    // ladder, and spends replicas proportionally across decades of T.
    const double ratio = (n_replicas > 1) ? std::pow(T_max / T_min, 1.0 / (n_replicas - 1)) : 1.0;

    // ladder[s] is the fixed temperature value that always lives in "slot" s
    // of the parallel-tempering ladder. Replica exchange only ever swaps
    // these same n_replicas values between replica objects (see the SWAP
    // block below) -- it never introduces a new value -- so accumulating
    // time-averaged observables per ladder slot (found via ladder_index()
    // on a replica's CURRENT kBT) rather than per replica object index is
    // exactly the right way to gather statistics at a fixed temperature
    // across the whole run, despite replicas migrating between slots.
    std::vector<double> ladder(n_replicas);
    for (int i = 0; i < n_replicas; ++i) {
        ladder[i] = T_min * std::pow(ratio, i);
        p.kBT     = ladder[i];
        p.seedT   = static_cast<unsigned int>(std::atoi(argv[2])) + i * 1000u;
        replicas.push_back(std::make_unique<CoupledElasticChains>(p));
    }
    auto ladder_index = [&](double T) {
        int best = 0;
        double best_diff = std::fabs(ladder[0] - T);
        for (int s = 1; s < n_replicas; ++s) {
            double diff = std::fabs(ladder[s] - T);
            if (diff < best_diff) { best_diff = diff; best = s; }
        }
        return best;
    };

    std::cout << "Running simulation...\n";

    // ── Main loop ─────────────────────────────────────────────────────────────
    const int swap_interval = 100;
    std::mt19937 swap_gen(54321);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    // ── TIME-AVERAGED SAMPLING ───────────────────────────────────────────────
    // A single snapshot at the end of the run is a very noisy estimate of
    // the equilibrium value at a given temperature. Instead, after a
    // burn-in period, periodically sample every replica's current
    // configuration and accumulate its diagnostics into the ladder slot
    // matching its CURRENT temperature (not its replica index, which
    // drifts under replica exchange -- see ladder_index() above). The
    // final files are the time-average of ~900 decorrelated-ish snapshots
    // per temperature instead of one.
    const int burn_in         = n_steps / 10;   // discard the first 10% while the system thermalizes
    const int sample_interval = 1000;           // ~900 samples per temperature over the full run
    long long n_samples = 0;

    std::vector<std::vector<double>> su_y_sum  (n_replicas, std::vector<double>(p.Ny / 2 + 1, 0.0));
    std::vector<std::vector<double>> srho_y_sum(n_replicas, std::vector<double>(p.Ny / 2 + 1, 0.0));
    std::vector<std::vector<double>> su_x_sum  (n_replicas, std::vector<double>(p.Nx / 2 + 1, 0.0));
    std::vector<std::vector<double>> srho_x_sum(n_replicas, std::vector<double>(p.Nx + 1, 0.0));
    std::vector<std::vector<double>> by_sum    (n_replicas, std::vector<double>(p.Ny / 2, 0.0));
    std::vector<std::vector<double>> bx_sum    (n_replicas, std::vector<double>(p.Nx / 2, 0.0));
    std::vector<double> h_u_sample;

    for (int step = 0; step < n_steps; ++step) {

        for (int i = 0; i < n_replicas; ++i)
            replicas[i]->step();

        // Periodically attempt temperature swaps (Parallel Tempering)
        if (step > 0 && step % swap_interval == 0) {
            // Alternate starting index to allow global diffusion
            int start_idx = ((step / swap_interval) % 2 == 0) ? 0 : 1;

            for (int i = start_idx; i < n_replicas - 1; i += 2) {
                int j = i + 1;

                double T_i = replicas[i]->get_kBT();
                double T_j = replicas[j]->get_kBT();
                double E_i = replicas[i]->compute_configurational_energy();
                double E_j = replicas[j]->compute_configurational_energy();

                double delta_beta = (1.0 / T_i) - (1.0 / T_j);
                double delta_E    = E_i - E_j;
                double arg        = delta_beta * delta_E;

                if (arg >= 0.0 || uniform_dist(swap_gen) < std::exp(arg)) {
                    replicas[i]->set_kBT(T_j);
                    replicas[j]->set_kBT(T_i);
                    std::cout << "[SWAP] Accepted between replica " << i << " and " << j << "\n";
                }
            }
        }

        if (step >= burn_in && step % sample_interval == 0) {
            for (int i = 0; i < n_replicas; ++i) {
                int slot = ladder_index(replicas[i]->get_kBT());
                replicas[i]->copyToHost(h_u_sample);

                auto su_y = compute_displacement_spectrum(h_u_sample, p.Nx, p.Ny);
                for (size_t k = 0; k < su_y.size(); ++k) su_y_sum[slot][k] += su_y[k];

                auto srho_y = compute_structure_factor(h_u_sample, p.Nx, p.Ny);
                for (size_t k = 0; k < srho_y.size(); ++k) srho_y_sum[slot][k] += srho_y[k];

                auto su_x = compute_transverse_spectrum(h_u_sample, p.Nx, p.Ny);
                for (size_t k = 0; k < su_x.size(); ++k) su_x_sum[slot][k] += su_x[k];

                auto srho_x = compute_transverse_structure_factor(h_u_sample, p.Nx, p.Ny);
                for (size_t k = 0; k < srho_x.size(); ++k) srho_x_sum[slot][k] += srho_x[k];

                auto corr = compute_correlation(h_u_sample, p.Nx, p.Ny);
                for (size_t r = 0; r < corr.first.size();  ++r) by_sum[slot][r] += corr.first[r];
                for (size_t r = 0; r < corr.second.size(); ++r) bx_sum[slot][r] += corr.second[r];
            }
            ++n_samples;
        }

        if (step % 1000 == 0)
            std::cout << "Step " << step << " / " << n_steps << "\n";
    }

    std::cout << "Collected " << n_samples << " time-averaged samples per temperature "
              << "(burn_in=" << burn_in << ", sample_interval=" << sample_interval << ")\n";

    // ── Write time-averaged diagnostics, one file set per ladder slot ────────
    for (int s = 0; s < n_replicas; ++s) {
        if (n_samples == 0) break;  // burn_in >= n_steps -- nothing collected

        double T_s = ladder[s];
        std::stringstream ssS, ssB, ssRho, ssTS, ssTRho;
        ssS    << "displacement_spectra_replica_"  << T_s << ".dat";
        ssB    << "correlation_replica_"           << T_s << ".dat";
        ssRho  << "structure_factor_replica_"      << T_s << ".dat";
        ssTS   << "transverse_spectrum_replica_"   << T_s << ".dat";
        ssTRho << "transverse_structure_factor_replica_" << T_s << ".dat";

        std::ofstream outfile_S(ssS.str());
        std::ofstream outfile_B(ssB.str());
        std::ofstream outfile_Rho(ssRho.str());
        std::ofstream outfile_TS(ssTS.str());
        std::ofstream outfile_TRho(ssTRho.str());

        if (outfile_S.is_open() && outfile_B.is_open() && outfile_Rho.is_open()
                && outfile_TS.is_open() && outfile_TRho.is_open()) {
            auto average = [&](std::vector<double> v) {
                for (auto& x : v) x /= static_cast<double>(n_samples);
                return v;
            };
            write_displacement_spectrum(average(su_y_sum[s]), p.Ny, outfile_S);
            write_correlation(average(by_sum[s]), average(bx_sum[s]), p.Nx, p.Ny, outfile_B);
            write_structure_factor(average(srho_y_sum[s]), p.Ny, outfile_Rho);
            write_transverse_spectrum(average(su_x_sum[s]), p.Nx, outfile_TS);
            write_transverse_structure_factor(average(srho_x_sum[s]), p.Nx, outfile_TRho);
        }
    }

    // ── Also keep the plain final-snapshot diagnostics (same quantities,
    //    from the single configuration at the very end of the run, as
    //    opposed to time-averaged over n_samples of them above) under a
    //    "snapshot_" prefix, so the two can be compared directly. ─────────
    std::vector<double> h_u_final;
    for (int i = 0; i < n_replicas; ++i) {
        double T_i = replicas[i]->get_kBT();
        std::stringstream ssS, ssB, ssRho, ssTS, ssTRho;
        ssS    << "snapshot_displacement_spectra_replica_"       << T_i << ".dat";
        ssB    << "snapshot_correlation_replica_"                << T_i << ".dat";
        ssRho  << "snapshot_structure_factor_replica_"           << T_i << ".dat";
        ssTS   << "snapshot_transverse_spectrum_replica_"        << T_i << ".dat";
        ssTRho << "snapshot_transverse_structure_factor_replica_" << T_i << ".dat";

        std::ofstream outfile_S(ssS.str());
        std::ofstream outfile_B(ssB.str());
        std::ofstream outfile_Rho(ssRho.str());
        std::ofstream outfile_TS(ssTS.str());
        std::ofstream outfile_TRho(ssTRho.str());

        if (outfile_S.is_open() && outfile_B.is_open() && outfile_Rho.is_open()
                && outfile_TS.is_open() && outfile_TRho.is_open()) {
            replicas[i]->copyToHost(h_u_final);
            auto corr = compute_correlation(h_u_final, p.Nx, p.Ny);
            write_correlation(corr.first, corr.second, p.Nx, p.Ny, outfile_B);
            write_displacement_spectrum(compute_displacement_spectrum(h_u_final, p.Nx, p.Ny), p.Ny, outfile_S);
            write_structure_factor(compute_structure_factor(h_u_final, p.Nx, p.Ny), p.Ny, outfile_Rho);
            write_transverse_spectrum(compute_transverse_spectrum(h_u_final, p.Nx, p.Ny), p.Nx, outfile_TS);
            write_transverse_structure_factor(compute_transverse_structure_factor(h_u_final, p.Nx, p.Ny), p.Nx, outfile_TRho);
        }
    }

    // ── Save parameters ───────────────────────────────────────────────────────
    std::ofstream param_file("simulation_parameters.txt");
    param_file << "Grid Size: "  << p.Nx << " x " << p.Ny << "\n"
               << "Time Steps: " << n_steps << "\n"
               << "cx, cy: "     << p.cx << ", " << p.cy << "\n"
               << "V0: "         << p.V0 << "\n"
               << "dt: "         << p.dt << "\n"
               << "kBT: "        << kBT_orig << "\n"
               << "rf: "         << p.rf << "\n"
               << "seedT: "      << seedT_orig << "\n"
               << "seedD: "      << seedD_orig << "\n"
               << "Replicas: "   << n_replicas << "\n"
               << "BurnIn: "     << burn_in << "\n"
               << "SampleInterval: " << sample_interval << "\n"
               << "NSamples: "   << n_samples << "\n";
#ifndef LARKIN
    param_file << "Disorder: Piecewise-constant pinning\n";
#else
    param_file << "Disorder: Larkin random force\n";
#endif
#ifdef HARDCORE
    param_file << "Exclusion: Hardcore\n";
#else
    param_file << "Exclusion: None\n";
#endif
    param_file.close();

    return 0;
}