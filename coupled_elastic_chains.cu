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

void compute_and_save_displacement_spectra(const std::vector<double>& h_u, int Nx, int Ny, std::ofstream& outfile) {
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

    outfile << "# k    qy    S_u(qy)\n";
    for (int k = 1; k < num_modes; ++k) {
        double qy      = 2.0 * M_PI * k / Ny;
        double final_S = S_u[k] / Nx;
        outfile << k << "    " << qy << "    " << final_S << "\n";
    }
    outfile.close();
}

void compute_and_save_structure_factor(const std::vector<double>& h_u, int Nx, int Ny, std::ofstream& outfile) {
    int num_modes = Ny / 2 + 1;
    std::vector<double> S_avg(num_modes, 0.0);

    std::cout << "Computing chain-averaged structure factor S(qy) on Host..." << std::endl;

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

    outfile << "# k    qy    S(qy)\n";
    for (int k = 0; k < num_modes; ++k) {
        double qy      = 2.0 * M_PI * k / Ny;
        double final_S = S_avg[k] / Nx;
        outfile << k << "    " << qy << "    " << final_S << "\n";
    }
    outfile.close();
    std::cout << "Structure factor exported." << std::endl;
}

void compute_and_save_correlation(const std::vector<double>& h_u, int Nx, int Ny, std::ofstream& outfile) {
    std::vector<double> B_y(Ny / 2, 0.0);
    std::vector<double> B_x(Nx / 2, 0.0);

    std::cout << "Computing correlation functions on Host..." << std::endl;

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

    outfile << "# r    B_y(Along Chain)    B_x(Across Chains)\n";
    int max_r = std::max(Nx / 2, Ny / 2);
    for (int r = 0; r < max_r; ++r) {
        outfile << r << "    ";
        outfile << (r < Ny / 2 ? std::to_string(B_y[r]) : "nan") << "    ";
        outfile << (r < Nx / 2 ? std::to_string(B_x[r]) : "nan") << "\n";
    }
    outfile.close();
    std::cout << "Correlation data exported." << std::endl;
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

    const double T_min   = p.kBT;
    const double T_max   = 1.0;
    const double delta_T = (n_replicas > 1) ? (T_max - T_min) / (n_replicas - 1) : 0.0;

    for (int i = 0; i < n_replicas; ++i) {
        p.kBT  = T_min + i * delta_T;
        p.seedT = static_cast<unsigned int>(std::atoi(argv[2])) + i * 1000u;
        replicas.push_back(std::make_unique<CoupledElasticChains>(p));
    }

    std::cout << "Running simulation...\n";

    // ── Main loop ─────────────────────────────────────────────────────────────
    const int swap_interval = 100;
    std::mt19937 swap_gen(54321);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

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

        if (step % 1000 == 0)
            std::cout << "Step " << step << " / " << n_steps << "\n";
    }

    // ── Copy result to host & analyse ─────────────────────────────────────────
    std::vector<double> h_u;

    for (int i = 0; i < n_replicas; ++i) {
        std::stringstream ssS, ssB;

        double T_i = replicas[i]->get_kBT();
        ssS << "displacement_spectra_replica_" << T_i << ".dat";
        ssB << "correlation_replica_"          << T_i << ".dat";

        std::ofstream outfile_S(ssS.str());
        std::ofstream outfile_B(ssB.str());

        if (outfile_S.is_open() && outfile_B.is_open()) {
            replicas[i]->copyToHost(h_u);
            compute_and_save_correlation(h_u, p.Nx, p.Ny, outfile_B);
            compute_and_save_displacement_spectra(h_u, p.Nx, p.Ny, outfile_S);
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
               << "Replicas: "   << n_replicas << "\n";
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