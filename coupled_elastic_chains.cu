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

#ifndef M_PI
#define M_PI 3.14159265358979323846f
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

__device__ float get_piecewise_pinning_force(unsigned int idx, unsigned int u_block,
                                              float V0, unsigned int seed) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, u_block, &state);
    return -V0 + 2.0f * V0 * curand_uniform(&state);
}

__device__ float quenched_random_force(unsigned int x, unsigned int y,
                                        float V0, unsigned int seed) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, y, x, &state);
    return -V0 + 2.0f * V0 * curand_uniform(&state);
}

// ─────────────────────────────── KERNELS ─────────────────────────────────────

__global__ void init_rand_kernel(curandState* rand_states, int Nx, int Ny,
                                  unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;
        curand_init(42ULL, idx, 0, &rand_states[idx]);
    }
}

__global__ void update_displacements_thermal_kernel(
        const float* __restrict__ d_u,
        float*       __restrict__ d_u_next,
        const float* __restrict__ d_phi,
        curandState* rand_states,
        unsigned int seed,
        int Nx, int Ny,
        float cx, float cy,
        float V0, float rf,
        float dt, float noise_scale) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;

        // Periodic boundary conditions
        int left_x  = (x - 1 + Nx) % Nx;
        int right_x = (x + 1) % Nx;
        int up_y    = (y - 1 + Ny) % Ny;
        int down_y  = (y + 1) % Ny;

        float u_curr  = d_u[idx];
        float u_left  = d_u[left_x  * Ny + y];
        float u_right = d_u[right_x * Ny + y];
        float u_up    = d_u[x * Ny + up_y];
        float u_down  = d_u[x * Ny + down_y];

        // 1. Elastic force
        float f_elastic = cx * (u_left + u_right - 2.0f * u_curr)
                        + cy * (u_up   + u_down  - 2.0f * u_curr);

        // 2. Pinning force
        int u_block_signed      = __float2int_rd((static_cast<float>(x) + u_curr) / rf);
        unsigned int u_block_ph = (unsigned int)(u_block_signed + PHILOX_OFFSET);

#ifndef LARKIN
        float f_pinning = get_piecewise_pinning_force(y, u_block_ph, V0, seed);
#else
        float f_pinning = quenched_random_force(x, y, V0, seed);
#endif

        // 3. Thermal noise (Gaussian via cuRAND)
        curandState local_state = rand_states[idx];
        float gaussian_noise    = curand_normal(&local_state);
        rand_states[idx]        = local_state;

        // 4. Langevin update
        float u_prop = u_curr + dt * (f_elastic + f_pinning) + noise_scale * gaussian_noise;

#ifdef HARDCORE
        float max_desplazamiento_izq = u_prop - u_left;
        float max_desplazamiento_der = u_right - u_prop;
        if (max_desplazamiento_izq < -0.9f) u_prop = u_left  - 0.9f;
        if (max_desplazamiento_der >  0.9f) u_prop = u_right - 0.9f;
#endif

        d_u_next[idx] = u_prop;
    }
}

// ──────────────────────── PHYSICAL SYSTEM CLASS ──────────────────────────────

struct SimParams {
    int          Nx   = 32;
    int          Ny   = 512;
    float        cx   = 1.0f;
    float        cy   = 1.0f;
    float        V0   = 0.1f;
    float        dt   = 0.01f;
    float        rf   = 10.0f;
    float        kBT  = 0.5f;
    unsigned int seed = 42u;

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
                if      (key == "Nx"  ) { int   v; ss >> v; p.Nx   = v; }
                else if (key == "Ny"  ) { int   v; ss >> v; p.Ny   = v; }
                else if (key == "cx"  ) { float v; ss >> v; p.cx   = v; }
                else if (key == "cy"  ) { float v; ss >> v; p.cy   = v; }
                else if (key == "V0"  ) { float v; ss >> v; p.V0   = v; }
                else if (key == "dt"  ) { float v; ss >> v; p.dt   = v; }
                else if (key == "rf"  ) { float v; ss >> v; p.rf   = v; }
                else if (key == "kBT" ) { float v; ss >> v; p.kBT  = v; }
                else if (key == "seed") { unsigned int v; ss >> v; p.seed = v; }
                else
                    throw std::runtime_error("Unknown parameter '" + key
                                             + "' on line " + std::to_string(line_no));
            } catch (const std::runtime_error&) {
                throw; // re-throw our own errors
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
          noise_scale_(std::sqrt(2.0f * p.kBT * p.dt))
    {
        size_t bytes = grid_size_ * sizeof(float);

        // Allocate device buffers
        CUDA_CHECK(cudaMalloc(&d_u_,          bytes));
        CUDA_CHECK(cudaMalloc(&d_u_next_,     bytes));
        CUDA_CHECK(cudaMalloc(&d_phi_,        bytes));
        CUDA_CHECK(cudaMalloc(&d_rand_states_, grid_size_ * sizeof(curandState)));

        // Initialise displacement field to zero on device
        CUDA_CHECK(cudaMemset(d_u_, 0, bytes));

        // Initialise random phase field on host, then copy to device
        std::vector<float> h_phi(grid_size_);
        std::mt19937 prng(params_.seed);
        std::uniform_real_distribution<float> dist_phi(0.0f, 2.0f * M_PI);
        for (auto& v : h_phi) v = dist_phi(prng);
        CUDA_CHECK(cudaMemcpy(d_phi_, h_phi.data(), bytes, cudaMemcpyHostToDevice));

        // Set up CUDA launch config
        threads_ = dim3(16, 16);
        blocks_  = dim3((params_.Nx + threads_.x - 1) / threads_.x,
                        (params_.Ny + threads_.y - 1) / threads_.y);

        // Initialise cuRAND states on the GPU
        init_rand_kernel<<<blocks_, threads_>>>(d_rand_states_, params_.Nx, params_.Ny, 42ULL);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CoupledElasticChains() {
        cudaFree(d_u_);
        cudaFree(d_u_next_);
        cudaFree(d_phi_);
        cudaFree(d_rand_states_);
    }

    // Non-copyable, non-movable (owns raw GPU pointers)
    CoupledElasticChains(const CoupledElasticChains&)            = delete;
    CoupledElasticChains& operator=(const CoupledElasticChains&) = delete;

    // ── Simulation interface ──────────────────────────────────────────────────

    /// Advance one Langevin time step on the GPU.
    void step() {
        update_displacements_thermal_kernel<<<blocks_, threads_>>>(
            d_u_, d_u_next_, d_phi_, d_rand_states_,
            params_.seed, params_.Nx, params_.Ny,
            params_.cx, params_.cy,
            params_.V0, params_.rf,
            params_.dt, noise_scale_
        );
        CUDA_CHECK(cudaGetLastError());

        // Double-buffer swap
        float* tmp = d_u_;
        d_u_       = d_u_next_;
        d_u_next_  = tmp;
    }

    /// Copy the current displacement field from device to the provided host vector.
    void copyToHost(std::vector<float>& h_u) const {
        h_u.resize(grid_size_);
        CUDA_CHECK(cudaMemcpy(h_u.data(), d_u_,
                              grid_size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    const SimParams& params()     const { return params_; }
    float            noiseScale() const { return noise_scale_; }
    int              gridSize()   const { return grid_size_; }

private:
    SimParams    params_;
    int          grid_size_;
    float        noise_scale_;

    float*       d_u_          = nullptr;
    float*       d_u_next_     = nullptr;
    float*       d_phi_        = nullptr;
    curandState* d_rand_states_ = nullptr;

    dim3 threads_;
    dim3 blocks_;
};

// ────────────────────────── ANALYSIS FUNCTIONS ───────────────────────────────

void compute_and_save_displacement_spectra(const std::vector<float>& h_u, int Nx, int Ny) {
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

    std::ofstream outfile("displacement_spectra.dat");
    outfile << "# k    qy    S_u(qy)\n";
    for (int k = 1; k < num_modes; ++k) {
        double qy      = 2.0 * M_PI * k / Ny;
        double final_S = S_u[k] / Nx;
        outfile << k << "    " << qy << "    " << final_S << "\n";
    }
    outfile.close();
}

void compute_and_save_structure_factor(const std::vector<float>& h_u, int Nx, int Ny) {
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

    std::ofstream outfile("structure_factor_output.dat");
    outfile << "# k    qy    S(qy)\n";
    for (int k = 0; k < num_modes; ++k) {
        double qy      = 2.0 * M_PI * k / Ny;
        double final_S = S_avg[k] / Nx;
        outfile << k << "    " << qy << "    " << final_S << "\n";
    }
    outfile.close();
    std::cout << "Structure factor exported to 'structure_factor_output.dat'" << std::endl;
}

void compute_and_save_correlation(const std::vector<float>& h_u, int Nx, int Ny) {
    std::vector<float> B_y(Ny / 2, 0.0f);
    std::vector<float> B_x(Nx / 2, 0.0f);

    std::cout << "Computing correlation functions on Host..." << std::endl;

    for (int dy = 0; dy < Ny / 2; ++dy) {
        double sum = 0.0;
        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y) {
                float diff = h_u[x * Ny + y] - h_u[x * Ny + ((y + dy) % Ny)];
                sum += diff * diff;
            }
        B_y[dy] = static_cast<float>(sum / (Nx * Ny));
    }

    for (int dx = 0; dx < Nx / 2; ++dx) {
        double sum = 0.0;
        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y) {
                float diff = h_u[x * Ny + y] - h_u[((x + dx) % Nx) * Ny + y];
                sum += diff * diff;
            }
        B_x[dx] = static_cast<float>(sum / (Nx * Ny));
    }

    std::ofstream outfile("bragg_glass_thermal_output.dat");
    outfile << "# r    B_y(Along Chain)    B_x(Across Chains)\n";
    int max_r = std::max(Nx / 2, Ny / 2);
    for (int r = 0; r < max_r; ++r) {
        outfile << r << "    ";
        outfile << (r < Ny / 2 ? std::to_string(B_y[r]) : "nan") << "    ";
        outfile << (r < Nx / 2 ? std::to_string(B_x[r]) : "nan") << "\n";
    }
    outfile.close();
    std::cout << "Data exported to 'bragg_glass_thermal_output.dat'" << std::endl;
}

// ─────────────────────────────── MAIN ────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Usage:
    //   ./program <seed>                      -- use hard-coded defaults
    //   ./program <seed> --params params.ini  -- load from file (CLI seed overrides file)
    if (argc < 2) {
        std::cerr << "Usage: ./program <seed> [--params <file>]\n";
        return 1;
    }

    SimParams p;

    // Parse optional --params flag
    for (int i = 2; i < argc - 1; ++i) {
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

    // CLI seed always wins — run the same config with different seeds
    // without editing the file.
    p.seed = static_cast<unsigned int>(std::atoi(argv[1]));

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
              << "  seed       : " << p.seed << "\n";
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

    // ── Build the physical system ─────────────────────────────────────────────
    CoupledElasticChains system(p);

    std::cout << "Noise scale: " << system.noiseScale() << "\n";
    std::cout << "Running simulation...\n";

    // ── Main loop ─────────────────────────────────────────────────────────────
    for (int step = 0; step < n_steps; ++step) {
        system.step();
        if (step % 1000 == 0)
            std::cout << "Step " << step << " / " << n_steps << "\n";
    }

    // ── Copy result to host & analyse ─────────────────────────────────────────
    std::vector<float> h_u;
    system.copyToHost(h_u);

    // Save parameters file
    std::ofstream param_file("simulation_parameters.txt");
    param_file << "Grid Size: "    << p.Nx << " x " << p.Ny << "\n"
               << "Time Steps: "   << n_steps << "\n"
               << "cx, cy: "       << p.cx << ", " << p.cy << "\n"
               << "V0: "           << p.V0 << "\n"
               << "dt: "           << p.dt << "\n"
               << "kBT: "          << p.kBT << "\n"
               << "noise_scale: "  << system.noiseScale() << "\n"
               << "rf: "           << p.rf << "\n"
               << "seed: "         << p.seed << "\n";
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

    compute_and_save_correlation(h_u, p.Nx, p.Ny);
    // compute_and_save_structure_factor(h_u, p.Nx, p.Ny);
    compute_and_save_displacement_spectra(h_u, p.Nx, p.Ny);

    return 0;
}
