#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <complex>
#include <fstream>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
/*#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG
typedef r123::Philox4x32 RNG4; // particular counter-based RNG
*/

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

// Kernel to initialize cuRAND states for every single grid point
__global__ void init_rand_kernel(curandState* rand_states, int Nx, int Ny, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;
        // Each thread gets same seed, a unique sequence number, and no offset
        curand_init(42ULL, idx, 0, &rand_states[idx]);
    }
}

// Elegimos un número entero grande (por ejemplo, 10 millones)
// Debe ser lo suficientemente grande para que "x + u_curr" nunca sea negativo
#define PHILOX_OFFSET 10000000

// Función de device que toma el índice espacial (idx) y el bloque dinámico (u_block)
// y devuelve un número pseudoaleatorio determinista en el rango [-V0, V0]
__device__ float get_piecewise_pinning_force(unsigned int idx, unsigned int u_block, float V0, unsigned int seed) {
    // 1. Inicializar el estado de Philox al vuelo (on-the-fly)
    // El primer argumento es el "Contador" (usamos u_block)
    // El segundo argumento es la "Clave/Key" (usamos el índice de la grilla idx)
    // El tercer argumento es el offset (0 de forma estándar)
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, u_block, &state);

    // 2. Generar un número flotante uniforme en el rango (0.0, 1.0]
    float rand_val = curand_uniform(&state);

    // 3. Escalarlo linealmente al rango [-V0, V0]
    return -V0 + 2.0f * V0 * rand_val;
}

__device__ float quenched_random_force(
    unsigned int x,
    unsigned int y,
    float V0,
    unsigned int seed)
{
    curandStatePhilox4_32_10_t state;
    curand_init(seed, y, x, &state);
    return -V0 + 2.0f*V0*curand_uniform(&state);
}



// Updated CUDA Kernel incorporating Thermal Noise
__global__ void update_displacements_thermal_kernel(const float* __restrict__ d_u, 
                                                    float* __restrict__ d_u_next, 
                                                    const float* __restrict__ d_phi, 
                                                    curandState* rand_states, unsigned int seed,
                                                    int Nx, int Ny, 
                                                    float cx, float cy, float V0, float rf,
                                                    float dt, float noise_scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny) {
        int idx = x * Ny + y;

        // Periodic Boundary Conditions
        int left_x  = (x - 1 + Nx) % Nx;
        int right_x = (x + 1) % Nx;
        int up_y    = (y - 1 + Ny) % Ny;
        int down_y  = (y + 1) % Ny;

        float u_curr  = d_u[idx];
        float u_left  = d_u[left_x * Ny + y];
        float u_right = d_u[right_x * Ny + y];
        float u_up    = d_u[x * Ny + up_y];
        float u_down  = d_u[x * Ny + down_y];

        // 1. Deterministic Elastic Force
        float f_elastic = cx * (u_left + u_right - 2.0f * u_curr) + 
                          cy * (u_up + u_down - 2.0f * u_curr);

        // 2. Escudo de exclusión con parachoques (Antidivergencias)
        float f_exclusion_x = 0.0f;
        /*float A_rep = 0.1f;         // Fuerza de la barrera
        float d_critica = 0.4f;     // Umbral de activación (distancia < 0.4)
        float d_minima = 0.05f;     // Parachoques absoluto para evitar división por cero

        // Vecino derecho acercándose por la izquierda
        float dist_derecha = 1.0f + (u_right - u_curr);
        if (dist_derecha < d_critica) {
            // Si la distancia es peligrosamente baja, la congelamos en d_minima
            float d_reg = (dist_derecha < d_minima) ? d_minima : dist_derecha;
            f_exclusion_x -= A_rep / (d_reg * d_reg); 
        }

        // Vecino izquierdo acercándose por la derecha
        float dist_izquierda = 1.0f + (u_curr - u_left);
        if (dist_izquierda < d_critica) {
            float d_reg = (dist_izquierda < d_minima) ? d_minima : dist_izquierda;
            f_exclusion_x += A_rep / (d_reg * d_reg);
        }*/     
        f_elastic += f_exclusion_x; // Agrega la fuerza de exclusión al término elástico
        


        // 2. Deterministic Pinning Force
        int u_block_signed = __float2int_rd((static_cast<float>(x) + u_curr) / rf);
        // Al sumarle el offset, garantizamos que el resultado sea >= 0 
        // y el casteo a unsigned int preserva el orden y la continuidad
        unsigned int u_block_philox = (unsigned int)(u_block_signed + PHILOX_OFFSET);
        #ifndef LARKIN
        float f_pinning = get_piecewise_pinning_force(y, u_block_philox, V0, seed);
        #else
        float f_pinning = quenched_random_force(x, y, V0, seed);
        #endif

        // 3. Thermal Noise Generation (Gaussian distributed) via cuRAND
        curandState local_state = rand_states[idx]; // Copy state to local register
        float gaussian_noise = curand_normal(&local_state);
        rand_states[idx] = local_state;             // Write state back to global memory

        // 4. Langevin Update Equation
        float u_prop = u_curr + dt * (f_elastic + f_pinning) + noise_scale * gaussian_noise;

        #ifdef HARDCORE
        // Garantizar geométricamente la condición de no-cruce respecto a las posiciones actuales
        // 
        float max_desplazamiento_izq = u_prop - u_left ;  // No puede acercarse a menos de 0.4 del vecino izquierdo
        float max_desplazamiento_der = u_right - u_prop; // No puede acercarse a menos de 0.4 del vecino derecho

        if (max_desplazamiento_izq < -0.9) u_prop = u_left - 0.9f; // Si se acerca demasiado al vecino izquierdo, lo frenamos
        if (max_desplazamiento_der > 0.9) u_prop = u_right - 0.9f; // Si se acerca demasiado al vecino derecho, lo frenamos 
        #endif

        d_u_next[idx] = u_prop;
    }
}

void compute_and_save_displacement_spectra(const std::vector<float>& h_u, int Nx, int Ny) {
    int num_modes = Ny / 2 + 1; 
    std::vector<double> S_u(num_modes, 0.0);

    for (int x = 0; x < Nx; ++x) {
        for (int k = 0; k < num_modes; ++k) {
            std::complex<double> fourier_sum(0.0, 0.0);
            double qy = 2.0 * M_PI * k / Ny;

            for (int y = 0; y < Ny; ++y) {
                // Transformamos directamente el campo de desplazamientos u
                double phase = qy * y;
                fourier_sum += std::complex<double>(h_u[x * Ny + y] * std::cos(phase), 
                                                    -h_u[x * Ny + y] * std::sin(phase));
            }
            S_u[k] += (std::norm(fourier_sum) / (Ny * Ny)); // Espectro de potencia de u
        }
    }

    std::ofstream outfile("displacement_spectra.dat");
    outfile << "# k    qy    S_u(qy)\n";
    for (int k = 1; k < num_modes; ++k) { // Ignoramos k=0 (traslación global)
        double qy = 2.0 * M_PI * k / Ny;
        double final_S = S_u[k] / Nx;
        outfile << k << "    " << qy << "    " << final_S << "\n";
    }
    outfile.close();
}

// New Function: Computes 1D Structure Factor along chains, averaged over all chains
void compute_and_save_structure_factor(const std::vector<float>& h_u, int Nx, int Ny) {
    int num_modes = Ny / 2 + 1; 
    std::vector<double> S_avg(num_modes, 0.0);

    std::cout << "Computing chain-averaged structure factor S(qy) on Host..." << std::endl;

    // Loop over each chain
    for (int x = 0; x < Nx; ++x) {
        // Loop over each Fourier mode k
        for (int k = 0; k < num_modes; ++k) {
            std::complex<double> fourier_sum(0.0, 0.0);
            double qy = 2.0 * M_PI * k / Ny;

            // Compute Fourier component of the density phase field along chain x
            for (int y = 0; y < Ny; ++y) {
                double phase = qy * y + 2.0 * M_PI * h_u[x * Ny + y];
                fourier_sum += std::complex<double>(std::cos(phase), std::sin(phase));
            }

            // Square modulus divided by Ny
            double Sq_chain = std::norm(fourier_sum) / Ny;
            S_avg[k] += Sq_chain;
        }
    }

    // Average over all chains and export
    std::ofstream outfile("structure_factor_output.dat");
    outfile << "# k    qy    S(qy)\n";
    for (int k = 0; k < num_modes; ++k) {
        double qy = 2.0 * M_PI * k / Ny;
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
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                float diff = h_u[x * Ny + y] - h_u[x * Ny + ((y + dy) % Ny)];
                sum += diff * diff;
            }
        }
        B_y[dy] = static_cast<float>(sum / (Nx * Ny));
    }

    for (int dx = 0; dx < Nx / 2; ++dx) {
        double sum = 0.0;
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                float diff = h_u[x * Ny + y] - h_u[((x + dx) % Nx) * Ny + y];
                sum += diff * diff;
            }
        }
        B_x[dx] = static_cast<float>(sum / (Nx * Ny));
    }

    std::ofstream outfile("bragg_glass_thermal_output.dat");
    outfile << "# r    B_y(Along Chain)    B_x(Across Chains)\n";
    int max_r = (Nx / 2 > Ny / 2) ? Nx / 2 : Ny / 2;
    for (int r = 0; r < max_r; ++r) {
        outfile << r << "    ";
        if (r < Ny / 2) outfile << B_y[r] << "    ";
        else outfile << "nan    ";
        if (r < Nx / 2) outfile << B_x[r] << "\n";
        else outfile << "nan\n";
    }
    outfile.close();
    std::cout << "Data exported to 'bragg_glass_thermal_output.dat'" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < 2){
        std::cerr << "Usage: ./program seed\n";
        return 1;
    }

    unsigned int seed = atoi(argv[1]);

    const int Nx = 32;       
    const int Ny = 512;       
    const int n_steps = 1000000; // Increased steps to settle with thermal fluctuations
    
    const float cx = 1.0f;    
    const float cy = 1.0f;    
    const float V0 = 0.1f;    
    const float dt = 0.01f;   // Slower time-step helps ensure stochastic stability
    const float rf = 10.0f; // Longitud de correlación del desorden (¡Auméntala para ver más Larkin!)

    // Thermal Temperature Parameters
    const float kBT = 0.5f;   // System temperature (set to 0.0f to revert to pure relaxation)
    const float noise_scale = std::sqrt(2.0f * kBT * dt);

    size_t grid_size = Nx * Ny;
    size_t bytes = grid_size * sizeof(float);

    // print all parameters
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Grid Size: " << Nx << " x " << Ny << std::endl;
    std::cout << "Time Steps: " << n_steps << std::endl;
    std::cout << "Elastic Constants: cx = " << cx << ", cy = " << cy << std::endl;
    std::cout << "Pinning Strength: V0 = " << V0 << std::endl;
    std::cout << "Time Step: dt = " << dt << std::endl;
    std::cout << "Thermal Noise Scale: " << noise_scale << " (kBT = " << kBT << ")" << std::endl;
    std::cout << "Correlation Length of Disorder (rf): " << rf << std::endl;
    #ifndef LARKIN
    std::cout << "Disorder Type: Pure Sinusoidal Pinning\n";
    std::cout << "Correlation Length of Disorder (rf): " << rf << "\n";
    #else
    std::cout << "Disorder Type: Larkin Random Force\n";
    #endif
    #ifdef HARDCORE
    std::cout << "Exclusion Type: Hardcore with 0.4 minimum spacing\n";
    #else
    std::cout << "Exclusion Type: None (Purely Elastic)\n";
    #endif
    std::cout << "Random Seed: " << atoi(argv[1]) << "\n";  


    std::vector<float> h_u(grid_size, 0.0f);

    std::vector<float> h_phi(grid_size);
    std::mt19937 prng(atoi(argv[1])); // Seed from command line argument for reproducibility
    std::uniform_real_distribution<float> dist_phi(0.0f, 2.0f * M_PI);
    for (size_t i = 0; i < grid_size; ++i) {
        h_phi[i] = dist_phi(prng);
    }

    float *d_u, *d_u_next, *d_phi;
    curandState* d_rand_states;

    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, bytes));
    CUDA_CHECK(cudaMalloc(&d_phi, bytes));
    CUDA_CHECK(cudaMalloc(&d_rand_states, grid_size * sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi, h_phi.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Step 1: Initialize random engine states on the GPU
    std::cout << "Initializing cuRAND state matrix..." << std::endl;
    init_rand_kernel<<<numBlocks, threadsPerBlock>>>(d_rand_states, Nx, Ny, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Main Simulation Loop
    std::cout << "Simulating with Thermal Noise scale: " << noise_scale << "..." << std::endl;
    for (int step = 0; step < n_steps; ++step) {
        update_displacements_thermal_kernel<<<numBlocks, threadsPerBlock>>>(
            d_u, d_u_next, d_phi, d_rand_states, seed,Nx, Ny, cx, cy, V0, rf, dt, noise_scale
        );
        
        CUDA_CHECK(cudaGetLastError());

        // Double buffer pointer flip
        float* temp = d_u;
        d_u = d_u_next;
        d_u_next = temp;

        if (step % 1000 == 0) {
            std::cout << "Iteration Progress: " << step << " / " << n_steps << std::endl;
        }
    }

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_rand_states));

    std::ofstream param_file("simulation_parameters.txt");
    param_file << "Simulation Parameters:\n";
    param_file << "Grid Size: " << Nx << " x " << Ny << "\n";
    param_file << "Time Steps: " << n_steps << "\n";
    param_file << "Elastic Constants: cx = " << cx << ", cy = " << cy << "\n";
    param_file << "Pinning Strength: V0 = " << V0 << "\n";
    param_file << "Time Step: dt = " << dt << "\n";
    param_file << "Thermal Noise Scale: " << noise_scale << " (kBT = " << kBT << ")\n";
    #ifndef LARKIN
    param_file << "Disorder Type: Pure Sinusoidal Pinning\n";
    param_file << "Correlation Length of Disorder (rf): " << rf << "\n";
    #else
    param_file << "Disorder Type: Larkin Random Force\n";    
    #endif
    #ifdef HARDCORE
    param_file << "Exclusion Type: Hardcore with 0.4 minimum spacing\n";
    #else
    param_file << "Exclusion Type: None (Purely Elastic)\n";
    #endif
    param_file << "Random Seed: " << atoi(argv[1]) << "\n";
    param_file.close();


    compute_and_save_correlation(h_u, Nx, Ny);
    //compute_and_save_structure_factor(h_u, Nx, Ny);
    compute_and_save_displacement_spectra(h_u, Nx, Ny);
    
    return 0;
}
