# 1. ¿El proceso tiene acceso a la GPU?
ls -la /dev/nvidia*

# 2. ¿Hay otro proceso bloqueando la GPU?
nvidia-smi pmon -c 1

# 3. ¿El binario puede linkear el runtime correctamente?
ldd ./sim | grep cuda

# 4. Test mínimo de CUDA
cat > /tmp/test_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    cudaError_t e = cudaFree(nullptr);
    printf("cudaFree(nullptr): %s\n", cudaGetErrorString(e));
    int n; 
    e = cudaGetDeviceCount(&n);
    printf("cudaGetDeviceCount: %s, count=%d\n", cudaGetErrorString(e), n);
    return 0;
}
EOF
nvcc -arch=sm_86 /tmp/test_cuda.cu -o /tmp/test_cuda && /tmp/test_cuda
