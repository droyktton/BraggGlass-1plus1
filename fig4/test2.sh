# ¿Qué versión de CUDA toolkit está instalada?
nvcc --version
ls /usr/local/cuda* -la

# ¿Contra qué libcuda linkea el binario compilado?
ldd /tmp/test_cuda

# ¿Hay conflicto entre cuda toolkit y driver?
# Driver 535 soporta hasta CUDA 12.2 — si el toolkit es más nuevo hay mismatch
dpkg -l | grep -i cuda | head -20
