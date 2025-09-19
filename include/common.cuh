#ifndef COMMON_CUH_
#define COMMON_CUH_

#include <cstdio>

#define TIME(message, ms, func_name, grid_size, block_size, shared_bytes, ...) \
  do {                                                                         \
    cudaEvent_t start, stop;                                                   \
    cudaEventCreate(&start);                                                   \
    cudaEventCreate(&stop);                                                    \
    cudaEventRecord(start, 0);                                                 \
    func_name<<<grid_size, block_size, shared_bytes>>>(__VA_ARGS__);           \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    CheckCUDAError(message);                                                   \
    cudaEventElapsedTime(&ms, start, stop);                                    \
    cudaEventDestroy(start);                                                   \
    cudaEventDestroy(stop);                                                    \
  } while (0)

inline void CheckCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#endif