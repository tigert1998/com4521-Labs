#ifndef COMMON_CUH_
#define COMMON_CUH_

#include <cstdio>

class CudaTimer {
  cudaEvent_t start_, stop_;

 public:
  inline CudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  inline float End() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float ms;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

  inline ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
};

inline void CheckCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#endif