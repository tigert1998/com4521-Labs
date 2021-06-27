#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char *);
void random_ints(int *a);

__device__ int d_a[N], d_b[N], d_c[N];

__global__ void vectorAdd(int max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < max) d_c[i] = d_a[i] + d_b[i];
}

void QueryDevices() {
  int n;

  cudaGetDeviceCount(&n);
  for (int i = 0; i < n; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) * 1e3 /
               (1 << 30));
  }
}

int main(void) {
  QueryDevices();

  int *a, *b, *c, *c_ref;  // host copies of a, b, c
  unsigned int size = N * sizeof(int);

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size);
  random_ints(a);
  b = (int *)malloc(size);
  random_ints(b);
  c = (int *)malloc(size);
  c_ref = (int *)malloc(size);
  for (int i = 0; i < N; i++) c_ref[i] = a[i] + b[i];

  cudaMemcpyToSymbol(d_a, a, size);
  cudaMemcpyToSymbol(d_b, b, size);
  checkCUDAError("CUDA memcpy");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // Launch add() kernel on GPU
  vectorAdd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
              THREADS_PER_BLOCK>>>(N);
  cudaEventRecord(stop);
  checkCUDAError("CUDA kernel");

  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("vectorAdd takes %.3lfms\n", ms);
  printf("Measured Global Memory Bandwidth (GB/s): %f\n",
         size * 3 / ms * 1e3 / (1 << 30));

  // Copy result back to host
  cudaMemcpyFromSymbol(c, d_c, size);
  checkCUDAError("CUDA memcpy");
  for (int i = 0; i < N; i++)
    if (c_ref[i] != c[i]) {
      puts("FAIL");
      break;
    }

  // Cleanup
  free(a);
  free(b);
  free(c);

  return 0;
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void random_ints(int *a) {
  for (unsigned int i = 0; i < N; i++) {
    a[i] = rand();
  }
}
