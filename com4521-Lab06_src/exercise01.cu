#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.cuh"

#define A_HEIGHT 512
#define A_WIDTH 1024
#define B_HEIGHT 1024
#define B_WIDTH 2048
#define C_HEIGHT A_HEIGHT
#define C_WIDTH B_WIDTH

#define BLOCK_SIZE 8
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

static_assert(A_WIDTH == B_HEIGHT, "A_HEIGHT != B_WIDTH");
static_assert(A_WIDTH % BLOCK_SIZE == 0 && A_HEIGHT % BLOCK_SIZE == 0 &&
                  B_HEIGHT % BLOCK_SIZE == 0,
              "matrix size is not a multiple of the block size");

// matrix

template <typename D, int NRows, int NCols>
using RowMajorMatrix = D[NRows][NCols];

template <typename D, int NRows, int NCols>
using ColMajorMatrix = D[NCols][NRows];

template <int NRows, int NCols>
void RandInit(RowMajorMatrix<float, NRows, NCols> *matrix) {
  for (int i = 0; i < NRows; i++)
    for (int j = 0; j < NCols; j++) (*matrix)[i][j] = (float)rand() / RAND_MAX;
}

template <int NRows, int NCols>
void RandInit(ColMajorMatrix<float, NRows, NCols> *matrix) {
  for (int i = 0; i < NCols; i++)
    for (int j = 0; j < NRows; j++) (*matrix)[i][j] = (float)rand() / RAND_MAX;
}

void MatrixMulCPU(const ColMajorMatrix<float, A_HEIGHT, A_WIDTH> &a,
                  const RowMajorMatrix<float, B_HEIGHT, B_WIDTH> &b,
                  RowMajorMatrix<float, C_HEIGHT, C_WIDTH> *c) {
  for (int i = 0; i < C_HEIGHT; i++)
    for (int j = 0; j < C_WIDTH; j++) {
      float tot = 0;
      for (int k = 0; k < A_WIDTH; k++) tot += a[k][i] * b[k][j];
      (*c)[i][j] = tot;
    }
}

int MatrixMulTest(const RowMajorMatrix<float, C_HEIGHT, C_WIDTH> &c,
                  const RowMajorMatrix<float, C_HEIGHT, C_WIDTH> &c_ref) {
  int errors = 0;
  for (int i = 0; i < C_HEIGHT; i++)
    for (int j = 0; j < C_WIDTH; j++) {
      errors += std::fabs(c[i][j] - c_ref[i][j]) >= 1e3;
    }
  return errors;
}

__device__ ColMajorMatrix<float, A_HEIGHT, A_WIDTH> d_a;
__device__ RowMajorMatrix<float, B_HEIGHT, B_WIDTH> d_b;
__device__ RowMajorMatrix<float, C_HEIGHT, C_WIDTH> d_c;

ColMajorMatrix<float, A_HEIGHT, A_WIDTH> h_a;
RowMajorMatrix<float, B_HEIGHT, B_WIDTH> h_b;
RowMajorMatrix<float, C_HEIGHT, C_WIDTH> h_c;
RowMajorMatrix<float, C_HEIGHT, C_WIDTH> h_c_ref;

void CheckCUDAError(const char *msg);

__global__ void MatrixMulCUDA() {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = bx * BLOCK_SIZE + tx;
  int y = by * BLOCK_SIZE + ty;

  float tot = 0;
  // iterate A_WIDTH (same as B_HEIGHT) to calculate the product
  for (int k = 0; k < A_WIDTH; k++) {
    tot += d_a[k][x] * d_b[k][y];
  }

  // Store the product value of C matrix
  d_c[x][y] = tot;
}

__global__ void MatrixMulCUDASharedMemory() {
  // Define some shared memory for a sub block of matrices A an B
  __shared__ ColMajorMatrix<float, BLOCK_SIZE, BLOCK_SIZE> s_a;
  __shared__ RowMajorMatrix<float, BLOCK_SIZE, BLOCK_SIZE> s_b;

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Running sum of product of A and B matrices
  float tot = 0;

  // iterate through the number of sub matrices of A and B
  for (int i = 0; i < NUM_SUBS; i++) {
    int a_x = tx + BLOCK_SIZE * i;
    int a_y = ty + BLOCK_SIZE * bx;
    int b_x = tx + BLOCK_SIZE * i;
    int b_y = ty + BLOCK_SIZE * by;

    s_a[tx][ty] = d_a[a_x][a_y];
    s_b[tx][ty] = d_b[b_x][b_y];

    // Sync to ensure sub matrix is fully loaded
    __syncthreads();

    // Sum products of A and B sub matrices
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      tot += s_a[k][tx] * s_b[k][ty];
    }

    // Sync to prevent run ahead (blocks loading new SM values before others
    // have completed)
    __syncthreads();
  }

  // caluclate the indices of sub matrix C
  int c_x = tx + BLOCK_SIZE * bx;
  int c_y = ty + BLOCK_SIZE * by;

  // Store the product value of C matrix
  d_c[c_x][c_y] = tot;
}

#define LOG_RES(name)                           \
  do {                                          \
    printf("%s: %fms\n", name, ms);             \
    cudaMemcpyFromSymbol(h_c, d_c, mem_size_c); \
    CheckCUDAError("CUDA memcpy results");      \
    auto errors = MatrixMulTest(h_c, h_c_ref);  \
    if (errors) {                               \
      printf("%d ERRORS\n", errors);            \
    } else {                                    \
      printf("TEST PASSED\n");                  \
    }                                           \
  } while (0)

void GetProperty(int *max_blocks_per_mp, int *max_threads_per_mp, int *num_mp) {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  printf("Device: %s\n", props.name);
  *max_blocks_per_mp = props.maxBlocksPerMultiProcessor;
  *max_threads_per_mp = props.maxThreadsPerMultiProcessor;
  *num_mp = props.multiProcessorCount;

  printf("max_blocks_per_mp: %d\n", *max_blocks_per_mp);
  printf("max_threads_per_mp: %d\n", *max_threads_per_mp);
  printf("num_mp: %d\n", *num_mp);
}

int main(int argc, char **argv) {
  unsigned int mem_size_a, mem_size_b, mem_size_c;
  int max_active_blocks, max_blocks_per_mp, max_threads_per_mp, num_mp;
  float ms, occupancy;

  GetProperty(&max_blocks_per_mp, &max_threads_per_mp, &num_mp);

  mem_size_a = sizeof(float) * A_WIDTH * A_HEIGHT;
  mem_size_b = sizeof(float) * B_WIDTH * B_HEIGHT;
  mem_size_c = sizeof(float) * C_WIDTH * C_HEIGHT;

  RandInit<A_HEIGHT, A_WIDTH>(&h_a);
  RandInit<B_HEIGHT, B_WIDTH>(&h_b);

  // copy host memory to device
  cudaMemcpyToSymbol(d_a, h_a, mem_size_a);
  cudaMemcpyToSymbol(d_b, h_b, mem_size_b);
  CheckCUDAError("CUDA memcpy");

  MatrixMulCPU(h_a, h_b, &h_c_ref);

  // Setup execution parameters
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(C_HEIGHT / BLOCK_SIZE, C_WIDTH / BLOCK_SIZE);

  TIME("MatrixMulCUDA", ms, MatrixMulCUDA, grid_size, block_size);
  LOG_RES("MatrixMulCUDA");

  TIME("MatrixMulCUDASharedMemory", ms, MatrixMulCUDASharedMemory, grid_size,
       block_size);
  LOG_RES("MatrixMulCUDASharedMemory");

  // Compute the ocupancy
  occupancy = 1.0f * max_blocks_per_mp * (BLOCK_SIZE * BLOCK_SIZE) /
              (max_threads_per_mp * num_mp);
  printf("theoretical occupancy = %.6lf\n", occupancy);
  return 0;
}

void CheckCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
