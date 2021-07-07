#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

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

void CheckCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template <typename D, int N>
class Array {
 private:
  D data_[N]{};

 public:
  __host__ __device__ const D *data() const { return data_; }
  __host__ __device__ D *data() { return data_; }
};

template <typename D>
class Array<D, 0> {
 private:
  D *ptr_ = nullptr;

 public:
  __host__ __device__ Array(D *ptr) : ptr_(ptr) {}
  __host__ __device__ const D *data() const { return ptr_; }
  __host__ __device__ D *data() { return ptr_; }
};

template <int R, int C>
class RowMajor {
 public:
  __host__ __device__ inline static int Index(int x, int y) {
    return x * C + y;
  }
};

template <int R, int C, int B>
class BlockedRowMajor {
  static_assert((B & (B - 1)) == 0, "");

 public:
  __host__ __device__ inline static int Index(int x, int y) {
    int bx = x / B;
    int by = y / B;
    int tx = x & (B - 1);
    int ty = y & (B - 1);
    return (bx * (C / B) + by) * B * B + (tx * B + ty);
  }
};

template <int R, int C>
class ColMajor {
 public:
  __host__ __device__ inline static int Index(int x, int y) {
    return x + y * R;
  }
};

template <int R, int C, int B>
class BlockedColMajor {
  static_assert((B & (B - 1)) == 0, "");

 public:
  __host__ __device__ inline static int Index(int x, int y) {
    int bx = x / B;
    int by = y / B;
    int tx = x & (B - 1);
    int ty = y & (B - 1);
    return (bx + by * (R / B)) * B * B + (tx + ty * B);
  }
};

template <typename D, int R, int C, int N, typename Layout>
class Matrix : public Array<D, N> {
 public:
  static constexpr int kNumRows = R;
  static constexpr int kNumCols = C;

  Matrix() = default;

  __host__ __device__ Matrix(D *ptr) : Array<D, N>(ptr) {}

  __host__ __device__ inline D &At(int x, int y) {
    return Array<D, N>::data()[Layout::Index(x, y)];
  }
  __host__ __device__ inline D At(int x, int y) const {
    return Array<D, N>::data()[Layout::Index(x, y)];
  }

  template <typename M>
  bool Equal(const M &m) {
    if (kNumRows == M::kNumRows && kNumCols == M::kNumCols) {
      for (int i = 0; i < kNumRows; i++)
        for (int j = 0; j < kNumCols; j++) {
          if (std::fabs(At(i, j) - m.At(i, j)) >= 1e-3) {
            return false;
          }
        }
      return true;
    }
    return false;
  }

  template <typename M1, typename M2>
  void Mul(const M1 &b, M2 *c) {
    static_assert(kNumCols == M1::kNumRows && kNumRows == M2::kNumRows &&
                      M1::kNumCols == M2::kNumCols,
                  "invalid multiply shape");
    for (int i = 0; i < M2::kNumRows; i++)
      for (int j = 0; j < M2::kNumCols; j++) {
        float tot = 0;
        for (int k = 0; k < kNumCols; k++) tot += At(i, k) * b.At(k, j);
        c->At(i, j) = tot;
      }
  }

  void Init() {
    for (int i = 0; i < kNumRows; i++)
      for (int j = 0; j < kNumCols; j++) {
        At(i, j) = (float)rand() / RAND_MAX;
      }
  }

  template <int M>
  void ToDevice(Array<D, M> *symbol) {
    cudaMemcpy(symbol->data(), this->data(), N * sizeof(D),
               cudaMemcpyHostToDevice);
    CheckCUDAError("Matrix::ToDevice");
  }

  template <int M>
  void FromDevice(Array<D, M> *symbol) {
    cudaMemcpy(this->data(), symbol->data(), N * sizeof(D),
               cudaMemcpyDeviceToHost);
    CheckCUDAError("Matrix::FromDevice");
  }
};

template <typename D, int R, int C>
class RowMajorMatrix : public Matrix<D, R, C, R * C, RowMajor<R, C>> {};
template <typename D, int R, int C>
class BlockedRowMajorMatrix
    : public Matrix<D, R, C, R * C, BlockedRowMajor<R, C, 8>> {};

template <typename D, int R, int C>
class RowMajorMatrixWrapper : public Matrix<D, R, C, 0, RowMajor<R, C>> {
 public:
  __host__ __device__ RowMajorMatrixWrapper(D *ptr)
      : Matrix<D, R, C, 0, RowMajor<R, C>>(ptr) {}
};

template <typename D, int R, int C>
class ColMajorMatrix : public Matrix<D, R, C, R * C, ColMajor<R, C>> {};
template <typename D, int R, int C>
class BlockedColMajorMatrix
    : public Matrix<D, R, C, R * C, BlockedColMajor<R, C, 8>> {};

template <typename D, int R, int C>
class ColMajorMatrixWrapper : public Matrix<D, R, C, 0, ColMajor<R, C>> {
 public:
  __host__ __device__ ColMajorMatrixWrapper(D *ptr)
      : Matrix<D, R, C, 0, ColMajor<R, C>>(ptr) {}
};

BlockedColMajorMatrix<float, A_HEIGHT, A_WIDTH> *d_a;
BlockedRowMajorMatrix<float, B_HEIGHT, B_WIDTH> *d_b;
BlockedRowMajorMatrix<float, C_HEIGHT, C_WIDTH> *d_c;

BlockedColMajorMatrix<float, A_HEIGHT, A_WIDTH> h_a;
BlockedRowMajorMatrix<float, B_HEIGHT, B_WIDTH> h_b;
BlockedRowMajorMatrix<float, C_HEIGHT, C_WIDTH> h_c;
RowMajorMatrix<float, C_HEIGHT, C_WIDTH> h_c_ref;

template <typename M1, typename M2, typename M3>
__global__ void MatrixMulCUDA(M1 *d_a, M2 *d_b, M3 *d_c) {
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
    tot += d_a->At(x, k) * d_b->At(k, y);
  }

  // Store the product value of C matrix
  d_c->At(x, y) = tot;
}

template <typename M1, typename M2, typename M3>
__global__ void MatrixMulCUDASharedMemory(M1 *d_a, M2 *d_b, M3 *d_c) {
  // Define some shared memory for a sub block of matrices A an B
  __shared__ float s_a_mem[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float s_b_mem[BLOCK_SIZE * BLOCK_SIZE];

  ColMajorMatrixWrapper<float, BLOCK_SIZE, BLOCK_SIZE> s_a(s_a_mem);
  RowMajorMatrixWrapper<float, BLOCK_SIZE, BLOCK_SIZE> s_b(s_b_mem);

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Running sum of product of A and B matrices
  float tot = 0;

  // iterate through the number of sub matrices of A and B
  for (int i = 0; i < NUM_SUBS; i++) {
    int a_x = tx + BLOCK_SIZE * bx;
    int a_y = ty + BLOCK_SIZE * i;
    int b_x = tx + BLOCK_SIZE * i;
    int b_y = ty + BLOCK_SIZE * by;

    s_a.At(tx, ty) = d_a->At(a_x, a_y);
    s_b.At(tx, ty) = d_b->At(b_x, b_y);

    // Sync to ensure sub matrix is fully loaded
    __syncthreads();

    // Sum products of A and B sub matrices
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      tot += s_a.At(tx, k) * s_b.At(k, ty);
    }

    // Sync to prevent run ahead (blocks loading new SM values before others
    // have completed)
    __syncthreads();
  }

  // caluclate the indices of sub matrix C
  int c_x = tx + BLOCK_SIZE * bx;
  int c_y = ty + BLOCK_SIZE * by;

  // Store the product value of C matrix
  d_c->At(c_x, c_y) = tot;
}

#define LOG_RES(name)                               \
  do {                                              \
    printf("%s: %fms\n", name, ms);                 \
    h_c.FromDevice(d_c);                            \
    puts(h_c.Equal(h_c_ref) ? "PASSED" : "FAILED"); \
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
  int max_active_blocks, max_blocks_per_mp, max_threads_per_mp, num_mp;
  float ms, occupancy;

  GetProperty(&max_blocks_per_mp, &max_threads_per_mp, &num_mp);

  cudaMalloc(&d_a, sizeof(decltype(*d_a)));
  cudaMalloc(&d_b, sizeof(decltype(*d_b)));
  cudaMalloc(&d_c, sizeof(decltype(*d_c)));

  h_a.Init();
  h_b.Init();

  // copy host memory to device
  h_a.ToDevice(d_a);
  h_b.ToDevice(d_b);

  h_a.Mul(h_b, &h_c_ref);

  // Setup execution parameters
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(C_HEIGHT / BLOCK_SIZE, C_WIDTH / BLOCK_SIZE);

  TIME("MatrixMulCUDA", ms, MatrixMulCUDA, grid_size, block_size, d_a, d_b,
       d_c);
  LOG_RES("MatrixMulCUDA");

  TIME("MatrixMulCUDASharedMemory", ms, MatrixMulCUDASharedMemory, grid_size,
       block_size, d_a, d_b, d_c);
  LOG_RES("MatrixMulCUDASharedMemory");

  // Compute the ocupancy
  occupancy = 1.0f * max_blocks_per_mp * (BLOCK_SIZE * BLOCK_SIZE) /
              (max_threads_per_mp * num_mp);
  printf("theoretical occupancy = %.6lf\n", occupancy);
  return 0;
}
