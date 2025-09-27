#pragma once

#include "mm_cuda_core.cuh"
#include "mm_tensor_core.cuh"

template <typename T, bool reduce, typename M1, typename M2, typename M3>
void MatrixMul(M1 a, M2 b, M3 c, T *bias, cudaStream_t stream) {
  uint32_t m = c.layout.matrix_height;
  uint32_t n = c.layout.matrix_width;
  uint32_t k = a.layout.matrix_width;

  if constexpr (std::is_same_v<T, __half>) {
    // half
    static constexpr uint32_t block_size_m = 32, block_size_n = 32,
                              block_size_k = 16;
    constexpr int num_warps = (block_size_m / 16) * (block_size_n / 16);
    dim3 block = {32 * num_warps, 1, 1};
    dim3 grid = {(m + block_size_m - 1) / block_size_m,
                 (n + block_size_n - 1) / block_size_n, 1};
    int shared_bytes =
        (block_size_m + block_size_n) * block_size_k * sizeof(T) +
        block_size_m * block_size_n * sizeof(float);

    MatrixMulGeneralTensorCore<T, block_size_m, block_size_n>
        <<<grid, block, shared_bytes, stream>>>(m, n, k, a, b, c, bias);
  } else {
    // float
    if (m * n <= 1024) {
      static constexpr int block_size = 2;
      static constexpr int block_size_k = 512;
      dim3 block = {block_size_k, 1, 1};
      dim3 grid = {16, 16, 1};
      int shared_bytes = 2 * block_size * block_size_k * sizeof(T);
      MatrixMulTallAndSkinny<T, block_size, block_size_k>
          <<<grid, block, shared_bytes, stream>>>(m, n, k, a, b, c, bias);
    } else if (m * n <= 65536) {
      static constexpr uint32_t block_size_m = 32, block_size_n = 32,
                                block_size_k = 64;
      constexpr int num_warps = 8;
      dim3 block = {32 * num_warps, 1, 1};
      dim3 grid = {(m + block_size_m - 1) / block_size_m,
                   (n + block_size_n - 1) / block_size_n, 1};
      int shared_bytes =
          (block_size_m + block_size_n) * block_size_k * sizeof(T);

      MatrixMulGeneral<T, block_size_m, block_size_n, block_size_k, reduce>
          <<<grid, block, shared_bytes, stream>>>(m, n, k, a, b, c, bias);
    } else {
      static constexpr uint32_t block_size_m = 64, block_size_n = 64,
                                block_size_k = 32;
      constexpr int num_warps = 8;
      dim3 block = {32 * num_warps, 1, 1};
      dim3 grid = {(m + block_size_m - 1) / block_size_m,
                   (n + block_size_n - 1) / block_size_n, 1};
      int shared_bytes =
          (block_size_m + block_size_n) * block_size_k * sizeof(T);

      MatrixMulGeneral<T, block_size_m, block_size_n, block_size_k, reduce>
          <<<grid, block, shared_bytes, stream>>>(m, n, k, a, b, c, bias);
    }
  }
}
