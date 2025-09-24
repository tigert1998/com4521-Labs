#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include "layouts.cuh"

template <typename T, uint32_t block_size_m, uint32_t block_size_n, typename M1,
          typename M2, typename M3>
__global__ void MatrixMulGeneralTensorCore(uint32_t m, uint32_t n, uint32_t k,
                                           M1 d_a, M2 d_b, M3 d_c, T *bias) {
  static constexpr uint32_t warp_size_m = 16;
  static constexpr uint32_t warp_size_n = 16;
  static_assert(block_size_m % warp_size_m == 0 &&
                block_size_n % warp_size_n == 0);
  static constexpr uint32_t warp_size_k = 16;
  static constexpr uint32_t num_warps_m = block_size_m / warp_size_m;
  static constexpr uint32_t num_warps_n = block_size_n / warp_size_n;
  static constexpr uint32_t num_warps = num_warps_m * num_warps_n;
  static constexpr uint32_t block_size_k = warp_size_k;

  extern __shared__ char shared[];
  T *s_a_mem = (T *)shared;
  T *s_b_mem = s_a_mem + block_size_m * block_size_k;
  float *s_c_mem = (float *)(s_b_mem + block_size_n * block_size_k);
  auto s_a = MatrixWrapper<T, ColMajorLayout>{
      s_a_mem, ColMajorLayout(block_size_m, block_size_k)};
  auto s_b = MatrixWrapper<T, RowMajorLayout>{
      s_b_mem, RowMajorLayout(block_size_k, block_size_n)};
  auto s_c = MatrixWrapper<float, ColMajorLayout>{
      s_c_mem, ColMajorLayout(block_size_m, block_size_n)};

  namespace wmma = nvcuda::wmma;

  wmma::fragment<wmma::matrix_a, warp_size_m, warp_size_n, warp_size_k, T,
                 wmma::col_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, warp_size_m, warp_size_n, warp_size_k, T,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, warp_size_m, warp_size_n, warp_size_k,
                 float>
      c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  int offset_x = blockIdx.x * block_size_m;
  int offset_y = blockIdx.y * block_size_n;

  int warp_id = threadIdx.x / warpSize;
  int warp_x = warp_id % num_warps_m;
  int warp_y = warp_id / num_warps_m;

  for (int p = 0; p < k; p += block_size_k) {
    __syncthreads();
#pragma unroll
    for (int i = threadIdx.x; i < block_size_m * block_size_k;
         i += num_warps * warpSize) {
      int x = i % block_size_m;
      int y = i / block_size_m;
      s_a.Set(x, y, d_a.Get(offset_x + x, p + y));
    }
#pragma unroll
    for (int i = threadIdx.x; i < block_size_n * block_size_k;
         i += num_warps * warpSize) {
      int x = i / block_size_n;
      int y = i % block_size_n;
      s_b.Set(x, y, d_b.Get(p + x, offset_y + y));
    }
    __syncthreads();
    wmma::load_matrix_sync(a_frag, s_a_mem + warp_x * warp_size_m,
                           block_size_m);
    wmma::load_matrix_sync(b_frag, s_b_mem + warp_y * warp_size_n,
                           block_size_n);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(
      s_c_mem + warp_y * warp_size_n * block_size_m + warp_x * warp_size_m,
      c_frag, block_size_m, wmma::mem_col_major);
  __syncthreads();

#pragma unroll
  for (int i = threadIdx.x; i < block_size_m * block_size_n;
       i += num_warps * warpSize) {
    int x = i % block_size_m;
    int y = i / block_size_m;
    d_c.Set(offset_x + x, offset_y + y, s_c.Get(x, y));
  }
}