#pragma once

#include "layouts.cuh"

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, typename M1, typename M2>
__forceinline__ __device__ void StoreIntoSMEM(
    int i, M1 d_a, M2 d_b, MatrixWrapper<T, ColMajorLayout> s_a,
    MatrixWrapper<T, RowMajorLayout> s_b) {
  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;

#pragma unroll
  for (int j = threadIdx.x; j < max(block_size_m, block_size_n) * block_size_k;
       j += blockDim.x) {
    if (j < block_size_m * block_size_k) {
      int x = j % block_size_m, y = j / block_size_m;
      s_a.SetNoCheck(x, y, d_a.Get(offset_m + x, (block_size_k * (i)) + y));
    }
    if (j < block_size_n * block_size_k) {
      int x = j / block_size_n, y = j % block_size_n;
      s_b.SetNoCheck(x, y, d_b.Get((block_size_k * (i)) + x, offset_n + y));
    }
  }
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t thread_size_m, uint32_t thread_size_n>
__forceinline__ __device__ void LoadFromSMEM(T *l, T *r, T *s_a_mem, T *s_b_mem,
                                             int tx, int ty, int j) {
#pragma unroll
  for (int k = 0; k < thread_size_m; k++)
    l[k] = *(s_a_mem + (j)*block_size_m + tx * thread_size_m + k);
#pragma unroll
  for (int k = 0; k < thread_size_n; k++)
    r[k] = *(s_b_mem + (j)*block_size_n + ty * thread_size_n + k);
}

template <typename T, uint32_t thread_size_m, uint32_t thread_size_n>
__forceinline__ __device__ void ComputeRegisters(T *accum, T *l, T *r) {
#pragma unroll
  for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
    accum[idx] += l[idx % thread_size_m] * r[idx / thread_size_m];
  }
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, uint32_t thread_size_m, uint32_t thread_size_n>
__forceinline__ __device__ void Compute(T *accum, int tx, int ty, T *s_a_mem,
                                        T *s_b_mem) {
  T l[thread_size_m], r[thread_size_n];
  for (int j = 0; j < block_size_k; j++) {
    LoadFromSMEM<T, block_size_m, block_size_n, thread_size_m, thread_size_n>(
        l, r, s_a_mem, s_b_mem, tx, ty, j);
    ComputeRegisters<T, thread_size_m, thread_size_n>(accum, l, r);
  }
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, bool reduce, typename M1, typename M2,
          typename M3>
__global__ void MatrixMulGeneral(uint32_t m, uint32_t n, uint32_t k, M1 d_a,
                                 M2 d_b, M3 d_c, T *bias) {
  static constexpr uint32_t thread_size_m = block_size_m / 16;
  static constexpr uint32_t thread_size_n = block_size_n / 16;

  extern __shared__ char shared[];

  static constexpr uint32_t num_warps = 8;

  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  // z-order
  // chinese doc: https://zhuanlan.zhihu.com/p/690052715
  int tx = (warp_id / 2) * 4 + (lane_id % 8) / 2;
  int ty = (warp_id % 2) * 8 + (lane_id / 8) * 2 + lane_id % 2;

  int num_subs = (k + block_size_k - 1) / block_size_k;

  T accum[thread_size_m * thread_size_n] = {(T)0};

  T *s_a_mem = (T *)shared;
  T *s_b_mem = (T *)shared + block_size_m * block_size_k;
  auto s_a = MatrixWrapper<T, ColMajorLayout>{
      s_a_mem, ColMajorLayout(block_size_m, block_size_k)};
  auto s_b = MatrixWrapper<T, RowMajorLayout>{
      s_b_mem, RowMajorLayout(block_size_k, block_size_n)};
  for (int i = 0; i < num_subs; i++) {
    __syncthreads();
    StoreIntoSMEM<T, block_size_m, block_size_n, block_size_k>(i, d_a, d_b, s_a,
                                                               s_b);
    __syncthreads();
    Compute<T, block_size_m, block_size_n, block_size_k, thread_size_m,
            thread_size_n>(accum, tx, ty, s_a_mem, s_b_mem);
  }

  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;
  if constexpr (reduce) {
#pragma unroll
    for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
      int x = offset_m + tx * thread_size_m + idx % thread_size_m;
      int y = offset_n + ty * thread_size_n + idx / thread_size_m;
      d_c.Set(x, y, (T)0);
    }

    __syncthreads();

#pragma unroll
    for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
      int x = offset_m + tx * thread_size_m + idx % thread_size_m;
      int y = offset_n + ty * thread_size_n + idx / thread_size_m;
      d_c.Add(x, y, accum[idx] + (bias != nullptr ? bias[y] : 0));
    }
  } else {
#pragma unroll
    for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
      int x = offset_m + tx * thread_size_m + idx % thread_size_m;
      int y = offset_n + ty * thread_size_n + idx / thread_size_m;
      d_c.Set(x, y, accum[idx] + (bias != nullptr ? bias[y] : 0));
    }
  }
}

template <typename T, uint32_t block_size, uint32_t block_size_k, typename M1,
          typename M2, typename M3>
__global__ void MatrixMulTallAndSkinny(uint32_t m, uint32_t n, uint32_t k,
                                       M1 d_a, M2 d_b, M3 d_c, T *bias) {
  extern __shared__ char shared[];
  T *s_a_mem = (T *)shared;
  T *s_b_mem = (T *)shared + block_size * block_size_k;
  auto s_a = MatrixWrapper<T, ColMajorLayout>{
      s_a_mem, ColMajorLayout(block_size, block_size_k)};
  auto s_b = MatrixWrapper<T, RowMajorLayout>{
      s_b_mem, RowMajorLayout(block_size_k, block_size)};

  for (int i = blockIdx.x; i < (m + block_size - 1) / block_size;
       i += gridDim.x) {
    for (int j = blockIdx.y; j < (n + block_size - 1) / block_size;
         j += gridDim.y) {
      T regs[block_size * block_size] = {(T)0};

      for (int idx = threadIdx.x; idx < k; idx += block_size_k) {
        __syncthreads();
        for (int p = 0; p < block_size; p++) {
          s_a.Set(p, threadIdx.x, d_a.Get(i * block_size + p, idx));
          s_b.Set(threadIdx.x, p, d_b.Get(idx, j * block_size + p));
        }
        __syncthreads();

        for (int p = 0; p < block_size * block_size; p++) {
          regs[p] += s_a.Get(p % block_size, threadIdx.x) *
                     s_b.Get(threadIdx.x, p / block_size);
        }
      }

      if (threadIdx.x == 0) {
        for (int p = 0; p < block_size * block_size; p++)
          d_c.Set(i * block_size + p % block_size,
                  j * block_size + p / block_size,
                  bias != nullptr ? bias[j * block_size + p / block_size] : 0);
      }
      __syncthreads();
      for (int p = 0; p < block_size * block_size; p++)
        d_c.Add(i * block_size + p % block_size,
                j * block_size + p / block_size, regs[p]);
    }
  }
}
