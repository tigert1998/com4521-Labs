#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/cstdint>
#include <vector>

#include "common.cuh"

class ColMajorLayout {
 public:
  int height, width;
  __device__ __host__ ColMajorLayout(int height, int width)
      : height(height), width(width) {}
  __device__ __host__ int Index(int x, int y) { return x + y * height; }
};

class RowMajorLayout {
 public:
  int height, width;
  __device__ __host__ RowMajorLayout(int height, int width)
      : height(height), width(width) {}
  __device__ __host__ int Index(int x, int y) { return x * width + y; }
};

template <typename T, typename L>
class MatrixWrapper {
 public:
  T *ptr;
  L layout;
  __device__ __host__ MatrixWrapper(T *ptr, L layout)
      : ptr(ptr), layout(layout) {}
  __device__ __host__ void Set(int x, int y, T value) {
    int i = layout.Index(x, y);
    if (i >= 0) ptr[i] = value;
  }
  __device__ __host__ T Get(int x, int y) {
    int i = layout.Index(x, y);
    return i < 0 ? (T)0 : ptr[i];
  }
};

template <typename T, typename M1, typename M2, typename M3>
__global__ void MatrixMul(uint32_t block_size_m, uint32_t block_size_n,
                          uint32_t block_size_k, uint32_t m, uint32_t n,
                          uint32_t k, M1 d_a, M2 d_b, M3 d_c) {
  extern __shared__ char shared[];
  T *s_a_mem = (T *)shared;
  T *s_b_mem = (T *)shared + block_size_m * block_size_k;

  auto s_a = MatrixWrapper<T, ColMajorLayout>{
      s_a_mem, ColMajorLayout(block_size_m, block_size_k)};
  auto s_b = MatrixWrapper<T, RowMajorLayout>{
      s_b_mem, RowMajorLayout(block_size_k, block_size_n)};

  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int num_warps = blockDim.x / warpSize;
  int tx = (warp_id / 2) * 4 + (lane_id % 8) / 2;
  int ty = (warp_id % 2) * 8 + (lane_id / 8) * 2 + lane_id % 2;

  int num_subs = (k + block_size_k - 1) / block_size_k;
  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;

  T answers[16] = {(T)0};
  for (int i = 0; i < num_subs; i++) {
    int offset_k = block_size_k * i;

    // load into shared memory
#pragma unroll
    for (int j = warp_id * warpSize + lane_id; j < block_size_m * block_size_k;
         j += num_warps * warpSize) {
      int x = j % block_size_m, y = j / block_size_m;
      s_a.Set(x, y, d_a.Get(offset_m + x, offset_k + y));
    }
#pragma unroll
    for (int j = warp_id * warpSize + lane_id; j < block_size_k * block_size_n;
         j += num_warps * warpSize) {
      int x = j / block_size_n, y = j % block_size_n;
      s_b.Set(x, y, d_b.Get(offset_k + x, offset_n + y));
    }
    __syncthreads();

    // z-order
    // chinese doc: https://zhuanlan.zhihu.com/p/690052715
    for (int j = 0; j < block_size_k; j++) {
      float4 l_vec = *(float4 *)(s_a_mem + j * block_size_m + tx * 4);
      float4 r_vec = *(float4 *)(s_b_mem + j * block_size_n + ty * 4);
      T *l = (T *)&l_vec;
      T *r = (T *)&r_vec;
#pragma unroll
      for (int idx = 0; idx < 16; idx++) {
        answers[idx] += l[idx % 4] * r[idx / 4];
      }
    }
  }

#pragma unroll
  for (int idx = 0; idx < 16; idx++) {
    int x = offset_m + tx * 4 + idx % 4, y = offset_n + ty * 4 + idx / 4;
    d_c.Set(x, y, answers[idx]);
  }
}

class Im2colLayout {
 public:
  int batch_size, channels, height, width, kernel_height, kernel_width,
      stride_height, stride_width, padding_height, padding_width;
  int output_height, output_width, matrix_height, matrix_width;
  __device__ __host__ Im2colLayout(int batch_size, int channels, int height,
                                   int width, int kernel_height,
                                   int kernel_width, int stride_height,
                                   int stride_width, int padding_height,
                                   int padding_width)
      : batch_size(batch_size),
        channels(channels),
        height(height),
        width(width),
        kernel_height(kernel_height),
        kernel_width(kernel_width),
        stride_height(stride_height),
        stride_width(stride_width),
        padding_height(padding_height),
        padding_width(padding_width) {
    output_height =
        (height + 2 * padding_height - kernel_height) / stride_height + 1;
    output_width =
        (width + 2 * padding_width - kernel_width) / stride_width + 1;
    matrix_height = batch_size * output_height * output_width;
    matrix_width = channels * kernel_height * kernel_width;
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;

    int batch_idx = x / (output_height * output_width);
    int output_x = x / output_width % output_height;
    int output_y = x % output_width;
    int channel_idx = y / (kernel_height * kernel_width);
    int kernel_x = y / kernel_width % kernel_height;
    int kernel_y = y % kernel_width;

    int input_x = output_x * stride_height - padding_height + kernel_x;
    int input_y = output_y * stride_width - padding_width + kernel_y;

    if (input_x < 0 || input_x >= height || input_y < 0 || input_y >= width)
      return -1;

    return batch_idx * channels * height * width +
           channel_idx * height * width + input_x * width + input_y;
  }
};

class WeightLayout {
 public:
  int output_channels, input_channels, kernel_height, kernel_width;
  int matrix_height, matrix_width;

  __device__ __host__ WeightLayout(int output_channels, int input_channels,
                                   int kernel_height, int kernel_width)
      : output_channels(output_channels),
        input_channels(input_channels),
        kernel_height(kernel_height),
        kernel_width(kernel_width) {
    matrix_height = input_channels * kernel_height * kernel_width;
    matrix_width = output_channels;
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;
    return y * matrix_height + x;
  }
};

class OutputLayout {
 public:
  int batch_size, channels, height, width;
  int matrix_height, matrix_width;
  __device__ __host__ OutputLayout(int batch_size, int channels, int height,
                                   int width)
      : batch_size(batch_size),
        channels(channels),
        height(height),
        width(width) {
    matrix_height = batch_size * height * width;
    matrix_width = channels;
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;

    int channel_idx = y;
    int batch_idx = x / (height * width);
    int height_width_idx = x % (height * width);
    return batch_idx * channels * height * width +
           channel_idx * height * width + height_width_idx;
  }
};

template <typename T>
float ImplicitGEMM(int batch_size, int input_channels, int height, int width,
                   int kernel_height, int kernel_width, int stride_height,
                   int stride_width, int padding_height, int padding_width,
                   int output_channels, T *input, T *weight, T *output) {
  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;

  Im2colLayout im2col_layout(batch_size, input_channels, height, width,
                             kernel_height, kernel_width, stride_height,
                             stride_width, padding_height, padding_width);
  MatrixWrapper<T, Im2colLayout> a(input, im2col_layout);
  WeightLayout weight_layout(output_channels, input_channels, kernel_height,
                             kernel_width);
  MatrixWrapper<T, WeightLayout> b(weight, weight_layout);
  OutputLayout col2im_layout(batch_size, output_channels, output_height,
                             output_width);
  MatrixWrapper<T, OutputLayout> c(output, col2im_layout);

  uint32_t m = batch_size * output_height * output_width;
  uint32_t n = output_channels;
  uint32_t k = input_channels * kernel_height * kernel_width;
  uint32_t block_size_m = 64, block_size_n = 64, block_size_k = 4;
  uint32_t num_warps = 8;
  dim3 block = {32 * num_warps, 1, 1};
  dim3 grid = {(m + block_size_m - 1) / block_size_m,
               (n + block_size_n - 1) / block_size_n, 1};
  int shared_bytes = (block_size_m + block_size_n) * block_size_k * sizeof(T);

  float ms;
  TIME("MatrixMul", ms, MatrixMul<T>, grid, block, shared_bytes, block_size_m,
       block_size_n, block_size_k, m, n, k, a, b, c);
  return ms;
}

template <typename T>
std::vector<T> Random(T *ptr, uint32_t size) {
  std::vector<T> v(size);
  for (int i = 0; i < size; i++) v[i] = rand() * 1.0 / RAND_MAX;
  cudaMemcpy(ptr, v.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  CheckCUDAError("cudaMemcpy");
  return v;
}

int main() {
  int batch_size = 64;
  int input_channels = 32;
  int height = 14;
  int width = 14, int kernel_height = 3;
  int kernel_width = 3;
  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 1;
  int padding_width = 1;
  int output_channels = 128;
  float *d_a, *d_b, *d_c;

  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;
  int m = batch_size * height * width;
  int n = output_channels;
  int k = input_channels * kernel_height * kernel_width;

  cudaMalloc(&d_a, sizeof(float) * m * k);
  cudaMalloc(&d_b, sizeof(float) * k * n);
  cudaMalloc(&d_c, sizeof(float) * m * n);
  auto h_a = Random(d_a, m * k);
  auto h_b = Random(d_b, k * n);

  float total = 0;
  int num_runs = 100;
  for (int i = 0; i < num_runs; i++) {
    total += ImplicitGEMM<float>(batch_size, input_channels, height, width,
                                 kernel_height, kernel_width, stride_height,
                                 stride_width, padding_height, padding_width,
                                 output_channels, d_a, d_b, d_c);
  }
  float flops = batch_size * output_height * output_width * output_channels *
                input_channels * kernel_height * kernel_width * 2;
  float ms = total / num_runs;
  printf("ImplicitGEMM: %.3fms\n", ms);
  printf("%.3f GFLOPs\n", flops * 1e3 / (1 << 30) / ms);
}