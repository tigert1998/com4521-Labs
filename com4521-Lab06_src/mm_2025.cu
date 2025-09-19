#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/cstdint>
#include <string>
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
  __device__ __host__ void SetNoCheck(int x, int y, T value) {
    ptr[layout.Index(x, y)] = value;
  }
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
__global__ void MatrixMul(uint32_t block_size_k, uint32_t m, uint32_t n,
                          uint32_t k, M1 d_a, M2 d_b, M3 d_c) {
  static constexpr uint32_t block_size_m = 64, block_size_n = 64;

  extern __shared__ char shared[];
  T *s_a_mem[2];
  s_a_mem[0] = (T *)shared;
  s_a_mem[1] = (T *)shared + block_size_m * block_size_k;
  T *s_b_mem[2];
  s_b_mem[0] = (T *)shared + 2 * block_size_m * block_size_k;
  s_b_mem[1] = (T *)shared + 2 * block_size_m * block_size_k +
               block_size_n * block_size_k;

  MatrixWrapper<T, ColMajorLayout> s_a[2] = {
      MatrixWrapper<T, ColMajorLayout>{
          s_a_mem[0], ColMajorLayout(block_size_m, block_size_k)},
      MatrixWrapper<T, ColMajorLayout>{
          s_a_mem[1], ColMajorLayout(block_size_m, block_size_k)},
  };
  MatrixWrapper<T, RowMajorLayout> s_b[2] = {
      MatrixWrapper<T, RowMajorLayout>{
          s_b_mem[0], RowMajorLayout(block_size_k, block_size_n)},
      MatrixWrapper<T, RowMajorLayout>{
          s_b_mem[1], RowMajorLayout(block_size_k, block_size_n)},
  };

  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int num_warps = blockDim.x / warpSize;

#define STORE(i, s_a, s_b)                                       \
  do {                                                           \
    _Pragma("unroll") for (int j = warp_id * warpSize + lane_id; \
                           j < block_size_m * block_size_k;      \
                           j += num_warps * warpSize) {          \
      int x = j % block_size_m, y = j / block_size_m;            \
      s_a.SetNoCheck(x, y, d_a.Get(offset_m + x, offset_k + y)); \
    }                                                            \
    _Pragma("unroll") for (int j = warp_id * warpSize + lane_id; \
                           j < block_size_k * block_size_n;      \
                           j += num_warps * warpSize) {          \
      int x = j / block_size_n, y = j % block_size_n;            \
      s_b.SetNoCheck(x, y, d_b.Get(offset_k + x, offset_n + y)); \
    }                                                            \
  } while (0)

#define COMPUTE(s_a_mem, s_b_mem)                                      \
  do {                                                                 \
    for (int j = 0; j < block_size_k; j++) {                           \
      float4 l_vec = *(float4 *)(s_a_mem + j * block_size_m + tx * 4); \
      float4 r_vec = *(float4 *)(s_b_mem + j * block_size_n + ty * 4); \
      T *l = (T *)&l_vec;                                              \
      T *r = (T *)&r_vec;                                              \
      _Pragma("unroll") for (int idx = 0; idx < 16; idx++) {           \
        answers[idx] += l[idx % 4] * r[idx / 4];                       \
      }                                                                \
    }                                                                  \
  } while (0)

  // z-order
  // chinese doc: https://zhuanlan.zhihu.com/p/690052715
  int tx = (warp_id / 2) * 4 + (lane_id % 8) / 2;
  int ty = (warp_id % 2) * 8 + (lane_id / 8) * 2 + lane_id % 2;

  int num_subs = (k + block_size_k - 1) / block_size_k;

  T answers[16] = {(T)0};

  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;
  int offset_k = 0;

  STORE(0, s_a[0], s_b[0]);
  __syncthreads();
  for (int i = 1; i < num_subs; i++) {
    offset_k = block_size_k * i;
    STORE(0, s_a[i % 2], s_b[i % 2]);
    COMPUTE(s_a_mem[(i - 1) % 2], s_b_mem[(i - 1) % 2]);
    __syncthreads();
  }
  COMPUTE(s_a_mem[(num_subs - 1) % 2], s_b_mem[(num_subs - 1) % 2]);

#pragma unroll
  for (int idx = 0; idx < 16; idx++) {
    int x = offset_m + tx * 4 + idx % 4, y = offset_n + ty * 4 + idx / 4;
    d_c.Set(x, y, answers[idx]);
  }
}

class Im2colLayout {
 public:
  int channels, height, width, kernel_height, kernel_width, stride_height,
      stride_width, padding_height, padding_width;
  int output_height, output_width, matrix_height, matrix_width;
  __device__ __host__ Im2colLayout(int batch_size, int channels, int height,
                                   int width, int kernel_height,
                                   int kernel_width, int stride_height,
                                   int stride_width, int padding_height,
                                   int padding_width)
      : channels(channels),
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
  int matrix_height, matrix_width;

  __device__ __host__ WeightLayout(int output_channels, int input_channels,
                                   int kernel_height, int kernel_width) {
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
  int channels, height, width;
  int matrix_height, matrix_width;
  __device__ __host__ OutputLayout(int batch_size, int channels, int height,
                                   int width)
      : channels(channels), height(height), width(width) {
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
  int shared_bytes =
      (block_size_m + block_size_n) * block_size_k * sizeof(T) * 2;

  float ms;
  TIME("MatrixMul", ms, MatrixMul<T>, grid, block, shared_bytes, block_size_k,
       m, n, k, a, b, c);
  return ms;
}

template <typename T>
std::vector<T> Random(T *ptr, uint32_t size) {
  std::vector<T> v(size);
  for (int i = 0; i < size; i++) v[i] = rand() * 1.0 / RAND_MAX;
  cudaMemcpy(ptr, v.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  CheckCUDAError("cudaMemcpy cudaMemcpyHostToDevice");
  return v;
}

void CheckZOrder() {
  int mat[16][16];
  for (int thread_idx = 0; thread_idx < 32 * 8; thread_idx++) {
    int lane_id = thread_idx % 32;
    int warp_id = thread_idx / 32;
    int tx = (warp_id / 2) * 4 + (lane_id % 8) / 2;
    int ty = (warp_id % 2) * 8 + (lane_id / 8) * 2 + lane_id % 2;
    mat[tx][ty] = thread_idx;
  }
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) printf("%4d ", mat[i][j]);
    printf("\n");
  }
}

template <typename T>
void CheckCorrectness(int batch_size, int input_channels, int height, int width,
                      int kernel_height, int kernel_width, int stride_height,
                      int stride_width, int padding_height, int padding_width,
                      int output_channels, const std::vector<T> &input,
                      const std::vector<T> &weight, T *d_output) {
  int num_mismatches = 0;
  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;

  std::vector<T> output(batch_size * output_channels * output_height *
                        output_width);
  cudaMemcpy(output.data(), d_output, output.size() * sizeof(T),
             cudaMemcpyDeviceToHost);
  CheckCUDAError("cudaMemcpy cudaMemcpyDeviceToHost");

  for (int b = 0; b < batch_size; ++b) {
    for (int co = 0; co < output_channels; ++co) {
      for (int ho = 0; ho < output_height; ++ho) {
        for (int wo = 0; wo < output_width; ++wo) {
          T sum = 0;

          int h_start = ho * stride_height - padding_height;
          int w_start = wo * stride_width - padding_width;

          for (int ci = 0; ci < input_channels; ++ci) {
            for (int kh = 0; kh < kernel_height; ++kh) {
              for (int kw = 0; kw < kernel_width; ++kw) {
                int h = h_start + kh;
                int w = w_start + kw;

                if (h >= 0 && h < height && w >= 0 && w < width) {
                  uint32_t input_idx = b * input_channels * height * width +
                                       ci * height * width + h * width + w;

                  uint32_t weight_idx =
                      co * input_channels * kernel_height * kernel_width +
                      ci * kernel_height * kernel_width + kh * kernel_width +
                      kw;

                  sum += input[input_idx] * weight[weight_idx];
                }
              }
            }
          }

          uint32_t output_idx =
              b * output_channels * output_height * output_width +
              co * output_height * output_width + ho * output_width + wo;

          if (abs(output[output_idx] - sum) >= 1e-3) {
            printf("output[%d] mismatch: %.3f vs %.3f\n", output_idx,
                   output[output_idx], sum);
            num_mismatches += 1;
          }
        }
      }
    }
  }
  printf("#mismatches: %d\n", num_mismatches);
}

struct BenchParams {
  int batch_size;
  int input_channels;
  int height;
  int width;
  int kernel_height;
  int kernel_width;
  int stride_height;
  int stride_width;
  int padding_height;
  int padding_width;
  int output_channels;
};

void Benchmark(const BenchParams &params, int num_runs) {
  float *d_a, *d_b, *d_c;

  int output_height =
      (params.height + 2 * params.padding_height - params.kernel_height) /
          params.stride_height +
      1;
  int output_width =
      (params.width + 2 * params.padding_width - params.kernel_width) /
          params.stride_width +
      1;
  int m = params.batch_size * params.height * params.width;
  int n = params.output_channels;
  int k = params.input_channels * params.kernel_height * params.kernel_width;

  cudaMalloc(&d_a, sizeof(float) * m * k);
  cudaMalloc(&d_b, sizeof(float) * k * n);
  cudaMalloc(&d_c, sizeof(float) * m * n);
  auto h_a = Random(d_a, m * k);
  auto h_b = Random(d_b, k * n);

  float total = 0;
  for (int i = 0; i < num_runs; i++) {
    total += ImplicitGEMM<float>(
        params.batch_size, params.input_channels, params.height, params.width,
        params.kernel_height, params.kernel_width, params.stride_height,
        params.stride_width, params.padding_height, params.padding_width,
        params.output_channels, d_a, d_b, d_c);
  }
  float flops = 2.0f * (double)params.batch_size * (double)output_height *
                (double)output_width * (double)params.output_channels *
                (double)params.input_channels * (double)params.kernel_height *
                (double)params.kernel_width;
  float ms = total / num_runs;

  printf("[Workload] m = %d, n = %d, k = %d\n", m, n, k);
  printf("#runs = %d\n", num_runs);

  CheckCorrectness(params.batch_size, params.input_channels, params.height,
                   params.width, params.kernel_height, params.kernel_width,
                   params.stride_height, params.stride_width,
                   params.padding_height, params.padding_width,
                   params.output_channels, h_a, h_b, d_c);

  printf("ImplicitGEMM: %.3fms\n", ms);
  printf("%.3f GFLOPs\n", flops * 1e3 / (float)(1 << 30) / ms);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

int main(int argc, char **argv) {
  BenchParams suite[] = {
      BenchParams{64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 128},
      BenchParams{64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 128},
      BenchParams{64, 128, 7, 7, 3, 3, 1, 1, 1, 1, 32},
  };

  if (argc >= 2) {
    int p = std::stoi(argv[1]);
    Benchmark(suite[p], 5);
  } else {
    for (auto params : suite) {
      Benchmark(params, 100);
    }
  }
  return 0;
}