#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/cstdint>
#include <memory>
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

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, bool enable_smem_prefetch, typename M1,
          typename M2, typename M3>
__global__ void MatrixMul(uint32_t m, uint32_t n, uint32_t k, M1 d_a, M2 d_b,
                          M3 d_c) {
  extern __shared__ char shared[];

  static constexpr uint32_t num_warps = 8;
  static constexpr uint32_t thread_size_m = block_size_m / 16;
  static constexpr uint32_t thread_size_n = block_size_n / 16;

  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

#define STORE_SMEM(i, s_a, s_b)                                              \
  do {                                                                       \
    _Pragma("unroll") for (int j = warp_id * warpSize + lane_id;             \
                           j < block_size_m * block_size_k;                  \
                           j += num_warps * warpSize) {                      \
      int x = j % block_size_m, y = j / block_size_m;                        \
      s_a.SetNoCheck(x, y, d_a.Get(offset_m + x, (block_size_k * (i)) + y)); \
    }                                                                        \
    _Pragma("unroll") for (int j = warp_id * warpSize + lane_id;             \
                           j < block_size_n * block_size_k;                  \
                           j += num_warps * warpSize) {                      \
      int x = j / block_size_n, y = j % block_size_n;                        \
      s_b.SetNoCheck(x, y, d_b.Get((block_size_k * (i)) + x, offset_n + y)); \
    }                                                                        \
  } while (0)

#define LOAD_SMEM(j, l, r, s_a_mem, s_b_mem)                         \
  do {                                                               \
    _Pragma("unroll") for (int k = 0; k < thread_size_m; k++) l[k] = \
        *(s_a_mem + (j) * block_size_m + tx * thread_size_m + k);    \
    _Pragma("unroll") for (int k = 0; k < thread_size_n; k++) r[k] = \
        *(s_b_mem + (j) * block_size_n + ty * thread_size_n + k);    \
  } while (0)

#define COMPUTE_REGS(l, r)                                                   \
  do {                                                                       \
    _Pragma("unroll") for (int idx = 0; idx < thread_size_m * thread_size_n; \
                           idx++) {                                          \
      accum[idx] += l[idx % thread_size_m] * r[idx / thread_size_m];         \
    }                                                                        \
  } while (0)

#define COMPUTE_PREFETCH(s_a_mem, s_b_mem)                                \
  do {                                                                    \
    T l[2][thread_size_m], r[2][thread_size_n];                           \
    LOAD_SMEM(0, l[0], r[0], s_a_mem, s_b_mem);                           \
    for (int j = 0; j < block_size_k - 1; j++) {                          \
      LOAD_SMEM(j + 1, l[(j + 1) % 2], r[(j + 1) % 2], s_a_mem, s_b_mem); \
      COMPUTE_REGS(l[j % 2], r[j % 2]);                                   \
    }                                                                     \
    COMPUTE_REGS(l[(block_size_k - 1) % 2], r[(block_size_k - 1) % 2]);   \
  } while (0)

#define COMPUTE_NO_PREFETCH(s_a_mem, s_b_mem) \
  do {                                        \
    T l[thread_size_m], r[thread_size_n];     \
    for (int j = 0; j < block_size_k; j++) {  \
      LOAD_SMEM(j, l, r, s_a_mem, s_b_mem);   \
      COMPUTE_REGS(l, r);                     \
    }                                         \
  } while (0)

  // z-order
  // chinese doc: https://zhuanlan.zhihu.com/p/690052715
  int tx = (warp_id / 2) * 4 + (lane_id % 8) / 2;
  int ty = (warp_id % 2) * 8 + (lane_id / 8) * 2 + lane_id % 2;

  int num_subs = (k + block_size_k - 1) / block_size_k;

  T accum[thread_size_m * thread_size_n] = {(T)0};

  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;

  if constexpr (enable_smem_prefetch) {
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
    // TODO
  } else {
    T *s_a_mem = (T *)shared;
    T *s_b_mem = (T *)shared + block_size_m * block_size_k;
    auto s_a = MatrixWrapper<T, ColMajorLayout>{
        s_a_mem, ColMajorLayout(block_size_m, block_size_k)};
    auto s_b = MatrixWrapper<T, RowMajorLayout>{
        s_b_mem, RowMajorLayout(block_size_k, block_size_n)};
    for (int i = 0; i < num_subs; i++) {
      __syncthreads();
      STORE_SMEM(i, s_a, s_b);
      __syncthreads();
      COMPUTE_PREFETCH(s_a_mem, s_b_mem);
    }
  }

#pragma unroll
  for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
    int x = offset_m + tx * thread_size_m + idx % thread_size_m;
    int y = offset_n + ty * thread_size_n + idx / thread_size_m;
    d_c.Set(x, y, accum[idx]);
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
float GEMM(uint32_t m, uint32_t n, uint32_t k, T *input, T *weight, T *output) {
  MatrixWrapper<T, ColMajorLayout> a(input, ColMajorLayout(m, k));
  MatrixWrapper<T, RowMajorLayout> b(weight, RowMajorLayout(k, n));
  MatrixWrapper<T, ColMajorLayout> c(output, ColMajorLayout(m, n));

  static constexpr uint32_t block_size_m = 64, block_size_n = 64,
                            block_size_k = 32;
  static constexpr bool enable_smem_prefetch = false;

  uint32_t num_warps = 8;
  dim3 block = {32 * num_warps, 1, 1};
  dim3 grid = {(m + block_size_m - 1) / block_size_m,
               (n + block_size_n - 1) / block_size_n, 1};
  int shared_bytes = (block_size_m + block_size_n) * block_size_k * sizeof(T) *
                     (enable_smem_prefetch ? 2 : 1);

  CudaTimer timer;
  MatrixMul<T, block_size_m, block_size_n, block_size_k, enable_smem_prefetch>
      <<<grid, block, shared_bytes>>>(m, n, k, a, b, c);
  CheckCUDAError("MatrixMul");
  return timer.End();
}

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
  OutputLayout output_layout(batch_size, output_channels, output_height,
                             output_width);
  MatrixWrapper<T, OutputLayout> c(output, output_layout);

  uint32_t m = batch_size * output_height * output_width;
  uint32_t n = output_channels;
  uint32_t k = input_channels * kernel_height * kernel_width;
  static constexpr uint32_t block_size_m = 64, block_size_n = 64,
                            block_size_k = 8;
  static constexpr bool enable_smem_prefetch = false;
  uint32_t num_warps = 8;
  dim3 block = {32 * num_warps, 1, 1};
  dim3 grid = {(m + block_size_m - 1) / block_size_m,
               (n + block_size_n - 1) / block_size_n, 1};
  int shared_bytes = (block_size_m + block_size_n) * block_size_k * sizeof(T) *
                     (enable_smem_prefetch ? 2 : 1);

  CudaTimer timer;
  MatrixMul<T, block_size_m, block_size_n, block_size_k, enable_smem_prefetch>
      <<<grid, block, shared_bytes>>>(m, n, k, a, b, c);
  CheckCUDAError("MatrixMul");
  return timer.End();
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
            if (num_mismatches < 8) {
              printf("output[%d] mismatch: %.3f vs %.3f\n", output_idx,
                     output[output_idx], sum);
            } else if (num_mismatches == 8) {
              printf("...\n");
            }
            num_mismatches += 1;
          }
        }
      }
    }
  }
  printf("#mismatches: %d\n", num_mismatches);
}

struct BenchParams {
  virtual void Benchmark(int num_runs) = 0;
  virtual ~BenchParams() {}
};

struct BenchConvParams : public BenchParams {
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

  BenchConvParams(int batch_size, int input_channels, int height, int width,
                  int kernel_height, int kernel_width, int stride_height,
                  int stride_width, int padding_height, int padding_width,
                  int output_channels)
      : batch_size(batch_size),
        input_channels(input_channels),
        height(height),
        width(width),
        kernel_height(kernel_height),
        kernel_width(kernel_width),
        stride_height(stride_height),
        stride_width(stride_width),
        padding_height(padding_height),
        padding_width(padding_width),
        output_channels(output_channels) {}

  ~BenchConvParams() override {}

  void Benchmark(int num_runs) override {
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
    for (int i = 0; i < num_runs; i++) {
      total += ImplicitGEMM<float>(batch_size, input_channels, height, width,
                                   kernel_height, kernel_width, stride_height,
                                   stride_width, padding_height, padding_width,
                                   output_channels, d_a, d_b, d_c);
    }
    float flops = 2.0f * (double)batch_size * (double)output_height *
                  (double)output_width * (double)output_channels *
                  (double)input_channels * (double)kernel_height *
                  (double)kernel_width;
    float ms = total / num_runs;

    printf("[Workload] m = %d, n = %d, k = %d\n", m, n, k);
    printf("#runs = %d\n", num_runs);

    CheckCorrectness(batch_size, input_channels, height, width, kernel_height,
                     kernel_width, stride_height, stride_width, padding_height,
                     padding_width, output_channels, h_a, h_b, d_c);

    printf("ImplicitGEMM Conv: %.3fms\n", ms);
    printf("%.3f GFLOPs\n\n", flops * 1e3 / (float)(1 << 30) / ms);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

struct BenchMatMulParams : public BenchParams {
  int m;
  int n;
  int k;

  BenchMatMulParams(int m, int n, int k) : m(m), n(n), k(k) {}

  ~BenchMatMulParams() override {}

  void Benchmark(int num_runs) override {
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, sizeof(float) * m * k);
    cudaMalloc(&d_b, sizeof(float) * k * n);
    cudaMalloc(&d_c, sizeof(float) * m * n);
    auto h_a = Random(d_a, m * k);
    auto h_b = Random(d_b, k * n);

    float total = 0;
    for (int i = 0; i < num_runs; i++) {
      total += GEMM<float>(m, n, k, d_a, d_b, d_c);
    }
    float flops = 2.0f * (double)m * (double)n * (double)k;
    float ms = total / num_runs;

    printf("[Workload] m = %d, n = %d, k = %d\n", m, n, k);
    printf("#runs = %d\n", num_runs);

    printf("ImplicitGEMM MatMul: %.3fms\n", ms);
    printf("%.3f GFLOPs\n\n", flops * 1e3 / (float)(1 << 30) / ms);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

int main(int argc, char **argv) {
  auto suite = std::vector<BenchParams *>{
      new BenchMatMulParams(1024, 1024, 1024),
      new BenchMatMulParams(12544, 128, 288),
      new BenchMatMulParams(3136, 32, 1152),
      new BenchConvParams(64, 1, 28, 28, 3, 3, 1, 1, 1, 1, 32),
      new BenchConvParams(64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 128),
      new BenchConvParams(64, 128, 7, 7, 3, 3, 1, 1, 1, 1, 32),
  };

  int num_runs = 500;
  if (argc >= 2) {
    num_runs = std::stoi(argv[1]);
  }

  for (int i = 0; i < suite.size(); i++) {
    suite[i]->Benchmark(num_runs);
  }

  for (int i = 0; i < suite.size(); i++) delete suite[i];

  return 0;
}