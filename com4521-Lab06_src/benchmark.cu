#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.cuh"
#include "dispatcher.cuh"
#include "mm_cuda_core.cuh"

template <typename T>
float ConvForwardImplicitGEMM(int batch_size, int input_channels, int height,
                              int width, int kernel_height, int kernel_width,
                              int stride_height, int stride_width,
                              int padding_height, int padding_width,
                              int output_channels, T *input, T *weight, T *bias,
                              T *output) {
  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;

  Im2colLayout im2col_layout(
      batch_size, input_channels, height, width, kernel_height, kernel_width,
      stride_height, stride_width, padding_height, padding_width, false);
  MatrixWrapper<T, Im2colLayout> a(input, im2col_layout);
  WeightLayout weight_layout(output_channels, input_channels, kernel_height,
                             kernel_width, false);
  MatrixWrapper<T, WeightLayout> b(weight, weight_layout);
  OutputLayout output_layout(batch_size, output_channels, output_height,
                             output_width);
  MatrixWrapper<T, OutputLayout> c(output, output_layout);

  CudaTimer timer;
  MatrixMul<T, false>(a, b, c, bias, 0);
  return timer.End();
}

template <typename T>
std::pair<float, float> ConvBackwardImplicitGEMM(
    int batch_size, int input_channels, int height, int width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int padding_height, int padding_width, int output_channels, T *input,
    T *weight, T *output_grad, T *input_grad, T *weight_grad) {
  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;

  // tensors
  OutputLayout output_layout(batch_size, output_channels, output_height,
                             output_width);
  MatrixWrapper<T, OutputLayout> c(output_grad, output_layout);
  // bhw * cout
  WeightLayout weight_layout(output_channels, input_channels, kernel_height,
                             kernel_width, true);
  MatrixWrapper<T, WeightLayout> b(weight, weight_layout);
  // cin k^2 * cout
  Im2colLayout im2col_layout(batch_size, input_channels, height, width,
                             kernel_height, kernel_width, stride_height,
                             stride_width, padding_height, padding_width, true);
  MatrixWrapper<T, Im2colLayout> a(input, im2col_layout);
  // bhw * cin k^2

  // grads
  Im2colLayout input_grad_layout(
      batch_size, input_channels, height, width, kernel_height, kernel_width,
      stride_height, stride_width, padding_height, padding_width, false);
  MatrixWrapper<T, Im2colLayout> input_grad_matrix(input_grad,
                                                   input_grad_layout);
  WeightLayout weight_grad_layout(output_channels, input_channels,
                                  kernel_height, kernel_width, false);
  MatrixWrapper<T, WeightLayout> weight_grad_matrix(weight_grad,
                                                    weight_grad_layout);

  // calculation
  float ms0, ms1;
  {
    CudaTimer timer;
    MatrixMul<T, true>(c, b, input_grad_matrix, nullptr, 0);
    ms0 = timer.End();
  }
  {
    CudaTimer timer;
    MatrixMul<T, false>(a, c, weight_grad_matrix, nullptr, 0);
    ms1 = timer.End();
  }
  return {ms0, ms1};
}

template <typename T>
std::vector<T> Random(T *ptr, uint32_t size) {
  std::vector<T> v(size);
  for (int i = 0; i < size; i++) v[i] = rand() * 1.0 / RAND_MAX - 0.5;
  cudaMemcpy(ptr, v.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  CheckCUDAError("cudaMemcpy cudaMemcpyHostToDevice");
  return v;
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

  template <typename T>
  void CheckCorrectness(int batch_size, int input_channels, int height,
                        int width, int kernel_height, int kernel_width,
                        int stride_height, int stride_width, int padding_height,
                        int padding_width, int output_channels,
                        const std::vector<T> &input,
                        const std::vector<T> &weight,
                        const std::vector<T> &bias, T *d_output) {
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
            T sum = bias[co];

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

            if (abs((float)(output[output_idx] - sum)) >= 1e-3) {
              if (num_mismatches < 8) {
                printf("output[%d] mismatch: %.3f vs %.3f\n", output_idx,
                       (float)output[output_idx], (float)sum);
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

  void Benchmark(int num_runs) override {
    float *d_a, *d_b, *d_bias, *d_c;

    int output_height =
        (height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width =
        (width + 2 * padding_width - kernel_width) / stride_width + 1;
    int m = batch_size * output_height * output_width;
    int n = output_channels;
    int k = input_channels * kernel_height * kernel_width;

    cudaMalloc(&d_a, sizeof(float) * m * k);
    cudaMalloc(&d_b, sizeof(float) * k * n);
    cudaMalloc(&d_bias, sizeof(float) * n);
    cudaMalloc(&d_c, sizeof(float) * m * n);
    auto h_a = Random(d_a, m * k);
    auto h_b = Random(d_b, k * n);
    auto h_bias = Random(d_bias, n);

    float total = 0;
    for (int i = 0; i < num_runs; i++) {
      total += ConvForwardImplicitGEMM<float>(
          batch_size, input_channels, height, width, kernel_height,
          kernel_width, stride_height, stride_width, padding_height,
          padding_width, output_channels, d_a, d_b, d_bias, d_c);
    }
    float flops = 2.0f * batch_size * output_height * output_width *
                  output_channels * input_channels * kernel_height *
                  kernel_width;
    float ms = total / num_runs;

    printf("[Workload] m = %d, n = %d, k = %d\n", m, n, k);
    printf("#runs = %d\n", num_runs);

    CheckCorrectness<float>(batch_size, input_channels, height, width,
                            kernel_height, kernel_width, stride_height,
                            stride_width, padding_height, padding_width,
                            output_channels, h_a, h_b, h_bias, d_c);

    printf("ConvForwardImplicitGEMM: %.3fms\n", ms);
    printf("%.3f GFLOPs\n\n", flops * 1e3 / (float)(1 << 30) / ms);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_bias);
  }
};

struct BenchConvBackwardParams : public BenchParams {
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

  BenchConvBackwardParams(int batch_size, int input_channels, int height,
                          int width, int kernel_height, int kernel_width,
                          int stride_height, int stride_width,
                          int padding_height, int padding_width,
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

  ~BenchConvBackwardParams() override {}

  void Benchmark(int num_runs) override {
    float *d_input, *d_weight, *d_input_grad, *d_weight_grad, *d_output_grad;

    int output_height =
        (height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width =
        (width + 2 * padding_width - kernel_width) / stride_width + 1;
    int m = batch_size * height * width;
    int n = output_channels;
    int k = input_channels * kernel_height * kernel_width;

    cudaMalloc(&d_input, sizeof(float) * m * k);
    cudaMalloc(&d_weight, sizeof(float) * k * n);
    cudaMalloc(&d_input_grad, sizeof(float) * m * k);
    cudaMalloc(&d_weight_grad, sizeof(float) * k * n);
    cudaMalloc(&d_output_grad, sizeof(float) * m * n);
    auto h_a = Random(d_input, m * k);
    auto h_b = Random(d_weight, k * n);
    auto h_bias = Random(d_output_grad, m * n);

    float ms0 = 0, ms1 = 0;
    for (int i = 0; i < num_runs; i++) {
      auto res = ConvBackwardImplicitGEMM<float>(
          batch_size, input_channels, height, width, kernel_height,
          kernel_width, stride_height, stride_width, padding_height,
          padding_width, output_channels, d_input, d_weight, d_output_grad,
          d_input_grad, d_weight_grad);
      ms0 += res.first;
      ms1 += res.second;
    }
    ms0 /= num_runs;
    ms1 /= num_runs;

    float flops0 = 2.0f * batch_size * output_height * output_width *
                   input_channels * kernel_height * kernel_width *
                   output_channels;
    float flops1 = 2.0f * input_channels * kernel_height * kernel_width *
                   output_channels * batch_size * output_height * output_width;

    printf("[Workload] m = %d, n = %d, k = %d & m = %d, n = %d, k = %d\n",
           batch_size * output_height * output_width,
           input_channels * kernel_height * kernel_width, output_channels,
           input_channels * kernel_height * kernel_width, output_channels,
           batch_size * output_height * output_width);
    printf("#runs = %d\n", num_runs);
    printf("ConvBackwardImplicitGEMM: %.3fms, %.3fms\n", ms0, ms1);
    printf("%.3f GFLOPs, %.3f GFLOPs\n\n",
           flops0 * 1e3 / (float)(1 << 30) / ms0,
           flops1 * 1e3 / (float)(1 << 30) / ms1);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_input_grad);
    cudaFree(d_weight_grad);
    cudaFree(d_output_grad);
  }
};

struct BenchMatMulParams : public BenchParams {
  int m;
  int n;
  int k;

  BenchMatMulParams(int m, int n, int k) : m(m), n(n), k(k) {}

  ~BenchMatMulParams() override {}

  template <typename T>
  void CheckCorrectness(T *d_a, T *d_b, T *d_c, int num_runs) {
    float eps = 1e-3;
    if (std::is_same_v<T, __half>) {
      eps = 1e-1;
    }

    T *d_c_ref;
    cudaMalloc(&d_c_ref, m * n * sizeof(T));
    CheckCUDAError("cudaMalloc");

    cublasHandle_t handle;
    cublasCreate(&handle);
    CheckCUDAError("cublasCreate");
    T alpha = 1, beta = 0;

    if constexpr (std::is_same_v<T, __half>) {
      printf("cublasHgemm\n");
    } else {
      printf("cublasSgemm\n");
    }
    float total_ms = 0;
    for (int i = 0; i < num_runs; i++) {
      CudaTimer timer;
      if constexpr (std::is_same_v<T, __half>) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_a, m,
                    d_b, n, &beta, d_c_ref, m);
      } else {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_a, m,
                    d_b, n, &beta, d_c_ref, m);
      }
      total_ms += timer.End();
    }
    float ms = total_ms / num_runs;

    std::vector<T> c(m * n);
    cudaMemcpy(c.data(), d_c, c.size() * sizeof(T), cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpy cudaMemcpyDeviceToHost");
    std::vector<T> c_ref(m * n);
    cudaMemcpy(c_ref.data(), d_c_ref, c_ref.size() * sizeof(T),
               cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpy cudaMemcpyDeviceToHost");

    int num_mismatches = 0;
    for (int i = 0; i < m * n; i++)
      num_mismatches += abs((float)c[i] - (float)c_ref[i]) >= eps;
    printf("cublas: %.3fms\n", ms);
    printf("cublas %.3f GFLOPs\n", 1.0 * m * n * k * 2 * 1e3 / (1 << 30) / ms);
    printf("#mismatches: %d\n", num_mismatches);

    cudaFree(d_c_ref);
  }

  void Benchmark(int num_runs) override {
    using T = float;

    T *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, sizeof(T) * m * k);
    cudaMalloc(&d_b, sizeof(T) * k * n);
    cudaMalloc(&d_c, sizeof(T) * m * n);
    Random(d_a, m * k);
    Random(d_b, k * n);

    MatrixWrapper<T, ColMajorLayout> a(d_a, ColMajorLayout(m, k));
    MatrixWrapper<T, RowMajorLayout> b(d_b, RowMajorLayout(k, n));
    MatrixWrapper<T, ColMajorLayout> c(d_c, ColMajorLayout(m, n));

    float total = 0;
    for (int i = 0; i < num_runs; i++) {
      CudaTimer timer;
      MatrixMul<T, false>(a, b, c, nullptr, 0);
      total += timer.End();
      CheckCUDAError("GEMM");
    }
    float flops = 2.0f * m * n * k;
    float ms = total / num_runs;

    printf("[Workload] m = %d, n = %d, k = %d\n", m, n, k);
    CheckCorrectness<T>(d_a, d_b, d_c, num_runs);
    printf("#runs = %d\n", num_runs);

    printf("MatMul: %.3fms\n", ms);
    printf("%.3f GFLOPs\n\n", flops * 1e3 / (float)(1 << 30) / ms);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

int main(int argc, char **argv) {
  auto suite = std::vector<BenchParams *>{
      new BenchMatMulParams(4096, 4096, 4096),
      new BenchMatMulParams(12544, 128, 288),
      new BenchMatMulParams(3136, 32, 1152),
      new BenchMatMulParams(9, 32, 64 * 28 * 28),
      new BenchConvBackwardParams(64, 1, 28, 28, 3, 3, 1, 1, 1, 1, 32),
      new BenchConvBackwardParams(64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 128),
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