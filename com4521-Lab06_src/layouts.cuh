#pragma once

class ColMajorLayout {
 public:
  int matrix_height, matrix_width;
  __device__ __host__ ColMajorLayout(int matrix_height, int matrix_width)
      : matrix_height(matrix_height), matrix_width(matrix_width) {}
  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;
    return x + y * matrix_height;
  }
};

class RowMajorLayout {
 public:
  int matrix_height, matrix_width;
  __device__ __host__ RowMajorLayout(int matrix_height, int matrix_width)
      : matrix_height(matrix_height), matrix_width(matrix_width) {}
  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;
    return x * matrix_width + y;
  }
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
  __device__ __host__ void Add(int x, int y, T value) {
    int i = layout.Index(x, y);
    if (i >= 0) atomicAdd(ptr + i, value);
  }
  __device__ __host__ T Get(int x, int y) {
    int i = layout.Index(x, y);
    return i < 0 ? (T)0 : ptr[i];
  }
};

class Im2colLayout {
 public:
  int channels, height, width, kernel_height, kernel_width, stride_height,
      stride_width, padding_height, padding_width;
  int output_height, output_width, matrix_height, matrix_width;
  bool t;
  __device__ __host__ Im2colLayout(int batch_size, int channels, int height,
                                   int width, int kernel_height,
                                   int kernel_width, int stride_height,
                                   int stride_width, int padding_height,
                                   int padding_width, bool t)
      : channels(channels),
        height(height),
        width(width),
        kernel_height(kernel_height),
        kernel_width(kernel_width),
        stride_height(stride_height),
        stride_width(stride_width),
        padding_height(padding_height),
        padding_width(padding_width),
        t(t) {
    output_height =
        (height + 2 * padding_height - kernel_height) / stride_height + 1;
    output_width =
        (width + 2 * padding_width - kernel_width) / stride_width + 1;

    if (t) {
      matrix_height = channels * kernel_height * kernel_width;
      matrix_width = batch_size * output_height * output_width;
    } else {
      matrix_height = batch_size * output_height * output_width;
      matrix_width = channels * kernel_height * kernel_width;
    }
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;

    int batch_idx, output_x, output_y, channel_idx, kernel_x, kernel_y;
    if (t) {
      batch_idx = y / (output_height * output_width);
      output_x = y / output_width % output_height;
      output_y = y % output_width;
      channel_idx = x / (kernel_height * kernel_width);
      kernel_x = x / kernel_width % kernel_height;
      kernel_y = x % kernel_width;
    } else {
      batch_idx = x / (output_height * output_width);
      output_x = x / output_width % output_height;
      output_y = x % output_width;
      channel_idx = y / (kernel_height * kernel_width);
      kernel_x = y / kernel_width % kernel_height;
      kernel_y = y % kernel_width;
    };

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
  bool t;

  __device__ __host__ WeightLayout(int output_channels, int input_channels,
                                   int kernel_height, int kernel_width, bool t)
      : t(t) {
    if (t) {
      matrix_height = output_channels;
      matrix_width = input_channels * kernel_height * kernel_width;
    } else {
      matrix_height = input_channels * kernel_height * kernel_width;
      matrix_width = output_channels;
    }
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;
    if (t) {
      return x * matrix_width + y;
    } else {
      return y * matrix_height + x;
    }
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
