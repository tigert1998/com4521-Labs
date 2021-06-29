#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <texture_fetch_functions.hpp>

#include "header.cuh"

#define IMAGE_DIM 2048
#define SAMPLE_SIZE 6
#define NUMBER_OF_SAMPLES (((SAMPLE_SIZE * 2) + 1) * ((SAMPLE_SIZE * 2) + 1))

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

void OutputImageFile(const char *filename, uchar4 *image);
void InputImageFile(const char *filename, uchar4 *image);
void CheckCUDAError(const char *msg);

#define IMAGE_BLUR(read_image, image_output)                       \
  do {                                                             \
    int x = threadIdx.x + blockIdx.x * blockDim.x;                 \
    int y = threadIdx.y + blockIdx.y * blockDim.y;                 \
    int output_offset = x + y * blockDim.x * gridDim.x;            \
    uchar4 pixel;                                                  \
    float4 average = make_float4(0, 0, 0, 0);                      \
    for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++) {            \
      for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++) {          \
        int x_offset = x + i;                                      \
        int y_offset = y + j;                                      \
        if (x_offset < 0) x_offset += IMAGE_DIM;                   \
        if (x_offset >= IMAGE_DIM) x_offset -= IMAGE_DIM;          \
        if (y_offset < 0) y_offset += IMAGE_DIM;                   \
        if (y_offset >= IMAGE_DIM) y_offset -= IMAGE_DIM;          \
        int offset = x_offset + y_offset * blockDim.x * gridDim.x; \
        pixel = read_image(x + i, y + j, offset);                  \
        average.x += pixel.x;                                      \
        average.y += pixel.y;                                      \
        average.z += pixel.z;                                      \
      }                                                            \
    }                                                              \
    average.x /= (float)NUMBER_OF_SAMPLES;                         \
    average.y /= (float)NUMBER_OF_SAMPLES;                         \
    average.z /= (float)NUMBER_OF_SAMPLES;                         \
    image_output[output_offset].x = (unsigned char)average.x;      \
    image_output[output_offset].y = (unsigned char)average.y;      \
    image_output[output_offset].z = (unsigned char)average.z;      \
    image_output[output_offset].w = 255;                           \
  } while (0)

__global__ void ImageBlur(uchar4 *image, uchar4 *image_output) {
#define READ_IMAGE(x, y, i) image[i]
  IMAGE_BLUR(READ_IMAGE, image_output);
#undef READ_IMAGE
}

texture<uchar4, cudaTextureType1D, cudaReadModeElementType> image_tex_1d;

__global__ void ImageBlurTexture1D(uchar4 *image_output) {
#define READ_IMAGE(x, y, i) tex1Dfetch(image_tex_1d, i)
  IMAGE_BLUR(READ_IMAGE, image_output);
#undef READ_IMAGE
}

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> image_tex_2d;

__global__ void ImageBlurTexture2D(uchar4 *image_output) {
#define READ_IMAGE(x, y, i) tex2D(image_tex_2d, x, y)
  IMAGE_BLUR(READ_IMAGE, image_output);
#undef READ_IMAGE
}

/* Host code */

int main(void) {
  unsigned int image_size;
  uchar4 *d_image, *d_image_output;
  uchar4 *h_image;
  cudaEvent_t start, stop;
  float3 ms;  //[0]=normal,[1]=tex1d,[2]=tex2d

  image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar4);

  // create timers
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on the GPU for the output image
  cudaMalloc((void **)&d_image, image_size);
  cudaMalloc((void **)&d_image_output, image_size);
  CheckCUDAError("CUDA malloc");

  auto channel_desc = cudaCreateChannelDesc<uchar4>();
  image_tex_2d.addressMode[0] = cudaAddressModeWrap;
  image_tex_2d.addressMode[1] = cudaAddressModeWrap;
  cudaBindTexture(nullptr, image_tex_1d, d_image, channel_desc, image_size);
  cudaBindTexture2D(nullptr, image_tex_2d, d_image, channel_desc, IMAGE_DIM,
                    IMAGE_DIM, IMAGE_DIM * sizeof(uchar4));

  // allocate and load host image
  h_image = (uchar4 *)malloc(image_size);
  InputImageFile("input.ppm", h_image);

  // copy image to device memory
  cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
  CheckCUDAError("CUDA memcpy to device");

  // cuda layout and execution
  dim3 blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
  dim3 threadsPerBlock(16, 16);

  // normal version
  TIME("kernel (normal)", ms.x, ImageBlur, blocksPerGrid, threadsPerBlock,
       d_image, d_image_output);

  // copy the image back from the GPU for output to file
  cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
  CheckCUDAError("CUDA memcpy from device");
  OutputImageFile("normal.ppm", h_image);

  TIME("kernel (texture 1D)", ms.y, ImageBlurTexture1D, blocksPerGrid,
       threadsPerBlock, d_image_output);

  // copy the image back from the GPU for output to file
  cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
  CheckCUDAError("CUDA memcpy from device");
  OutputImageFile("tex_1d.ppm", h_image);

  TIME("kernel (texture 2D)", ms.z, ImageBlurTexture2D, blocksPerGrid,
       threadsPerBlock, d_image_output);

  // copy the image back from the GPU for output to file
  cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
  CheckCUDAError("CUDA memcpy from device");
  OutputImageFile("tex_2d.ppm", h_image);

  // output timings
  printf("Execution times:\n");
  printf("\tNormal version: %f\n", ms.x);
  printf("\ttex1D version: %f\n", ms.y);
  printf("\ttex2D version: %f\n", ms.z);

  // output image

  // cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_image);
  cudaFree(d_image_output);
  free(h_image);

  cudaUnbindTexture(image_tex_1d);
  cudaUnbindTexture(image_tex_2d);

  return 0;
}

void OutputImageFile(const char *filename, uchar4 *image) {
  FILE *f;  // output file handle

  // open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL) {
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    exit(1);
  }
  fprintf(f, "P6\n");
  fprintf(f, "# COM4521 Lab 05 Exercise03\n");
  fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
  for (int x = 0; x < IMAGE_DIM; x++) {
    for (int y = 0; y < IMAGE_DIM; y++) {
      int i = x + y * IMAGE_DIM;
      fwrite(&image[i], sizeof(unsigned char), 3,
             f);  // only write rgb (ignoring a)
    }
  }

  fclose(f);
}

void InputImageFile(const char *filename, uchar4 *image) {
  FILE *f;  // input file handle
  char temp[256];
  unsigned int x, y, s;

  // open the input file and write header info for PPM filetype
  f = fopen("input.ppm", "rb");
  if (f == NULL) {
    fprintf(stderr, "Error opening 'input.ppm' input file\n");
    exit(1);
  }
  fscanf(f, "%s\n", &temp);
  fscanf(f, "%d %d\n", &x, &y);
  fscanf(f, "%d\n", &s);
  if ((x != y) && (x != IMAGE_DIM)) {
    fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
    exit(1);
  }

  for (int x = 0; x < IMAGE_DIM; x++) {
    for (int y = 0; y < IMAGE_DIM; y++) {
      int i = x + y * IMAGE_DIM;
      fread(&image[i], sizeof(unsigned char), 3, f);  // only read rgb
      // image[i].w = 255;
    }
  }

  fclose(f);
}

void CheckCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
