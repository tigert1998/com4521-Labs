/*
 * Source code for this lab class is modifed from the book CUDA by Exmaple and
 * provided by permission of NVIDIA Corporation
 */

#include <vector_functions.h>
#include <vector_types.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IMAGE_DIM 2048
#define MAX_SPHERES 2048

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

void OutputImageFile(uchar4 *image, const std::string &filename);
void CheckCUDAError(const char *msg);

struct Sphere {
  float r, b, g;
  float radius;
  float x, y, z;
};

enum AccessType {
  kNormal,
  kReadOnly,
  kConst,
};

struct SphereBundle {
  Sphere *normal_spheres;
  const Sphere *__restrict__ readonly_spheres;
  Sphere *const_spheres;
};

/* Device Code */

__constant__ unsigned int d_sphere_count;

__constant__ Sphere const_spheres[MAX_SPHERES];

template <AccessType AT>
__device__ Sphere GetSphere(int index, SphereBundle bundle);

template <>
__device__ Sphere GetSphere<kNormal>(int index, SphereBundle bundle) {
  return bundle.normal_spheres[index];
}

template <>
__device__ Sphere GetSphere<kReadOnly>(int index, SphereBundle bundle) {
  static_assert(sizeof(Sphere) % sizeof(float) == 0, "");
  Sphere ret;
#pragma unroll
  for (int i = 0; i < sizeof(Sphere) / sizeof(float); i++) {
    ((float *)&ret)[i] = __ldg(((float *)&bundle.readonly_spheres[index]) + i);
  }
  return ret;
}

template <>
__device__ Sphere GetSphere<kConst>(int index, SphereBundle bundle) {
  return bundle.const_spheres[index];
}

#define SPHERE_INTERSECT(s, ox, oy, t, n)                        \
  do {                                                           \
    float dx = ox - s.x;                                         \
    float dy = oy - s.y;                                         \
    if (dx * dx + dy * dy < s.radius * s.radius) {               \
      float dz = sqrtf(s.radius * s.radius - dx * dx - dy * dy); \
      n = dz / s.radius;                                         \
      t = dz + s.z;                                              \
    } else {                                                     \
      t = -INF;                                                  \
    }                                                            \
  } while (0)

template <AccessType AT>
__global__ void RayTrace(uchar4 *image, SphereBundle bundle) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float ox = (x - IMAGE_DIM / 2.0f);
  float oy = (y - IMAGE_DIM / 2.0f);

  float r = 0, g = 0, b = 0;
  float maxz = -INF;
  for (int i = 0; i < d_sphere_count; i++) {
    Sphere s = GetSphere<AT>(i, bundle);
    float n, t;
    SPHERE_INTERSECT(s, ox, oy, t, n);
    if (t > maxz) {
      float fscale = n;
      r = s.r * fscale;
      g = s.g * fscale;
      b = s.b * fscale;
      maxz = t;
    }
  }

  image[offset].x = (int)(r * 255);
  image[offset].y = (int)(g * 255);
  image[offset].z = (int)(b * 255);
  image[offset].w = 255;
}

/* Host code */

SphereBundle Prepare(Sphere *h_s) {
  auto size = MAX_SPHERES * sizeof(Sphere);
  SphereBundle sphere_bundle;
  cudaMalloc((void **)&sphere_bundle.normal_spheres, size);
  CheckCUDAError("CUDA malloc");
  sphere_bundle.readonly_spheres = sphere_bundle.normal_spheres;
  cudaGetSymbolAddress((void **)&sphere_bundle.const_spheres, const_spheres);
  CheckCUDAError("CUDA get symbol address");

  cudaMemcpy(sphere_bundle.normal_spheres, h_s, size, cudaMemcpyHostToDevice);
  cudaMemcpy(sphere_bundle.const_spheres, h_s, size, cudaMemcpyHostToDevice);
  CheckCUDAError("CUDA memcpy to device");

  return sphere_bundle;
}

int main(void) {
  unsigned int image_size;
  uchar4 *h_image, *d_image;
  Sphere *h_s = (Sphere *)malloc(MAX_SPHERES * sizeof(Sphere));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float3 timing_data;  // timing data where [0]=normal, [1]=read-only, [2]=const

  image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar4);

  // allocate memory on the GPU for the output image
  cudaMalloc((void **)&d_image, image_size);
  CheckCUDAError("CUDA malloc");

  // create some random spheres
  for (int i = 0; i < MAX_SPHERES; i++) {
    h_s[i].r = rnd(1.0f);
    h_s[i].g = rnd(1.0f);
    h_s[i].b = rnd(1.0f);
    h_s[i].x = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
    h_s[i].y = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
    h_s[i].z = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
    h_s[i].radius = rnd(100.0f) + 20;
  }
  // copy to device memory
  auto bundle = Prepare(h_s);

  // generate host image
  h_image = (uchar4 *)malloc(image_size);

  // cuda layout
  dim3 blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
  dim3 threadsPerBlock(16, 16);

  // output timings
  printf("Timing Data Table\n Spheres | Normal | Read-only | Const\n");
  for (unsigned int sphere_count = 16; sphere_count <= MAX_SPHERES;
       sphere_count *= 2) {
    cudaMemcpyToSymbol(d_sphere_count, &sphere_count, sizeof(unsigned int));
    CheckCUDAError("CUDA copy sphere count to device");

    // generate a image from the sphere data
    cudaEventRecord(start, 0);
    RayTrace<kNormal><<<blocksPerGrid, threadsPerBlock>>>(d_image, bundle);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing_data.x, start, stop);
    CheckCUDAError("kernel (normal)");

    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    CheckCUDAError("CUDA memcpy from device");
    OutputImageFile(h_image, std::string("normal.") +
                                 std::to_string(sphere_count) + ".ppm");

    cudaEventRecord(start, 0);
    RayTrace<kReadOnly><<<blocksPerGrid, threadsPerBlock>>>(d_image, bundle);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing_data.y, start, stop);
    CheckCUDAError("kernel (read only)");

    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    CheckCUDAError("CUDA memcpy from device");
    OutputImageFile(h_image, std::string("readonly.") +
                                 std::to_string(sphere_count) + ".ppm");

    cudaEventRecord(start, 0);
    RayTrace<kConst><<<blocksPerGrid, threadsPerBlock>>>(d_image, bundle);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing_data.z, start, stop);
    CheckCUDAError("kernel (constant)");

    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    CheckCUDAError("CUDA memcpy from device");
    OutputImageFile(h_image, std::string("constant.") +
                                 std::to_string(sphere_count) + ".ppm");

    printf(" %-7i | %-6.3f | %-9.3f | %.3f\n", sphere_count, timing_data.x,
           timing_data.y, timing_data.z);
  }

  // cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_image);
  free(h_image);

  return 0;
}

void OutputImageFile(uchar4 *image, const std::string &filename) {
  FILE *f;  // output file handle

  // open the output file and write header info for PPM filetype
  f = fopen(filename.c_str(), "wb");
  if (f == NULL) {
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    exit(1);
  }
  fprintf(f, "P6\n");
  fprintf(f, "# COM4521 Lab 05 Exercise02\n");
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

void CheckCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
