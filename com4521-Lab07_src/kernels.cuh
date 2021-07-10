#ifndef KERNEL_H  // ensures header is only included once
#define KERNEL_H

// #ifndef __CUDACC__
// #define __CUDACC__
// #endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_RECORDS 2048
#define THREADS_PER_BLOCK 256
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

struct student_record {
  int student_id;
  float assignment_mark;
};

struct student_records {
  int student_ids[NUM_RECORDS];
  float assignment_marks[NUM_RECORDS];
};

__device__ float d_max_mark = 0;
__device__ int d_max_mark_student_id = 0;

// lock for global Atomics
#define UNLOCKED 0
#define LOCKED 1
__device__ volatile int lock = UNLOCKED;

// Function creates an atomic compare and swap to save the maximum mark and
// associated student id
__device__ void SetMaxMarkAtomic(float mark, int id) {
  bool needlock = true;

  while (needlock) {
    // get lock to perform critical section of code
    if (atomicCAS((int *)&lock, UNLOCKED, LOCKED) == 0) {
      // critical section of code
      if (d_max_mark < mark) {
        d_max_mark_student_id = id;
        d_max_mark = mark;
      }

      // free lock
      atomicExch((int *)&lock, 0);
      needlock = false;
    }
  }
}

// Naive atomic implementation
__global__ void MaximumMarkAtomicKernel(student_records *d_records) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float mark = d_records->assignment_marks[idx];
  int id = d_records->student_ids[idx];

  SetMaxMarkAtomic(mark, id);
}

// Exercise 2) Recursive Reduction
__global__ void MaximumMarkRecursiveKernel(student_records *d_records,
                                           student_records *d_reduced_records) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Exercise 2.1) Load a single student record into shared memory
  __shared__ student_record s_records[THREADS_PER_BLOCK];

  s_records[threadIdx.x] = {
      .student_id = d_records->student_ids[idx],
      .assignment_mark = d_records->assignment_marks[idx],
  };
  __syncthreads();

  // Exercise 2.2) Compare two values and write the result to d_reduced_records
  if ((idx & 1) == 0) {
    d_reduced_records->assignment_marks[idx >> 1] =
        max(s_records[threadIdx.x].assignment_mark,
            s_records[threadIdx.x | 1].assignment_mark);
    d_reduced_records->student_ids[idx >> 1] =
        s_records[threadIdx.x].assignment_mark >
                s_records[threadIdx.x | 1].assignment_mark
            ? s_records[threadIdx.x].student_id
            : s_records[threadIdx.x | 1].student_id;
  }
}

// Exercise 3) Using block level reduction
__global__ void MaximumMarkSMKernel(student_records *d_records,
                                    student_records *d_reduced_records) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Exercise 3.1) Load a single student record into shared memory
  __shared__ int ids[THREADS_PER_BLOCK];
  __shared__ float marks[THREADS_PER_BLOCK];
  ids[threadIdx.x] = d_records->student_ids[idx];
  marks[threadIdx.x] = d_records->assignment_marks[idx];
  __syncthreads();

  // Exercise 3.2) Strided shared memory conflict free reduction
  static_assert(THREADS_PER_BLOCK == 256, "");
#pragma unroll
  for (int i = 0; i <= 7; i++) {
    int stride = 1 << (7 - i);

    if (threadIdx.x < stride) {
      ids[threadIdx.x] = marks[threadIdx.x + stride] > marks[threadIdx.x]
                             ? ids[threadIdx.x + stride]
                             : ids[threadIdx.x];
      marks[threadIdx.x] = max(marks[threadIdx.x + stride], marks[threadIdx.x]);
    }

    __syncthreads();
  }

  // Exercise 3.3) Write the result
  if (threadIdx.x == 0) {
    d_reduced_records->student_ids[blockIdx.x] = ids[0];
    d_reduced_records->assignment_marks[blockIdx.x] = marks[0];
  }
}

// Exercise 4) Using warp level reduction
__global__ void MaximumMarkShuffleKernel(student_records *d_records,
                                         student_records *d_reduced_records) {
  // Exercise 4.1) Complete the kernel
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int id = d_records->student_ids[idx];
  float mark = d_records->assignment_marks[idx];

  static_assert(THREADS_PER_BLOCK == 256, "");

  constexpr unsigned kFullMask = (unsigned int)((1ll << 32) - 1);

#pragma unroll
  for (int i = 0; i <= 7; i++) {
    int mask = 1 << (7 - i);

    int oid = __shfl_xor_sync(kFullMask, id, mask);
    float omark = __shfl_xor_sync(kFullMask, mark, mask);

    id = mark > omark ? id : oid;
    mark = max(mark, omark);
  }

  if ((idx & 31) == 0) {
    d_reduced_records->student_ids[idx >> 5] = id;
    d_reduced_records->assignment_marks[idx >> 5] = mark;
  }
}

#endif  // KERNEL_H