#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// include kernels and cuda headers after definitions of structures
#include "common.cuh"
#include "kernels.cuh"

void ReadRecords(student_record *records);

void MaximumMarkAtomic(student_records *, student_records *, student_records *,
                       student_records *);
void MaximumMarkRecursive(student_records *, student_records *,
                          student_records *, student_records *);
void MaximumMarkSM(student_records *, student_records *, student_records *,
                   student_records *);
void MaximumMarkShuffle(student_records *, student_records *, student_records *,
                        student_records *);

int main(void) {
  student_record *records_aos;
  student_records *h_records;
  student_records *h_records_result;
  student_records *d_records;
  student_records *d_records_result;

  // host allocation
  records_aos = (student_record *)malloc(sizeof(student_record) * NUM_RECORDS);
  h_records = (student_records *)malloc(sizeof(student_records));
  h_records_result = (student_records *)malloc(sizeof(student_records));

  // device allocation
  cudaMalloc((void **)&d_records, sizeof(student_records));
  cudaMalloc((void **)&d_records_result, sizeof(student_records));
  CheckCUDAError("CUDA malloc");

  // read file
  ReadRecords(records_aos);

  // Exercise 1.1) Convert records_aos to a structure of arrays in h_records
  for (int i = 0; i < NUM_RECORDS; i++) {
    h_records->student_ids[i] = records_aos[i].student_id;
    h_records->assignment_marks[i] = records_aos[i].assignment_mark;
  }

  // free AOS as it is no longer needed
  free(records_aos);

  // apply each approach in turn
  MaximumMarkAtomic(h_records, h_records_result, d_records, d_records_result);
  MaximumMarkRecursive(h_records, h_records_result, d_records,
                       d_records_result);
  MaximumMarkSM(h_records, h_records_result, d_records, d_records_result);
  MaximumMarkShuffle(h_records, h_records_result, d_records, d_records_result);

  // Cleanup
  free(h_records);
  free(h_records_result);
  cudaFree(d_records);
  cudaFree(d_records_result);
  CheckCUDAError("CUDA cleanup");

  return 0;
}

void ReadRecords(student_record *records) {
  FILE *f = nullptr;
  f = fopen("com4521.dat", "rb");  // read and binary flags
  if (f == nullptr) {
    fprintf(stderr, "Error: Could not find com4521.dat file \n");
    exit(1);
  }

  // read student data
  if (fread(records, sizeof(student_record), NUM_RECORDS, f) != NUM_RECORDS) {
    fprintf(stderr, "Error: Unexpected end of file!\n");
    exit(1);
  }
  fclose(f);
}

void MaximumMarkAtomic(student_records *h_records,
                       student_records *h_records_result,
                       student_records *d_records,
                       student_records *d_records_result) {
  float max_mark;
  int max_mark_student_id;
  float time;
  cudaEvent_t start, stop;

  max_mark = 0;
  max_mark_student_id = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // memory copy records to device
  cudaMemcpy(d_records, h_records, sizeof(student_records),
             cudaMemcpyHostToDevice);
  CheckCUDAError("1) CUDA memcpy");

  cudaEventRecord(start, 0);
  // find highest mark using GPU
  dim3 blocks_per_grid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
  dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
  MaximumMarkAtomicKernel<<<blocks_per_grid, threads_per_block>>>(d_records);
  cudaDeviceSynchronize();
  CheckCUDAError("Atomics: CUDA kernel");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Copy result back to host
  cudaMemcpyFromSymbol(&max_mark, d_max_mark, sizeof(float));
  cudaMemcpyFromSymbol(&max_mark_student_id, d_max_mark_student_id,
                       sizeof(int));
  CheckCUDAError("Atomics: CUDA memcpy back");

  cudaEventElapsedTime(&time, start, stop);

  // output result
  printf("Atomics: Highest mark recorded %f was by student %d\n", max_mark,
         max_mark_student_id);
  printf("\tExecution time was %f ms\n", time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Exercise 2)
void MaximumMarkRecursive(student_records *h_records,
                          student_records *h_records_result,
                          student_records *d_records,
                          student_records *d_records_result) {
  int i;
  float max_mark;
  int max_mark_student_id;
  student_records *d_records_temp1, *d_records_temp2;
  float time;
  cudaEvent_t start, stop;

  max_mark = 0;
  max_mark_student_id = 0;

  cudaMalloc(&d_records_temp1, sizeof(student_records));
  cudaMalloc(&d_records_temp2, sizeof(student_records));
  CheckCUDAError("malloc d_records_temp");

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // memory copy records to device
  cudaMemcpy(d_records, h_records, sizeof(student_records),
             cudaMemcpyHostToDevice);
  CheckCUDAError("Recursive: CUDA memcpy");

  cudaEventRecord(start, 0);

  // Exercise 2.3) Recursively call GPU steps until there are THREADS_PER_BLOCK
  // values left
  dim3 blocks_per_grid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
  dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
  MaximumMarkRecursiveKernel<<<blocks_per_grid, threads_per_block>>>(
      d_records, d_records_temp2);

  for (int i = NUM_RECORDS / 2; i > THREADS_PER_BLOCK; i /= 2) {
    std::swap(d_records_temp1, d_records_temp2);
    blocks_per_grid = dim3(i / THREADS_PER_BLOCK, 1, 1);
    threads_per_block = dim3(THREADS_PER_BLOCK, 1, 1);
    MaximumMarkRecursiveKernel<<<blocks_per_grid, threads_per_block>>>(
        d_records_temp1, d_records_temp2);
  }

  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  // Exercise 2.4) copy back the final THREADS_PER_BLOCK values
  cudaMemcpy(h_records_result, d_records_temp2, sizeof(student_records),
             cudaMemcpyDeviceToHost);
  CheckCUDAError("memcpy to h_records_result");
  cudaFree(d_records_temp1);
  cudaFree(d_records_temp2);

  // Exercise 2.5) reduce the final THREADS_PER_BLOCK values on CPU

  for (int i = 0; i < THREADS_PER_BLOCK; i++) {
    if (h_records_result->assignment_marks[i] > max_mark) {
      max_mark = h_records_result->assignment_marks[i];
      max_mark_student_id = h_records_result->student_ids[i];
    }
  }

  // output the result
  printf("Recursive: Highest mark recorded %f was by student %d\n", max_mark,
         max_mark_student_id);
  printf("\tExecution time was %f ms\n", time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Exercise 3)
void MaximumMarkSM(student_records *h_records,
                   student_records *h_records_result,
                   student_records *d_records,
                   student_records *d_records_result) {
  unsigned int i;
  float max_mark;
  int max_mark_student_id;
  float time;
  cudaEvent_t start, stop;

  max_mark = 0;
  max_mark_student_id = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // memory copy records to device
  cudaMemcpy(d_records, h_records, sizeof(student_records),
             cudaMemcpyHostToDevice);
  CheckCUDAError("SM: CUDA memcpy");

  cudaEventRecord(start, 0);

  // Exercise 3.4) Call the shared memory reduction kernel
  dim3 blocks_per_grid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
  dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
  MaximumMarkSMKernel<<<blocks_per_grid, threads_per_block>>>(d_records,
                                                              d_records_result);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Exercise 3.5) Copy the final block values back to CPU
  cudaMemcpy(h_records_result, d_records_result, sizeof(student_records),
             cudaMemcpyDeviceToHost);
  CheckCUDAError("memcpy to h_records_result");

  // Exercise 3.6) Reduce the block level results on CPU
  for (int i = 0; i < blocks_per_grid.x; i++) {
    if (h_records_result->assignment_marks[i] > max_mark) {
      max_mark = h_records_result->assignment_marks[i];
      max_mark_student_id = h_records_result->student_ids[i];
    }
  }

  cudaEventElapsedTime(&time, start, stop);

  // output result
  printf("SM: Highest mark recorded %f was by student %d\n", max_mark,
         max_mark_student_id);
  printf("\tExecution time was %f ms\n", time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Exercise 4)
void MaximumMarkShuffle(student_records *h_records,
                        student_records *h_records_result,
                        student_records *d_records,
                        student_records *d_records_result) {
  unsigned int i;
  unsigned int warps_per_grid;
  float max_mark;
  int max_mark_student_id;
  float time;
  cudaEvent_t start, stop;

  max_mark = 0;
  max_mark_student_id = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // memory copy records to device
  cudaMemcpy(d_records, h_records, sizeof(student_records),
             cudaMemcpyHostToDevice);
  CheckCUDAError("Shuffle: CUDA memcpy");

  cudaEventRecord(start, 0);

  dim3 blocks_per_grid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
  dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
  MaximumMarkShuffleKernel<<<blocks_per_grid, threads_per_block>>>(
      d_records, d_records_result);
  // Exercise 4.2) Execute the kernel, copy back result, reduce final values on
  // CPU

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  cudaMemcpy(h_records_result, d_records_result, sizeof(student_records),
             cudaMemcpyDeviceToHost);
  CheckCUDAError("memcpy to h_records_result");
  for (int i = 0; i < NUM_RECORDS >> 5; i++) {
    if (h_records_result->assignment_marks[i] > max_mark) {
      max_mark = h_records_result->assignment_marks[i];
      max_mark_student_id = h_records_result->student_ids[i];
    }
  }

  // output result
  printf("Shuffle: Highest mark recorded %f was by student %d\n", max_mark,
         max_mark_student_id);
  printf("\tExecution time was %f ms\n", time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}