#ifndef COMMON_CUH_
#define COMMON_CUH_

#define TIME(message, ms, func_name, grid_size, block_size, ...) \
  do {                                                           \
    cudaEvent_t start, stop;                                     \
    cudaEventCreate(&start);                                     \
    cudaEventCreate(&stop);                                      \
    cudaEventRecord(start, 0);                                   \
    func_name<<<grid_size, block_size>>>(__VA_ARGS__);           \
    cudaEventRecord(stop, 0);                                    \
    cudaEventSynchronize(stop);                                  \
    CheckCUDAError(message);                                     \
    cudaEventElapsedTime(&ms, start, stop);                      \
    cudaEventDestroy(start);                                     \
    cudaEventDestroy(stop);                                      \
  } while (0)

#endif