#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <tuple>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common.cuh"

using std::pair;

typedef enum {
  CALCULATOR_ADD,
  CALCULATOR_SUB,
  CALCULATOR_DIV,
  CALCULATOR_MUL
} CALCULATOR_COMMANDS;

typedef enum { INPUT_RANDOM, INPUT_LINEAR } INPUT_TYPE;

#define SAMPLES 262144
#define TPB 256
#define FILE_BUFFER_SIZE 32
#define MAX_COMMANDS 32
#define INPUT INPUT_LINEAR

__constant__ CALCULATOR_COMMANDS d_commands[MAX_COMMANDS];
__constant__ float d_operands[MAX_COMMANDS];

int NUM_STREAMS;

int readCommandsFromFile(CALCULATOR_COMMANDS *commands, float *operands);
void InitInput(float *input);
int ReadLine(FILE *f, char buffer[]);
pair<float, int> cudaCalculatorDefaultStream(CALCULATOR_COMMANDS *commands,
                                             float *operands, int num_commands);
pair<float, int> cudaCalculatorNStream1(CALCULATOR_COMMANDS *commands,
                                        float *operands, int num_commands);
pair<float, int> cudaCalculatorNStream2(CALCULATOR_COMMANDS *commands,
                                        float *operands, int num_commands);
int CheckResults(float *h_input, float *h_output, CALCULATOR_COMMANDS *commands,
                 float *operands, int num_commands);

__global__ void ParallelCalculator(float *input, float *output,
                                   int num_commands) {
  float out;
  int idx;

  idx = threadIdx.x + blockIdx.x * blockDim.x;

  // get input
  out = input[idx];

  // apply commands
  for (int i = 0; i < num_commands; i++) {
    CALCULATOR_COMMANDS cmd = d_commands[i];
    float v = d_operands[i];

    switch (cmd) {
      case (CALCULATOR_ADD): {
        out += v;
        break;
      }
      case (CALCULATOR_SUB): {
        out -= v;
        break;
      }
      case (CALCULATOR_DIV): {
        out /= v;
        break;
      }
      case (CALCULATOR_MUL): {
        out *= v;
        break;
      }
    }
  }

  output[idx] = out;
}

void Benchmark(const char *name, int warmup_runs, int num_runs,
               const std::function<pair<float, int>(CALCULATOR_COMMANDS *,
                                                    float *, int)> &func,
               CALCULATOR_COMMANDS *h_commands, float *h_operands,
               int num_commands) {
  for (int i = 0; i < warmup_runs; i++) {
    int errors;
    std::tie(std::ignore, errors) = func(h_commands, h_operands, num_commands);
    if (errors > 0) {
      printf("[%s] %d errors\n", name, errors);
      exit(1);
    }
  }

  double tot = 0;
  for (int i = 0; i < num_runs; i++) {
    float ms;
    std::tie(ms, std::ignore) = func(h_commands, h_operands, num_commands);
    tot += ms;
  }
  tot /= num_runs;
  printf("[%s] #streams: %d, %.6fms\n", name, NUM_STREAMS, tot);
}

int main(int argc, char **argv) {
  int num_commands;

  CALCULATOR_COMMANDS h_commands[MAX_COMMANDS];
  float h_operands[MAX_COMMANDS];

  // get calculator operators from file
  num_commands = readCommandsFromFile(h_commands, h_operands);

  printf("%d commands found in file\n", num_commands);

  // copy commands and operands to device
  cudaMemcpyToSymbol(d_commands, h_commands,
                     sizeof(CALCULATOR_COMMANDS) * MAX_COMMANDS);
  CheckCUDAError("Commands copy to constant memory");
  cudaMemcpyToSymbol(d_operands, h_operands, sizeof(float) * MAX_COMMANDS);
  CheckCUDAError("Commands copy to constant memory");

  for (int i : {2, 4, 8, 16}) {
    NUM_STREAMS = i;
    Benchmark("DefaultStream", 10, 100, cudaCalculatorDefaultStream, h_commands,
              h_operands, num_commands);
    Benchmark("NStream1", 10, 100, cudaCalculatorNStream1, h_commands,
              h_operands, num_commands);
    Benchmark("NStream2", 10, 100, cudaCalculatorNStream2, h_commands,
              h_operands, num_commands);
  }
}

pair<float, int> cudaCalculatorDefaultStream(CALCULATOR_COMMANDS *commands,
                                             float *operands,
                                             int num_commands) {
  float *h_input, *h_output;
  float *d_input, *d_output;
  float time;
  cudaEvent_t start, stop;
  int errors;

  // init cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory
  h_input = (float *)malloc(sizeof(float) * SAMPLES);
  h_output = (float *)malloc(sizeof(float) * SAMPLES);

  // allocate device memory
  cudaMalloc((void **)&d_input, sizeof(float) * SAMPLES);
  cudaMalloc((void **)&d_output, sizeof(float) * SAMPLES);
  CheckCUDAError("CUDA Memory allocate: default stream");

  // init the host input
  InitInput(h_input);

  // begin timing
  cudaEventRecord(start);

  // Stage 1) Synchronous host to device memory copy
  cudaMemcpy(d_input, h_input, sizeof(float) * SAMPLES, cudaMemcpyHostToDevice);
  CheckCUDAError("CUDA Memory copy H2D: default stream");

  // Stage 2) Execute kernel
  ParallelCalculator<<<SAMPLES / TPB, TPB>>>(d_input, d_output, num_commands);
  CheckCUDAError("CUDA Kernel: default stream");

  // Stage 3) Synchronous device to host memory copy
  cudaMemcpy(h_output, d_output, sizeof(float) * SAMPLES,
             cudaMemcpyDeviceToHost);
  CheckCUDAError("CUDA Memory copy D2H: default stream");

  // end timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // check for errors and print timing
  errors = CheckResults(h_input, h_output, commands, operands, num_commands);

  // cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);
  return {time, errors};
}

pair<float, int> cudaCalculatorNStream1(CALCULATOR_COMMANDS *commands,
                                        float *operands, int num_commands) {
  float *h_input, *h_output;
  float *d_input, *d_output;
  float time;
  cudaEvent_t start, stop;
  int i, errors;

  cudaStream_t *streams = new cudaStream_t[NUM_STREAMS];

  // init cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Exercise 2.1) Allocate GPU and CPU memory
  int size = sizeof(float) * SAMPLES;
  cudaMallocHost(&h_input, size);
  cudaMallocHost(&h_output, size);
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);

  // Exercise 2.2) Initialise the streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(streams + i);
    CheckCUDAError("cudaStreamCreate");
  }

  // init the host input
  InitInput(h_input);

  // begin timing
  cudaEventRecord(start);

  // Exercise 2.3) Loop through the streams and schedule a H2D copy, kernel
  // execution and D2H copy
  int batch_samples = SAMPLES / NUM_STREAMS;
  for (i = 0; i < NUM_STREAMS; i++) {
    // Stage 1) Asynchronous host to device memory copy
    float *d_input_i = d_input + batch_samples * i;
    float *d_output_i = d_output + batch_samples * i;
    float *h_input_i = h_input + batch_samples * i;
    float *h_output_i = h_output + batch_samples * i;

    cudaMemcpyAsync(d_input_i, h_input_i, size / NUM_STREAMS,
                    cudaMemcpyHostToDevice, streams[i]);
    CheckCUDAError("cudaMemcpyAsync");

    // Stage 2) Execute kernel
    ParallelCalculator<<<SAMPLES / NUM_STREAMS / TPB, TPB, 0, streams[i]>>>(
        d_input_i, d_output_i, num_commands);

    // Stage 3) Asynchronous device to host memory copy
    cudaMemcpyAsync(h_output_i, d_output_i, size / NUM_STREAMS,
                    cudaMemcpyHostToDevice, streams[i]);
    CheckCUDAError("cudaMemcpyAsync");
  }

  // end timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // check for errors and print timing
  errors = CheckResults(h_input, h_output, commands, operands, num_commands);

  // Exercise 2.4)
  // Cleanup by destroying each stream
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
    CheckCUDAError("cudaStreamDestroy");
  }
  delete[] streams;

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);

  return {time, errors};
}

pair<float, int> cudaCalculatorNStream2(CALCULATOR_COMMANDS *commands,
                                        float *operands, int num_commands) {
  float *h_input, *h_output;
  float *d_input, *d_output;
  float time;
  cudaEvent_t start, stop;
  int i, errors;
  cudaStream_t *streams = new cudaStream_t[NUM_STREAMS];

  // init cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // TODO: Allocate GPU and CPU memory
  int size = sizeof(float) * SAMPLES;
  cudaMallocHost(&h_input, size);
  cudaMallocHost(&h_output, size);
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);

  // TODO: Initialise the streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(streams + i);
    CheckCUDAError("cudaStreamCreate");
  }

  // init the host input
  InitInput(h_input);

  // begin timing
  cudaEventRecord(start);

  int batch_samples = SAMPLES / NUM_STREAMS;

#define P(ptr, i) ((ptr) + batch_samples * (i))
  for (i = 0; i < NUM_STREAMS; i++) {
    // Exercise 2.5) Asynchronous host to device memory copy
    cudaMemcpyAsync(P(d_input, i), P(h_input, i), size / NUM_STREAMS,
                    cudaMemcpyHostToDevice, streams[i]);
    CheckCUDAError("cudaMemcpyAsync");
  }

  for (i = 0; i < NUM_STREAMS; i++) {
    // Stage 2) Execute kernel
    ParallelCalculator<<<batch_samples / TPB, TPB, 0, streams[i]>>>(
        P(d_input, i), P(d_output, i), num_commands);
  }

  for (i = 0; i < NUM_STREAMS; i++) {
    // Stage 3) Asynchronous device to host memory copy
    cudaMemcpyAsync(P(h_output, i), P(d_output, i), size / NUM_STREAMS,
                    cudaMemcpyHostToDevice, streams[i]);
    CheckCUDAError("cudaMemcpyAsync");
  }
#undef P

  // end timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // check for errors and print timing
  errors = CheckResults(h_input, h_output, commands, operands, num_commands);

  // TODO: Cleanup by destroying each stream
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
    CheckCUDAError("cudaStreamDestroy");
  }
  delete[] streams;

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
  return {time, errors};
}

int readCommandsFromFile(CALCULATOR_COMMANDS *commands, float *operands) {
  FILE *f;
  float in_value;
  unsigned int line;
  char buffer[FILE_BUFFER_SIZE];
  char command[4];
  line = 0;

  printf("Welcome to the COM4521 Parallel floating point Calculator\n");
  f = fopen("commands.calc", "r");
  if (f == NULL) {
    fprintf(stderr, "File not found\n");
    return 0;
  }

  while (ReadLine(f, buffer)) {
    line++;

    if (line >= MAX_COMMANDS) {
      fprintf(stderr, "To many commands in form maximum is %u\n", MAX_COMMANDS);
      return 0;
    }

    if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) &&
          buffer[3] == ' ')) {
      fprintf(stderr, "Incorrect command format at line %u\n", line);
      return 0;
    }

    sscanf(buffer, "%s %f", command, &in_value);

    if (strcmp(command, "add") == 0) {
      commands[line] = CALCULATOR_ADD;
    } else if (strcmp(command, "sub") == 0) {
      commands[line] = CALCULATOR_SUB;
    } else if (strcmp(command, "div") == 0) {
      commands[line] = CALCULATOR_DIV;
    } else if (strcmp(command, "mul") == 0) {
      commands[line] = CALCULATOR_MUL;
    } else {
      fprintf(stderr, "Unknown command at line %u!\n", line);
      return 0;
    }

    operands[line] = in_value;
  }

  fclose(f);

  return line;
}

void InitInput(float *input) {
  for (int i = 0; i < SAMPLES; i++) {
    if (INPUT == INPUT_LINEAR)
      input[i] = (float)i;
    else if (INPUT == INPUT_RANDOM)
      input[i] = rand() / (float)RAND_MAX;
  }
}

int ReadLine(FILE *f, char buffer[]) {
  int i = 0;
  char c;
  while ((c = getc(f)) != '\n') {
    if (c == EOF) return 0;
    buffer[i++] = c;
    if (i == FILE_BUFFER_SIZE) {
      fprintf(stderr, "Buffer size is too small for line input\n");
      exit(0);
    }
  }
  buffer[i] = '\0';

  if (strncmp(buffer, "exit", 4) == 0)
    return 0;
  else
    return 1;
}

int CheckResults(float *h_input, float *h_output, CALCULATOR_COMMANDS *commands,
                 float *operands, int num_commands) {
  int i, j, errors;

  errors = 0;
  for (i = 0; i < SAMPLES; i++) {
    float out = h_input[i];
    for (j = 0; j < num_commands; j++) {
      CALCULATOR_COMMANDS cmd = commands[j];
      float v = operands[j];

      switch (cmd) {
        case (CALCULATOR_ADD): {
          out += v;
          break;
        }
        case (CALCULATOR_SUB): {
          out -= v;
          break;
        }
        case (CALCULATOR_DIV): {
          out /= v;
          break;
        }
        case (CALCULATOR_MUL): {
          out *= v;
          break;
        }
      }
    }
    // test the result
    if (h_output[i] != out) {
      // fprintf(stderr, "Error: GPU result (%f) differs from CPU result (%f) at
      // index %d\n", h_output[i], out, i);
      errors++;
    }
  }

  return errors;
}
