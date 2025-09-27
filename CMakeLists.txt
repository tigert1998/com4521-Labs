cmake_minimum_required(VERSION 3.17)
project(COM4521Labs LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(benchmark "com4521-Lab06_src/benchmark.cu")
target_include_directories(benchmark PRIVATE "include" "com4521-Lab06_src")
target_link_libraries(benchmark CUDA::toolkit CUDA::cudart CUDA::cublas)