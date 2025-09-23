cmake_minimum_required(VERSION 3.17)
project(COM4521Labs LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(lab6 "com4521-Lab06_src/exercise01.cu")
target_include_directories(lab6 PRIVATE "include")
target_link_libraries(lab6 CUDA::cublas)
add_executable(benchmark "com4521-Lab06_src/benchmark.cu")
target_include_directories(benchmark PRIVATE "include")
target_link_libraries(benchmark CUDA::toolkit CUDA::cudart)