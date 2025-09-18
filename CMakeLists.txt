cmake_minimum_required(VERSION 3.17)
project(COM4521Labs LANGUAGES CXX CUDA)

add_executable(lab6 "com4521-Lab06_src/exercise01.cu")
target_include_directories(lab6 PRIVATE "include")
target_link_libraries(lab6 "cublas")