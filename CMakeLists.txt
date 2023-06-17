cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(CudaTest CUDA)
set(CMAKE_CUDA_STANDARD 17)

add_executable(CudaTest main.cu)

set_target_properties(CudaTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
