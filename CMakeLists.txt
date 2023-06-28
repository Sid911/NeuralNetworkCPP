cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_package( OpenCV REQUIRED )
project(NNCPP CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-O3 --expt-relaxed-constexpr  --diag-suppress 20011 --diag-suppress 20012 --diag-suppress 20014")

add_executable(NNCPP main.cu
        NN/Models/NNSequentialModel.cu NN/Models/NNSequentialModel.cuh
        NN/Layers/NNDenseLayer.cu NN/Layers/NNDenseLayer.cuh
        NN/Layers/NNInputLayer.cu NN/Layers/NNInputLayer.cuh NN/Layers/NNLayer.cuh
#        NN/example.cu

#        external/Eigen/Cholesky
#        external/Eigen/Dense
        external/Eigen/Core
        )

include_directories(external)
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( NNCPP ${OpenCV_LIBS} )
set_target_properties(NNCPP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
