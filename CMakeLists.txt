cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(NNCPP CUDA)
set(CMAKE_CUDA_STANDARD 17)

add_executable(NNCPP main.cu
        NN/Models/NNSequentialModel.cu NN/Models/NNSequentialModel.cuh
        NN/Layers/NNDenseLayer.cu NN/Layers/NNDenseLayer.cuh
        NN/Layers/NNInputLayer.cu NN/Layers/NNInputLayer.cuh NN/Layers/NNLayer.cuh

        external/Eigen/Cholesky
        external/Eigen/Dense
        external/Eigen/Core
        )

include_directories(external)
set_target_properties(NNCPP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
