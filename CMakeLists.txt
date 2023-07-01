cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(NNCPP CUDA CXX)

find_package( OpenCV REQUIRED )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-O3 --expt-relaxed-constexpr  --diag-suppress 20011 --diag-suppress 20012 --diag-suppress 20014")


add_subdirectory(DataLoaders)
add_executable(NNCPP
#        main.cu
        NN/Models/NNSequentialModel.cu NN/Models/NNSequentialModel.cuh
        NN/Layers/NNDenseLayer.cu NN/Layers/NNDenseLayer.cuh
        NN/Layers/NNInputLayer.cu NN/Layers/NNInputLayer.cuh NN/Layers/NNLayer.cuh
        #        DataLoaders/ImageData.h
        #        NN/example.cu

        #        external/Eigen/Cholesky
        #        external/Eigen/Dense
        external/Eigen/Core
        NN/Utils/Logger.cuh
        mnist_digits.cu
        #        external/unsupported/Eigen/CXX11/Tensor
)

#set_target_properties(NNCPP PROPERTIES LINKER_LANGUAGE CXX)
target_precompile_headers(NNCPP PUBLIC pch.cuh)
# copy data directory

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${PROJECT_SOURCE_DIR}/data
        ${PROJECT_BINARY_DIR}/data
        COMMENT "Copying resources into binary directory")

add_dependencies(NNCPP copy_resources)


include_directories(external)
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( NNCPP ${OpenCV_LIBS} )
set_target_properties(NNCPP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
