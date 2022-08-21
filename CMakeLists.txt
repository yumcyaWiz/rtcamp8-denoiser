cmake_minimum_required(VERSION 3.20)
project(fredholm LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ version and std
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# CUDA C++ version and std
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# cwl
add_subdirectory("cwl/")

# external
add_subdirectory("externals/")

# kernels
add_library(kernels
    "src/kernels/bilateral.cu"
    "src/kernels/nl-means.cu"
    "src/kernels/a-trous.cu"
)
target_include_directories(kernels PUBLIC
    "src/kernels"
)
target_link_libraries(kernels PUBLIC
    cwl
    sutil
)

# test
add_executable(test "src/test.cpp")
target_link_libraries(test PRIVATE
    CUDA::cudart
    argparse
    stb_image
    stb_image_write
    sutil
    cwl
    kernels
)