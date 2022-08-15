# cwl
add_library(cwl INTERFACE)
target_include_directories(cwl INTERFACE
    "include/"
)
target_link_libraries(cwl INTERFACE
    CUDA::cudart
)