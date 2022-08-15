# stb_image
add_library(stb_image
    "stb_image.cpp"
)
target_include_directories(stb_image PUBLIC
    "stb/"
)

# argparse
add_library(argparse INTERFACE)
target_include_directories(argparse INTERFACE
    "argparse/include"
)

# sutil
add_library(sutil INTERFACE)
target_include_directories(sutil INTERFACE
    "sutil/"
)