# stb_image
add_library(stb_image
    "stb_image.cpp"
)
target_include_directories(stb_image PUBLIC
    "stb/"
)

# stb_image_write
add_library(stb_image_write
    "stb_image_write.cpp"
)
target_include_directories(stb_image_write PUBLIC
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