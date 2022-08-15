# stb
add_library(stb_image
    "stb_image.cpp"
)
target_include_directories(stb_image PUBLIC
    "stb/"
)