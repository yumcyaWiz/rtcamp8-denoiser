#include <memory>
#include <stdexcept>
#include <vector>

#include "argparse/argparse.hpp"
#include "bilateral.h"
#include "cwl/buffer.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sutil/vec_math.h"

std::vector<float3> load_hdr_image(const std::string &filepath, int &width,
                                   int &height)
{
  int c;
  const float *image = stbi_loadf(filepath.c_str(), &width, &height, &c, 3);
  if (!image) {
    std::cerr << "failed to load " + filepath << std::endl;
    std::exit(1);
  }

  std::vector<float3> ret(width * height);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      const int idx = 3 * i + 3 * width * j;
      ret[i + width * j] =
          make_float3(image[idx + 0], image[idx + 1], image[idx + 2]);
    }
  }

  return ret;
}

void save_hdr_image(const std::string &filepath, int width, int height,
                    const std::vector<float3> &image)
{
  if (!stbi_write_hdr(filepath.c_str(), width, height, 3,
                      reinterpret_cast<const float *>(image.data()))) {
    std::cerr << "failed to save " + filepath << std::endl;
    std::exit(1);
  }
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser args("main");
  args.add_argument("beauty").help("beauty image input filepath");
  args.add_argument("albedo").help("albedo image input filepath");
  args.add_argument("normal").help("normal image input filepath");
  args.add_argument("denoised").help("denoised image output filepath");

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    std::exit(1);
  }

  // parse args
  const std::string beauty_filepath = args.get<std::string>("beauty");
  const std::string albedo_filepath = args.get<std::string>("albedo");
  const std::string normal_filepath = args.get<std::string>("normal");
  const std::string denoised_filepath = args.get<std::string>("denoised");

  // load input images on host
  int width, height;
  const std::vector<float3> beauty =
      load_hdr_image(beauty_filepath, width, height);
  const std::vector<float3> albedo =
      load_hdr_image(albedo_filepath, width, height);
  const std::vector<float3> normal =
      load_hdr_image(normal_filepath, width, height);

  // load input images on host
  const auto beauty_d = std::make_unique<cwl::CUDABuffer<float3>>(beauty);
  const auto albedo_d = std::make_unique<cwl::CUDABuffer<float3>>(albedo);
  const auto normal_d = std::make_unique<cwl::CUDABuffer<float3>>(normal);

  // prepare output image on host
  const auto denoised_d =
      std::make_unique<cwl::CUDABuffer<float3>>(width * height);

  // launch denoiser
  bilateral_kernel_launch(
      beauty_d->get_device_ptr(), albedo_d->get_device_ptr(),
      normal_d->get_device_ptr(), width, height, denoised_d->get_device_ptr());

  // save denoised image
  std::vector<float3> denoised;
  denoised_d->copy_from_device_to_host(denoised);
  save_hdr_image(denoised_filepath, width, height, denoised);

  return 0;
}