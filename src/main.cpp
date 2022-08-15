#include "argparse/argparse.hpp"
#include <cstdio>
#include <stdexcept>

int main(int argc, char *argv[]) {
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

  const std::string beauty = args.get<std::string>("beauty");
  const std::string albedo = args.get<std::string>("albedo");
  const std::string normal = args.get<std::string>("normal");
  const std::string denoised = args.get<std::string>("denoised");

  return 0;
}