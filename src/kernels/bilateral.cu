#include "shared.cu"
#include "sutil/vec_math.h"

void __global__ bilateral_kernel(const float3* beauty, const float3* albedo,
                                 const float3* normal, int width, int height,
                                 float3* denoised)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;

  const int K = 8;
  const float sigma_l = 4.0f;
  const float sigma_n = 128.0f;

  const int image_idx = i + width * j;
  const float3 b0 = beauty[image_idx];
  const float3 a0 = albedo[image_idx];
  const float3 n0 = normal[image_idx];

  float3 b_sum = make_float3(0.0f);
  float w_sum = 0.0f;
  for (int v = 0; v < K; ++v) {
    for (int u = 0; u < K; ++u) {
      const int idx = get_image_idx(i + u, j + v, width, height);
      const float3 b1 = beauty[idx];
      const float3 a1 = albedo[idx];
      const float3 n1 = normal[idx];

      const float wl =
          luminance_weight(rgb_to_luminance(b0), rgb_to_luminance(b1), sigma_l);
      const float wn = normal_weight(n0, n1, sigma_n);
      const float w = wl * wn;

      b_sum += w * b1;
      w_sum += w;
    }
  }

  denoised[image_idx] = b_sum / w_sum;
}

void __host__ bilateral_launch(const float3* beauty, const float3* albedo,
                               const float3* normal, int width, int height,
                               float3* denoised)
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(max(width / threads_per_block.x, 1),
                    max(height / threads_per_block.y, 1));
  bilateral_kernel<<<blocks, threads_per_block>>>(beauty, albedo, normal, width,
                                                  height, denoised);
}