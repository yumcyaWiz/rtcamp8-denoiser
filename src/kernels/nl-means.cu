#include "nl-means.h"
#include "shared.cu"

void __global__ nlmeans_kernel(const float3* beauty, int width, int height,
                               float3* denoised)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;

  const int K = 8;
  const int P = 7;
  const float sigma_b = 128.0f;
  const float sigma_p = 16.0f;

  const int image_idx = i + width * j;
  const float3 b0 = beauty[image_idx];

  float3 b_sum = make_float3(0.0f);
  float w_sum = 0.0f;

  for (int v = -K; v <= K; ++v) {
    for (int u = -K; u <= K; ++u) {
      float dist = 0.0f;
      for (int t = -P; t <= P; ++t) {
        for (int s = -P; s <= P; ++s) {
          const float3 t0 = beauty[get_image_idx(i + s, j + t, width, height)];
          const float3 t1 =
              beauty[get_image_idx(i + u + s, j + v + t, width, height)];
          dist += length(t0 - t1);
        }
      }
      dist /= (P * P);

      const float3 b1 = beauty[get_image_idx(i + u, j + v, width, height)];

      const float w_b = gaussian_kernel(dist, sigma_b);
      const float w_p = gaussian_kernel(sqrtf(u * u + v * v), sigma_p);
      const float w = w_b * w_p;

      b_sum += w * reinhard(b1);
      w_sum += w;
    }
  }
  w_sum += EPS;

  denoised[image_idx] = reinhard_inverse(b_sum / w_sum);
}

void __host__ nlmeans_kernel_launch(const float3* beauty, const float3* albedo,
                                    const float3* normal, int width, int height,
                                    float3* denoised)
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(width / threads_per_block.x + 1,
                    height / threads_per_block.y + 1);
  nlmeans_kernel<<<blocks, threads_per_block>>>(beauty, width, height,
                                                denoised);
}