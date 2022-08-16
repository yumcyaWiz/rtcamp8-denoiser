#include "a-trous.h"
#include "cwl/buffer.h"
#include "cwl/util.h"
#include "shared.cu"
#include "sutil/vec_math.h"

void __global__ a_trous_kernel(const float3* beauty, const float3* albedo,
                               const float3* normal, int width, int height,
                               int level, float3* denoised)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;

  const float sigma_rt = 16.0f * powf(2, -level);
  const float sigma_n = 128.0f;
  const float sigma_a = 0.01f;

  const int image_idx = i + width * j;
  const float3 b0 = beauty[image_idx];
  const float3 a0 = albedo[image_idx];
  const float3 n0 = 2.0f * normal[image_idx] - 1.0f;

  const float filter[] = {1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f,
                          1.0f / 16.0f};

  float3 b_sum = make_float3(0.0f);
  float w_sum = 0.0f;
  const int step_width = pow(2, level);
  for (int v = -2; v <= 2; ++v) {
    for (int u = -2; u <= 2; ++u) {
      const int idx =
          get_image_idx(i + step_width * u, j + step_width * v, width, height);
      const float3 b1 = beauty[idx];
      const float3 a1 = albedo[idx];
      const float3 n1 = 2.0f * normal[idx] - 1.0f;

      const float h = filter[max(u + 2, v + 2)];
      const float w_rt = gaussian_kernel(length(b0 - b1), sigma_rt);
      const float wn = normal_weight(n0, n1, sigma_n);
      const float wa = albedo_weight(a0, a1, sigma_a);
      const float w = h * wn * wa;

      b_sum += w * reinhard(b1);
      w_sum += w;
    }
  }
  w_sum += EPS;

  denoised[image_idx] = reinhard_inverse(b_sum / w_sum);
}

void __host__ a_trous_kernel_launch(const float3* beauty, const float3* albedo,
                                    const float3* normal, int width, int height,
                                    float3* denoised)
{
  const int N_iter = 1;
  const int N_level = 3;

  cwl::CUDABuffer<float3> temp0(width * height);
  temp0.copy_from_device_to_device(beauty);
  cwl::CUDABuffer<float3> temp1(width * height);

  const dim3 threads_per_block(16, 16);
  const dim3 blocks(width / threads_per_block.x + 1,
                    height / threads_per_block.y + 1);
  for (int k = 0; k < N_iter; ++k) {
    for (int i = 0; i < N_level; ++i) {
      const float3* in =
          ((i + k) % 2 == 0) ? temp0.get_device_ptr() : temp1.get_device_ptr();
      float3* out =
          ((i + k) % 2 == 0) ? temp1.get_device_ptr() : temp0.get_device_ptr();
      a_trous_kernel<<<blocks, threads_per_block>>>(in, albedo, normal, width,
                                                    height, i, out);
      CUDA_SYNC_CHECK();
    }
  }

  const float3* out = ((N_iter + N_level) % 2 == 0) ? temp1.get_device_ptr()
                                                    : temp0.get_device_ptr();
  cudaMemcpy(denoised, out, width * height * sizeof(float3),
             cudaMemcpyDeviceToDevice);
  CUDA_SYNC_CHECK();
}