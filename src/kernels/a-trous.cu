void __global__ a_trous_kernel(const float3* beauty, const float3* albedo,
                               const float3* normal, int width, int height,
                               float3* denoised)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;
}

void __host__ a_trous_kernel_launch(const float3* beauty, const float3* albedo,
                                    const float3* normal, int width, int height,
                                    float3* denoised)
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(max(width / threads_per_block.x, 1),
                    max(height / threads_per_block.y, 1));
  a_trous_kernel<<<blocks, threads_per_block>>>(beauty, albedo, normal, width,
                                                height, denoised);
}