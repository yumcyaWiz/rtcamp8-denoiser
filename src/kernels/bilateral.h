#pragma once

#include "sutil/vec_math.h"

void __global__ bilateral_kernel(const float3* beauty, int width, int height,
                                 float3* denoised);

void __global__ joint_bilateral_kernel(const float3* beauty,
                                       const float3* albedo,
                                       const float3* normal, int width,
                                       int height, float3* denoised);

void __host__ bilateral_kernel_launch(const float3* beauty,
                                      const float3* albedo,
                                      const float3* normal, int width,
                                      int height, float3* denoised);

void __host__ joint_bilateral_kernel_launch(const float3* beauty,
                                            const float3* albedo,
                                            const float3* normal, int width,
                                            int height, float3* denoised);