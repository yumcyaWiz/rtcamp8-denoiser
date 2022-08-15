#pragma once

#include "sutil/vec_math.h"

void __global__ a_trous_kernel(const float3* beauty, const float3* albedo,
                               const float3* normal, int width, int height,
                               int level, float3* denoised);

void __host__ a_trous_kernel_launch(const float3* beauty, const float3* albedo,
                                    const float3* normal, int width, int height,
                                    float3* denoised);