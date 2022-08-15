#pragma once
#include "sutil/vec_math.h"

#define EPS 0.001f

static __forceinline__ __device__ int get_image_idx(int i, int j, int width,
                                                    int height)
{
  return clamp(i, 0, width - 1) + width * clamp(j, 0, height - 1);
}

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
static __forceinline__ __device__ float rgb_to_luminance(const float3& rgb)
{
  return dot(rgb, make_float3(0.2126729f, 0.7151522f, 0.0721750f));
}

// Schied, Christoph, et al. "Spatiotemporal variance-guided filtering:
// real-time reconstruction for path-traced global illumination." Proceedings of
// High Performance Graphics. 2017. 1-12.
static __forceinline__ __device__ float luminance_weight(float l0, float l1,
                                                         float sigma)
{
  return expf(-abs(l0 - l1) / (sigma * 1.0f + EPS));
}

// Schied, Christoph, et al. "Spatiotemporal variance-guided filtering:
// real-time reconstruction for path-traced global illumination." Proceedings of
// High Performance Graphics. 2017. 1-12.
static __forceinline__ __device__ float normal_weight(const float3& n0,
                                                      const float3& n1,
                                                      float sigma)
{
  return powf(max(dot(n0, n1), 0.0f), sigma);
}