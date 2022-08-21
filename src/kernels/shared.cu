#pragma once
#include "sutil/vec_math.h"

#define EPS 0.001f

static __forceinline__ __device__ int get_image_idx(int i, int j, int width,
                                                    int height)
{
  return clamp(i, 0, width - 1) + width * clamp(j, 0, height - 1);
}

static __forceinline__ __device__ float3 compute_albedo(const float3& albedo,
                                                        const float3& beauty)
{
  float3 ret = albedo;
  if (ret.x == 0 || ret.y == 0 || ret.z == 0) { ret = beauty; }

  if (ret.x == 0 || ret.y == 0 || ret.z == 0) { return make_float3(1.0f); }

  return ret;
}

static __forceinline__ __device__ float3 compute_albedo2(const float3& albedo)
{
  float3 ret = albedo;
  if (ret.x == 0 || ret.y == 0 || ret.z == 0) { return make_float3(1.0f); }

  return ret;
}

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
static __forceinline__ __device__ float rgb_to_luminance(const float3& rgb)
{
  return dot(rgb, make_float3(0.2126729f, 0.7151522f, 0.0721750f));
}

static __forceinline__ __device__ float3 reinhard(const float3& rgb)
{
  const float l = rgb_to_luminance(rgb);
  return rgb * 1.0f / (1.0f + l);
}

static __forceinline__ __device__ float3 reinhard_inverse(const float3& rgb)
{
  const float l = min(max(rgb_to_luminance(rgb), 0.01f), 0.99f);
  return rgb * 1.0f / (1.0f - l);
}

static __forceinline__ __device__ bool is_ok_to_demodulate_albedo(
    const float3& beauty, const float3& albedo)
{
  // NOTE: 0.0f gives some black dots at the edges
  if (max(max(albedo.x, albedo.y), albedo.z) < 0.1f) { return false; }

  if (albedo.x <= 0.0f || albedo.y <= 0.0f || albedo.z <= 0.0f) {
    return false;
  }

  const float3 m = reinhard(beauty) / albedo;
  const float m_avg = (m.x + m.y + m.z) / 3.0f;
  const float threshold = 0.1f;
  return abs(m.x - m_avg) < threshold && abs(m.y - m_avg) < threshold &&
         abs(m.y - m_avg) < threshold;
}

static __forceinline__ __device__ float gaussian_kernel(float x, float sigma)
{
  return expf(-(x * x) / (2.0f * sigma));
}

static __forceinline__ __device__ float beauty_weight(const float3& b0,
                                                      const float3& b1,
                                                      float sigma)
{
  const float l = length(b0 - b1);
  return expf(-(l * l) / (2.0f * sigma));
}

// Schied, Christoph, et al. "Spatiotemporal variance-guided filtering:
// real-time reconstruction for path-traced global illumination." Proceedings of
// High Performance Graphics. 2017. 1-12.
static __forceinline__ __device__ float luminance_weight(float l0, float l1,
                                                         float sigma)
{
  return expf(-((l0 - l1) * (l0 - l1)) / (2.0f * sigma));
}

static __forceinline__ __device__ float albedo_weight(const float3& a0,
                                                      const float3& a1,
                                                      float sigma)
{
  const float l = length(a0 - a1);
  return expf(-(l * l) / (2.0f * sigma));
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