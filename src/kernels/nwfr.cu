#include <cstdio>

#include "matrix.cu"
#include "nwfr.h"
#include "shared.cu"
#include "sutil/vec_math.h"

static __forceinline__ __device__ uint xxhash32(const float2& p)
{
  const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
  const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
  uint h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
  h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
  h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
  return h32 ^ (h32 >> 16);
}

struct PCGState {
  unsigned long long state = 0;
  unsigned long long inc = 1;
};

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
static __forceinline__ __device__ __host__ uint pcg32_random_r(PCGState* rng)
{
  unsigned long long oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static __forceinline__ __device__ float funiform(PCGState& state)
{
  return pcg32_random_r(&state) * (1.0f / (1ULL << 32));
}

// Bitterli, Benedikt, et al. "Nonlinearly weighted first‐order regression for
// denoising Monte Carlo renderings." Computer Graphics Forum. Vol. 35. No. 4.
// 2016.
void __global__ nwfr_kernel(const float3* beauty, const float3* albedo,
                            const float3* normal, int width, int height,
                            float3* denoised)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;

  const int K = 8;
  const int P = 8;
  const float sigma_b = 32.0f;
  const float lambda = 0.01f;

  const int image_idx = i + width * j;
  const float3 b0 = beauty[image_idx];
  const float3 a0 = albedo[image_idx];
  const float3 n0 = 2.0f * normal[image_idx] - 1.0f;
  const float3 m0 = compute_albedo(a0);

  float s11 = 0.0f;
  float s12[6];
  float s22[6][6];

  PCGState state;
  state.state = xxhash32(make_float2(i, j));
  state.inc = 1;

  // compute s11, s12, s22
  // Takeda, Hiroyuki, Sina Farsiu, and Peyman Milanfar. "Kernel regression for
  // image processing and reconstruction." IEEE Transactions on image
  // processing 16.2 (2007): 349-366.
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

      const int idx = get_image_idx(i + u, j + v, width, height);
      const float3 b1 = beauty[idx];
      const float3 a1 = albedo[idx];
      const float3 n1 = 2.0f * normal[idx] - 1.0f;
      const float K = gaussian_kernel(dist, sigma_b);

      s11 += K;

      s12[0] = (a1 - a0).x * K;
      s12[1] = (a1 - a0).y * K;
      s12[2] = (a1 - a0).z * K;
      s12[3] = (n1 - n0).x * K;
      s12[4] = (n1 - n0).y * K;
      s12[5] = (n1 - n0).z * K;

      s22[0][0] = (a1 - a0).x * (a1 - a0).x * K;
      s22[0][1] = (a1 - a0).x * (a1 - a0).y * K;
      s22[0][2] = (a1 - a0).x * (a1 - a0).z * K;

      s22[0][3] = (a1 - a0).x * (n1 - n0).x * K;
      s22[0][4] = (a1 - a0).x * (n1 - n0).y * K;
      s22[0][5] = (a1 - a0).x * (n1 - n0).z * K;

      s22[1][0] = (a1 - a0).y * (a1 - a0).x * K;
      s22[1][1] = (a1 - a0).y * (a1 - a0).y * K;
      s22[1][2] = (a1 - a0).y * (a1 - a0).z * K;

      s22[1][3] = (a1 - a0).y * (n1 - n0).x * K;
      s22[1][4] = (a1 - a0).y * (n1 - n0).y * K;
      s22[1][5] = (a1 - a0).y * (n1 - n0).z * K;

      s22[2][0] = (a1 - a0).z * (a1 - a0).x * K;
      s22[2][1] = (a1 - a0).z * (a1 - a0).y * K;
      s22[2][2] = (a1 - a0).z * (a1 - a0).z * K;

      s22[2][3] = (a1 - a0).z * (n1 - n0).x * K;
      s22[2][4] = (a1 - a0).z * (n1 - n0).y * K;
      s22[2][5] = (a1 - a0).z * (n1 - n0).z * K;

      s22[3][0] = (n1 - n0).x * (a1 - a0).x * K;
      s22[3][1] = (n1 - n0).x * (a1 - a0).y * K;
      s22[3][2] = (n1 - n0).x * (a1 - a0).z * K;

      s22[3][3] = (n1 - n0).x * (n1 - n0).x * K;
      s22[3][4] = (n1 - n0).x * (n1 - n0).y * K;
      s22[3][5] = (n1 - n0).x * (n1 - n0).z * K;

      s22[4][0] = (n1 - n0).y * (a1 - a0).x * K;
      s22[4][1] = (n1 - n0).y * (a1 - a0).y * K;
      s22[4][2] = (n1 - n0).y * (a1 - a0).z * K;

      s22[4][3] = (n1 - n0).y * (n1 - n0).x * K;
      s22[4][4] = (n1 - n0).y * (n1 - n0).y * K;
      s22[4][5] = (n1 - n0).y * (n1 - n0).z * K;

      s22[5][0] = (n1 - n0).z * (a1 - a0).x * K;
      s22[5][1] = (n1 - n0).z * (a1 - a0).y * K;
      s22[5][2] = (n1 - n0).z * (a1 - a0).z * K;

      s22[5][3] = (n1 - n0).z * (n1 - n0).x * K;
      s22[5][4] = (n1 - n0).z * (n1 - n0).y * K;
      s22[5][5] = (n1 - n0).z * (n1 - n0).z * K;
    }
  }

  // ridge regularization
  s11 += lambda;
  for (int idx = 0; idx < 6; ++idx) { s22[idx][idx] += lambda; }

  // stochastic regularization
  // s11 += lambda * (2.0f * funiform(state) - 1.0f);
  // for (int idx = 0; idx < 6; ++idx) {
  //   s12[idx] += lambda;
  //   for (int idx2 = 0; idx2 < 6; ++idx2) { s22[idx][idx2] += lambda; }
  // }

  float s22_inv[6][6];
  float A[6][12];
  matrix_inverse(&s22[0][0], &A[0][0], 6, &s22_inv[0][0]);

  float temp0[6];
  matrix_vec_mult(&s22_inv[0][0], &s12[0], 6, 6, &temp0[0]);

  float temp1 = vec_dot(&s12[0], &temp0[0], 6);
  float divider = max(s11 - temp1, EPS);

  float3 sum = make_float3(0.0f);
  float3 sum_demodulated = make_float3(0.0f);
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

      const int idx = get_image_idx(i + u, j + v, width, height);
      const float3 b1 = beauty[idx];
      const float3 a1 = albedo[idx];
      const float3 n1 = 2.0f * normal[idx] - 1.0f;
      const float K = gaussian_kernel(dist, sigma_b);

      float x[6];
      x[0] = (a1 - a0).x;
      x[1] = (a1 - a0).y;
      x[2] = (a1 - a0).z;
      x[3] = (n1 - n0).x;
      x[4] = (n1 - n0).y;
      x[5] = (n1 - n0).z;

      float temp2[6];
      matrix_vec_mult(&s22_inv[0][0], &x[0], 6, 6, &temp2[0]);

      float temp3 = vec_dot(&s12[0], &temp2[0], 6);

      const float w = max(1.0f - temp3, EPS) * K / divider;

      if (is_ok_to_demodulate_albedo(b1, a1)) {
        sum_demodulated += w * reinhard(b1) / a1;
      } else {
        sum += w * reinhard(b1);
      }

      w_sum += w;
    }
  }
  w_sum += EPS;

  denoised[image_idx] = reinhard_inverse(sum / w_sum) +
                        reinhard_inverse(m0 * sum_demodulated / w_sum);
}

void __host__ nwfr_kernel_launch(const float3* beauty, const float3* albedo,
                                 const float3* normal, int width, int height,
                                 float3* denoised)
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(width / threads_per_block.x + 1,
                    height / threads_per_block.y + 1);
  nwfr_kernel<<<blocks, threads_per_block>>>(beauty, albedo, normal, width,
                                             height, denoised);
}