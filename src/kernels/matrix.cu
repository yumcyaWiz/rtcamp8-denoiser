#pragma once

static __forceinline__ __device__ float vec_dot(float* v0, float* v1, int n)
{
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) { sum += v0[i] * v1[i]; }
  return sum;
}

static __forceinline__ __device__ void matrix_vec_mult(float* matrix,
                                                       float* vector,
                                                       int n_rows, int n_cols,
                                                       float* dst)
{
  for (int i = 0; i < n_rows; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < n_cols; ++j) {
      sum += matrix[i * n_cols + j] * vector[j];
    }
    dst[i] = sum;
  }
}

static __forceinline__ __device__ void vec_matrix_mult(float* vector,
                                                       float* matrix,
                                                       int n_rows, int n_cols,
                                                       float* dst)
{
  for (int j = 0; j < n_cols; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < n_rows; ++i) {
      sum += vector[i] * matrix[i * n_cols + j];
    }
    dst[j] = sum;
  }
}

static __forceinline__ __device__ void matrix_matrix_mult(float* m0, float* m1,
                                                          int m, int n, int l,
                                                          float* dst)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < l; ++k) { sum += m0[i * n + k] * m1[k * n + j]; }
      dst[i * n + j] = sum;
    }
  }
}

static __forceinline__ __device__ void matrix_inverse(float* m, float* A, int n,
                                                      float* dst)
{
  // gauss jordan method
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i * (2 * n) + j] = m[i * n + j];
      if (i == j) { A[i * (2 * n) + j] += 0.001f; }
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i * (2 * n) + j + n] = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (int k = 0; k < n; ++k) {
    float a_kk = A[k * (2 * n) + k];
    for (int j = 0; j < 2 * n; ++j) { A[k * (2 * n) + j] /= a_kk; }

    for (int i = 0; i < n; ++i) {
      if (i == k) continue;

      double a_ik = A[i * (2 * n) + k];
      for (int j = 0; j < 2 * n; ++j) {
        A[i * (2 * n) + j] -= A[k * (2 * n) + j] * a_ik;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) { dst[i * n + j] = A[i * (2 * n) + j + n]; }
  }
}
