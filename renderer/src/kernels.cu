#include "helper_math.h"
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#define tem template <typename scalar_t>

const float gs_coeff_3d = 0.06349363593424097f; // 1 / (2 * pi) ** (3/2)
const float gs_coeff_2d = 0.15915494309189535f; // 1 / (2 * pi)

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

__device__ void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

__device__ void swap(float3 &a, float3 &b) {
  float3 temp = a;
  a = b;
  b = temp;
}

__device__ void inverse3x3(float3 *mat, float3 *inv) {
  // Calculate matrix of minors
  float3 m[3];
  m[0] = make_float3(mat[1].y * mat[2].z - mat[1].z * mat[2].y,
                     mat[1].x * mat[2].z - mat[1].z * mat[2].x,
                     mat[1].x * mat[2].y - mat[1].y * mat[2].x);
  m[1] = make_float3(mat[0].y * mat[2].z - mat[0].z * mat[2].y,
                     mat[0].x * mat[2].z - mat[0].z * mat[2].x,
                     mat[0].x * mat[2].y - mat[0].y * mat[2].x);
  m[2] = make_float3(mat[0].y * mat[1].z - mat[0].z * mat[1].y,
                     mat[0].x * mat[1].z - mat[0].z * mat[1].x,
                     mat[0].x * mat[1].y - mat[0].y * mat[1].x);

  float det = mat[0].x * m[0].x - mat[0].y * m[1].x + mat[0].z * m[2].x;

  float3 c[3];
  c[0] = make_float3(m[0].x, -m[1].x, m[2].x);
  c[1] = make_float3(-m[0].y, m[1].y, -m[2].y);
  c[2] = make_float3(m[0].z, -m[1].z, m[2].z);

  for (int i = 0; i < 3; i++) {
    swap(c[i].y, c[i].x);
    swap(c[i].z, c[i].x);
  }

  for (int i = 0; i < 3; i++) {
    inv[IDX2C(i, 0, 3)] = make_float3(c[0].x, c[1].x, c[2].x) / det;
    inv[IDX2C(i, 1, 3)] = make_float3(c[0].y, c[1].y, c[2].y) / det;
    inv[IDX2C(i, 2, 3)] = make_float3(c[0].z, c[1].z, c[2].z) / det;
  }
}

__device__ void qvec2rotmat(float4 *qvec, float3 *svec, float3 *m) {
  float w = qvec->x, x = qvec->y, y = qvec->z, z = qvec->w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)) * svec[0];
  m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)) * svec[0];
  m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)) * svec[0];
}

__device__ void qvec2rotmat_col_major(float4 *qvec, float3 *svec, float3 *m) {
  // get rot matrix from qvec and svec, but store into m with column major
  float w = qvec->x, x = qvec->y, y = qvec->z, z = qvec->w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
  m[1] = make_float3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
  m[2] = make_float3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));

  // m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)) * svec[0];
  // m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)) * svec[0];
  // m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)) * svec[0];
}

__device__ void get_major_axis(float4 &qvec, float3 &svec, float3 *ma) {}

__device__ float gaussian_kernel_3d_with_inv(float3 *mean, float3 *sigma_inv,
                                             float3 *query) {
  // float3 *sigam_inv_row = reinterpret_cast<float3 *>(sigma_inv);
  float3 x = query[0] - mean[0];
  float3 tmp = make_float3(dot(x, sigma_inv[0]), dot(x, sigma_inv[1]),
                           dot(x, sigma_inv[2]));

  return __expf(-0.5f * dot(x, tmp)) * gs_coeff_3d;
}

__device__ float gaussian_kernel_3d(float3 *mean, float4 *qvec, float3 *svec,
                                    float3 *query) {
  float3 sigma_inv[3], sigma[3];
  qvec2rotmat(qvec, svec, sigma);
  inverse3x3(sigma, sigma_inv);
  float det_sqr = sqrt(svec[0].x * svec[0].y * svec[0].z);

  return gaussian_kernel_3d_with_inv(mean, sigma_inv, query) / det_sqr;
}