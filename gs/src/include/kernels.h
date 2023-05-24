#pragma once
#include "common.h"
#include "helper_math.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

__device__ void transpose(float3 *m, float3 *mt) {
  mt[0] = make_float3(m[0].x, m[1].x, m[2].x);
  mt[1] = make_float3(m[0].y, m[1].y, m[2].y);
  mt[2] = make_float3(m[0].z, m[1].z, m[2].z);
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

__device__ void qvec2rotmat(float4 &qvec, float3 &svec, float3 *m) {
  float w = qvec.x, x = qvec.y, y = qvec.z, z = qvec.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)) * svec;
  m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)) * svec;
  m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)) * svec;
}

__device__ void qvec2rotmat(float4 &qvec, float3 *m) {
  float w = qvec.x, x = qvec.y, y = qvec.z, z = qvec.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy));
  m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx));
  m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));
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
}

__device__ void qvec2rotmat_col_major(float4 &qvec, float3 &svec, float3 *m) {
  // get rot matrix from qvec and svec, but store into m with column major
  float w = qvec.x, x = qvec.y, y = qvec.z, z = qvec.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;
  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
  m[1] = make_float3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
  m[2] = make_float3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));
}

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
  transpose(sigma, sigma_inv);
  float det_sqr = sqrt(svec[0].x * svec[0].y * svec[0].z);

  return gaussian_kernel_3d_with_inv(mean, sigma_inv, query) / det_sqr;
}

// __device__ void covariance2bbox(float3 &mean, float4 &qvec, float3 &svec,
//                                 bbox3d &bbox, float thresh = 3.0f) {
//   float3 rotmat[3];
//   qvec2rotmat(qvec, rotmat);
// #pragma unroll
//   for (int i = 0; i < 8; ++i) {
//     int x = -1 + 2 * (i & 1);
//     int y = -1 + 2 * (i & (1 << 1));
//     int z = -1 + 2 * (i & (1 << 2));
//     bbox.vertices[i] =
//         mean + (x * rotmat[0] + y * rotmat[1] + z * rotmat[2]) * svec *
//         thresh;
//   }
//   return;
// }

__device__ bool point_in_frustum(float3 &query, float3 *normal, float3 *pts) {
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    if (dot(query - pts[i], normal[i]) < -EPS)
      return false;
  }
  return true;
}

// __device__ bool onorforward_bbox3d_plane(bbox3d &bbox, float3 *normal,
//                                          float3 *pts) {
//   assert(true);
//   return false;
// }

// __device__ bool intersect_bbox3d_frustum(bbox3d &bbox, float3 *normal,
//                                          float3 *pts) {
// #pragma unroll
//   for (int i = 0; i < 8; ++i) {
//     if (point_in_frustum(bbox.vertices[i], normal, pts))
//       return true;
//   }
//   return false;
// }

__device__ bool onorforward_sphere_plane(float3 &query, float radius,
                                         float3 &normal, float3 &pts) {
  return dot(query - pts, normal) > -radius;
}

__device__ bool intersect_sphere_frustum(float3 &query, float radius,
                                         float3 *normal, float3 *pts) {
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    if (!onorforward_sphere_plane(query, radius, normal[i], pts[i])) {
      return false;
    }
  }
  return true;
}