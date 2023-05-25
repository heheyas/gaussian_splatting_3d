#pragma once
#include "common.h"
#include "data_spec.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include <stdint.h>

__global__ void culling_gaussian_bsphere_kernel(uint32_t N, float3 *mean,
                                                float4 *qvec, float3 *svec,
                                                float3 *normal, float3 *pts,
                                                bool *mask, float thresh) {
  tid_1d(tid);
  if (tid >= N)
    return;
  float r = fmaxf(fmaxf(svec[tid].x, svec[tid].y), svec[tid].z) * thresh;
  mask[tid] = intersect_sphere_frustum(mean[tid], r, normal, pts);
}

// __global__ void culling_gaussian_bbox_kernel(float3 *mean, float4 *qvec,
//                                              float3 *svec, float3 *normal,
//                                              float3 *pts, bool *mask,
//                                              float thresh) {
//   tid_1d(tid);
//   // mean += tid;
//   // qvec += tid;
//   // svec += tid;
//   // mask += tid;

//   if (point_in_frustum(mean[tid], normal, pts)) {
//     mask[tid] = true;
//   } else {
//     bbox3d bbox;
//     covariance2bbox(mean[tid], qvec[tid], svec[tid], bbox, thresh);
//     if (intersect_bbox3d_frustum(bbox, normal, pts))
//       *mask = true;
//     else
//       *mask = false;
//   }
// }

// void culling_gaussian_bbox(float *mean, float *qvec, float *svec, float
// *normal,
//                            float *pts, bool *mask) {
//   float3 *mean_ = reinterpret_cast<float3 *>(mean);
//   float4 *qvec_ = reinterpret_cast<float4 *>(qvec);
//   float3 *svec_ = reinterpret_cast<float3 *>(svec);
//   float3 *normal_ = reinterpret_cast<float3 *>(normal);
//   float3 *pts_ = reinterpret_cast<float3 *>(pts);
// }

void culling_gaussian_bsphere_cuda(uint32_t N, float *mean, float *qvec,
                                   float *svec, float *normal, float *pts,
                                   bool *mask, float thresh) {
  float3 *mean_ = reinterpret_cast<float3 *>(mean);
  float4 *qvec_ = reinterpret_cast<float4 *>(qvec);
  float3 *svec_ = reinterpret_cast<float3 *>(svec);
  float3 *normal_ = reinterpret_cast<float3 *>(normal);
  float3 *pts_ = reinterpret_cast<float3 *>(pts);

  uint32_t n_blocks = div_round_up(N, (uint32_t)N_THREADS);
  culling_gaussian_bsphere_kernel<<<n_blocks, N_THREADS>>>(
      N, mean_, qvec_, svec_, normal_, pts_, mask, thresh);
}