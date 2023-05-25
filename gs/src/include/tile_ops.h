#pragma once
#include "common.h"
#include "data_spec.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include <device_atomic_functions.h>
#include <stdint.h>

// __global__ void count_tiled_gaussians_kernel(
//     float2 *mean, float2 *cov_inv, float *topleft_, float pixel_size_x,
//     float pixel_size_y, uint32_t n_tiles_h, uint32_t n_tiles_w,
//     uint32_t tile_size, int *num_gaussians, float thresh) {
//   tid_1d(tid);
//   uint32_t tile_x = blockIdx.y;
//   uint32_t tile_y = blockIdx.z;
//   uint32_t tile_id = tile_x * n_tiles_h + tile_y;
//   // float2 topleft = mat
//   if (intersect_tile_gaussian2d(reinterpret_cast<float2 *>(topleft)[0],
//                                 tile_size, pixel_size_x, pixel_size_y,
//                                 mean[tid], cov_inv + (2 * tid, thresh))) {
//     atomicAdd(num_gaussians + tile_id, 1);
//   }
//   return;
// }

__global__ void count_tiled_gaussians_kernel_sm(
    uint32_t N, float *mean, float *cov_inv, float *topleft, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, int *num_gaussians, float thresh) {
  tid_1d(global_id);
  int local_id = threadIdx.x;
  int n_turns = (2 * N) / blockDim.x;
  int n_left = (2 * N) % blockDim.x;

  // load data onto scatchpad mem
  extern __shared__ float g_mean[];
  float *g_cov_inv = g_mean + 2 * N;

  for (int i = 0; i < n_turns; ++i) {
    g_mean[local_id] = mean[local_id];
  }
  if (local_id < n_left) {
    g_mean[local_id] = mean[local_id];
  }

  n_turns = (4 * N) / blockDim.x;
  n_left = (4 * N) % blockDim.x;
  for (int i = 0; i < n_turns; ++i) {
    g_cov_inv[local_id] = cov_inv[local_id];
  }
  if (local_id < n_left) {
    g_cov_inv[local_id] = cov_inv[local_id];
  }

  int cnt = 0;
  int tile_x = global_id / n_tiles_w;
  int tile_y = global_id % n_tiles_w;

  float2 *mean_ = reinterpret_cast<float2 *>(g_mean);
  float2 *cov_inv_ = reinterpret_cast<float2 *>(g_cov_inv);

  float2 tile_topleft = make_float2(topleft[0] + pixel_size_x * tile_x,
                                    topleft[1] + pixel_size_y * tile_y);

#pragma unroll
  for (int i = 0; i < N; ++i) {
    cnt += intersect_tile_gaussian2d(tile_topleft, tile_size, pixel_size_x,
                                     pixel_size_y, mean_ + i, cov_inv_ + 2 * i,
                                     thresh);
  }

  num_gaussians[global_id] += cnt;

  return;
}

// void count_tiled_gaussians_cuda(uint32_t N, float *mean, float *cov_inv,
//                                 float *topleft, uint32_t tile_size,
//                                 uint32_t n_tiles_h, uint32_t n_tiles_w,
//                                 float pixel_size_x, float pixel_size_y,
//                                 int *num_gaussians, float thresh) {
//   float2 *mean_ = reinterpret_cast<float2 *>(mean);
//   float2 *cov_inv_ = reinterpret_cast<float2 *>(cov_inv);
//   uint32_t n_blocks = div_round_up(N, (uint32_t)N_THREADS);
//   const dim3 block{n_blocks, n_tiles_w, n_tiles_h};
//   count_tiled_gaussians_kernel<<<block, N_THREADS>>>(
//       mean_, cov_inv_, top_left, pixel_size_x, pixel_size_y, n_tiles_h,
//       n_tiles_w, tile_size, num_gaussians, thresh);
//   return;
// }

void count_tiled_gaussians_cuda_sm(uint32_t N, float *mean, float *cov_inv,
                                   float *topleft, uint32_t tile_size,
                                   uint32_t n_tiles_h, uint32_t n_tiles_w,
                                   float pixel_size_x, float pixel_size_y,
                                   int *num_gaussians, float thresh) {
  const int max_num_gaussians_sm = MAX_N_FLOAT_SM / 6;
  int n_blocks = div_round_up(n_tiles_h * n_tiles_w, (uint32_t)N_THREADS);
  for (int i = 0; i < N; i += max_num_gaussians_sm) {
    uint32_t num_gaussians_sm = min(max_num_gaussians_sm, N - i);
    count_tiled_gaussians_kernel_sm<<<n_blocks, N_THREADS,
                                      num_gaussians * 6 * sizeof(float)>>>(
        num_gaussians_sm, mean + 2 * i, cov_inv + 4 * i, top_left, tile_size,
        n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, num_gaussians,
        thresh);
  }
}