#pragma once
#include "common.h"
#include "data_spec.h"
#include "kernels.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

__device__ inline void carry(uint32_t N, uint32_t dsize, float *sm, float *gm,
                             int *offset, int *gaussian_ids) {
  // still incorrect !
  // offset is not required
  int local_id = threadIdx.x;
  int n_turns = (dsize * N) / blockDim.x;
  int n_left = (dsize * N) % blockDim.x;
  for (int i = 0; i < n_turns; i++) {
    sm[local_id + i * blockDim.x] =
        gm[gaussian_ids[(local_id + i * blockDim.x) / dsize]];
  }
  if (local_id < n_left) {
    sm[local_id + n_turns * blockDim.x] =
        gm[gaussian_ids[(local_id + n_turns * blockDim.x) / dsize]];
  }
}

__device__ void
vol_render_one_batch(uint32_t N_gaussians_this_time, float *mean, float *cov,
                     float *color, float *alpha, float *out, float *cum_alpha,
                     float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
                     uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y,
                     uint32_t H, uint32_t W, float thresh, bool first) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  float color_this_time[3];
  float cum_alpha_this_time;
  if (first) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      color_this_time[i] = 0.0f;
    }
    cum_alpha_this_time = 1.0f;
  } else {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      color_this_time[i] = out[3 * local_id + i];
    }
    cum_alpha_this_time = cum_alpha[local_id];
  }

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha_this_time < thresh) {
      return;
    }
    // TODO: add gaussian kernel here
    // float coeff = alpha[i] * cum_alpha_this_time * gaussian_kernel_2d();
    float coeff = alpha[i] * cum_alpha_this_time;
    color_this_time[0] += color[3 * i + 0] * coeff;
    color_this_time[1] += color[3 * i + 1] * coeff;
    color_this_time[2] += color[3 * i + 2] * coeff;
    cum_alpha_this_time *= (1 - alpha[i]);
  }

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    out[3 * local_id + i] = color_this_time[i];
  }
  cum_alpha[local_id] = cum_alpha_this_time;
}

__global__ void tile_based_vol_rendering_entry(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *offset, int *gaussian_ids, float *out_rgb,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.x * gridDim.y + blockIdx.y;
  int n_gaussians_this_tile = offset[tile_id + 1] - offset[tile_id];
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  // mean + cov + color + alpha (maybe add a det ?)
  int n_pixel_per_tile = tile_size * tile_size;
  int n_float_per_pixel = 3 + 1;
  int max_gaussian_sm =
      (MAX_N_FLOAT_SM - n_float_per_pixel * n_pixel_per_tile) /
      n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  float *sm_cum_alpha = sm_alpha + 1 * n_pixel_per_tile;
  float *sm_out = sm_cum_alpha + 3 * n_pixel_per_tile;

  // no need to initialize
  //   cudaMemset(sm_cum_alpha, 0, sizeof(float) * n_pixel_per_tile);
  //   cudaMemset(sm_out, 0, sizeof(float) * 4 * n_pixel_per_tile);

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, N - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, offset, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, offset, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, offset, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, offset, gaussian_ids);
    __syncthreads();
    vol_render_one_batch(num_gaussian_sm, sm_mean, sm_cov, sm_color, sm_alpha,
                         sm_out, sm_cum_alpha, topleft, tile_size, n_tiles_h,
                         n_tiles_w, pixel_size_x, pixel_size_y, H, W, thresh,
                         n == 0);
    __syncthreads();
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out_rgb[3 * (global_y * W + global_x) + 0] = sm_out[3 * local_id + 0];
  out_rgb[3 * (global_y * W + global_x) + 1] = sm_out[3 * local_id + 1];
  out_rgb[3 * (global_y * W + global_x) + 2] = sm_out[3 * local_id + 2];
}

// TODO: add backward compatibility
// TODO: add spherical harmonic support
void tile_based_vol_rendering_cuda(uint32_t N, uint32_t N_with_dub, float *mean,
                                   float *cov, float *color, float *alpha,
                                   int *offset, int *gaussian_ids,
                                   float *out_rgb, float *topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, uint32_t H, uint32_t W,
                                   float thresh) {
  const dim3 block(n_tiles_h, n_tiles_w, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_entry<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, offset, gaussian_ids, out_rgb,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh);
}