#pragma once
#include "common.h"
#include "data_spec.h"
#include "kernels.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

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
    g_mean[local_id + i * blockDim.x] = mean[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    g_mean[local_id + n_turns * blockDim.x] =
        mean[local_id + n_turns * blockDim.x];
  }

  n_turns = (4 * N) / blockDim.x;
  n_left = (4 * N) % blockDim.x;
  for (int i = 0; i < n_turns; ++i) {
    g_cov_inv[local_id + i * blockDim.x] = cov_inv[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    g_cov_inv[local_id + n_turns * blockDim.x] =
        cov_inv[local_id + n_turns * blockDim.x];
  }

  __syncthreads();

  if (global_id >= n_tiles_h * n_tiles_w)
    return;

  int cnt = 0;
  int tile_x = global_id % n_tiles_w;
  int tile_y = global_id / n_tiles_w;

  float2 *mean_ = reinterpret_cast<float2 *>(g_mean);
  float2 *cov_inv_ = reinterpret_cast<float2 *>(g_cov_inv);

  float2 tile_topleft =
      make_float2(topleft[0] + pixel_size_x * tile_x * tile_size,
                  topleft[1] + pixel_size_y * tile_y * tile_size);

#pragma unroll
  for (int i = 0; i < N; ++i) {
    cnt += intersect_tile_gaussian2d(tile_topleft, tile_size, pixel_size_x,
                                     pixel_size_y, mean_ + i, cov_inv_ + 2 * i,
                                     thresh);
  }

  num_gaussians[global_id] += cnt;

  return;
}

__global__ void count_tiled_gaussians_bcircle_kernel_sm(
    uint32_t N, float *mean, float *radius, float *topleft, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, int *num_gaussians) {

  tid_1d(global_id);
  int local_id = threadIdx.x;
  int n_turns = (2 * N) / blockDim.x;
  int n_left = (2 * N) % blockDim.x;

  // load data onto scatchpad mem
  extern __shared__ float g_mean[];
  float *g_radius = g_mean + 2 * N;

  for (int i = 0; i < n_turns; ++i) {
    g_mean[local_id + i * blockDim.x] = mean[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    g_mean[local_id + n_turns * blockDim.x] =
        mean[local_id + n_turns * blockDim.x];
  }

  n_turns = (N) / blockDim.x;
  n_left = (N) % blockDim.x;
  for (int i = 0; i < n_turns; ++i) {
    g_radius[local_id + i * blockDim.x] = radius[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    g_radius[local_id + n_turns * blockDim.x] =
        radius[local_id + n_turns * blockDim.x];
  }

  __syncthreads();
  if (global_id >= n_tiles_h * n_tiles_w)
    return;

  int cnt = 0;
  int tile_x = global_id % n_tiles_w;
  int tile_y = global_id / n_tiles_w;

  float2 *mean_ = reinterpret_cast<float2 *>(g_mean);

  float2 tile_topleft =
      make_float2(topleft[0] + pixel_size_x * tile_x * tile_size,
                  topleft[1] + pixel_size_y * tile_y * tile_size);

#pragma unroll
  for (int i = 0; i < N; ++i) {
    cnt +=
        intersect_tile_gaussian2d_bcircle(tile_topleft, tile_size, pixel_size_x,
                                          pixel_size_y, mean_ + i, g_radius[i]);
  }

  num_gaussians[global_id] += cnt;

  return;
}

void count_tiled_gaussians_cuda_sm(uint32_t N, float *mean, float *cov_inv,
                                   float *topleft, uint32_t tile_size,
                                   uint32_t n_tiles_h, uint32_t n_tiles_w,
                                   float pixel_size_x, float pixel_size_y,
                                   int *num_gaussians, float thresh) {
  const int max_num_gaussians_sm = MAX_N_FLOAT_SM / 6;
  uint32_t n_blocks = div_round_up(n_tiles_h * n_tiles_w, (uint32_t)N_THREADS);
#pragma unroll
  for (int i = 0; i < N; i += max_num_gaussians_sm) {
    uint32_t num_gaussians_sm = min(max_num_gaussians_sm, N - i);
    printf("num_gaussians_sm: %d\n", num_gaussians_sm);
    count_tiled_gaussians_kernel_sm<<<n_blocks, N_THREADS,
                                      num_gaussians_sm * 6 * sizeof(float)>>>(
        num_gaussians_sm, mean + 2 * i, cov_inv + 4 * i, topleft, tile_size,
        n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, num_gaussians,
        thresh);
    cudaError_t last_error;
    checkLastCudaError(last_error);
    cudaDeviceSynchronize();
  }
  return;
}

void count_tiled_gaussians_bcircle_cuda_sm(
    uint32_t N, float *mean, float *radius, float *topleft, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, int *num_gaussians) {
  const int max_num_gaussians_sm = MAX_N_FLOAT_SM / 3;
  uint32_t n_blocks = div_round_up(n_tiles_h * n_tiles_w, (uint32_t)N_THREADS);
#pragma unroll
  for (int i = 0; i < N; i += max_num_gaussians_sm) {
    uint32_t num_gaussians_sm = min(max_num_gaussians_sm, N - i);
    printf("num_gaussians_sm: %d\n", num_gaussians_sm);
    count_tiled_gaussians_bcircle_kernel_sm<<<
        n_blocks, N_THREADS, num_gaussians_sm * 3 * sizeof(float)>>>(
        num_gaussians_sm, mean + 2 * i, radius + i, topleft, tile_size,
        n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, num_gaussians);
    cudaError_t last_error;
    checkLastCudaError(last_error);
    cudaDeviceSynchronize();
  }
  return;
}

__global__ void fill_tiledepth_bsphere_kernel_sm(
    uint32_t N_base, uint32_t N, int *gaussian_ids, double *tiledepth,
    float *depth, int *tile_n_gaussians, int *offset, float *mean,
    float *radius, float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y) {
  tid_1d(global_id);
  int local_id = threadIdx.x;
  int n_turns = (2 * N) / blockDim.x;
  int n_left = (2 * N) % blockDim.x;

  // load data onto scatchpad mem
  extern __shared__ float g_mean[];
  float *g_radius = g_mean + 2 * N;

  for (int i = 0; i < n_turns; ++i) {
    g_mean[local_id + i * blockDim.x] = mean[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    g_mean[local_id + n_turns * blockDim.x] =
        mean[local_id + n_turns * blockDim.x];
  }

  n_turns = (N) / blockDim.x;
  n_left = (N) % blockDim.x;
  for (int i = 0; i < n_turns; ++i) {
    g_radius[local_id + i * blockDim.x] = radius[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    g_radius[local_id + n_turns * blockDim.x] =
        radius[local_id + n_turns * blockDim.x];
  }

  __syncthreads();

  if (global_id >= n_tiles_h * n_tiles_w)
    return;

  int cnt = 0;
  int off = offset[global_id] + tile_n_gaussians[global_id];
  int tile_x = global_id % n_tiles_w;
  int tile_y = global_id / n_tiles_w;

  float2 *mean_ = reinterpret_cast<float2 *>(g_mean);
  float2 tile_topleft =
      make_float2(topleft[0] + pixel_size_x * tile_x * tile_size,
                  topleft[1] + pixel_size_y * tile_y * tile_size);

  int *tile_ids = reinterpret_cast<int *>(tiledepth);
  float *tile_depths = reinterpret_cast<float *>(tiledepth);

#pragma unroll
  for (int i = 0; i < N; ++i) {
    if (intersect_tile_gaussian2d_bcircle(tile_topleft, tile_size, pixel_size_x,
                                          pixel_size_y, mean_ + i,
                                          g_radius[i])) {
      cnt += 1;
      // tile_ids[2 * off] = global_id;
      // tile_depths[2 * off + 1] = depth[N_base + i];
      tile_ids[2 * off + 1] = global_id;
      tile_depths[2 * off] = depth[N_base + i];
      off += 1;
      gaussian_ids[off] = N_base + i;
    }
  }
  tile_n_gaussians[global_id] += cnt;
}

void fill_tiledepth_bsphere_cuda(uint32_t N, int *gaussian_ids,
                                 double *tiledepth, float *depth,
                                 int *tile_n_gaussians, int *offset,
                                 float *mean, float *radius, float *topleft,
                                 uint32_t tile_size, uint32_t n_tiles_h,
                                 uint32_t n_tiles_w, float pixel_size_x,
                                 float pixel_size_y) {
  // this function will reset the value of tile_n_gaussians, but should be
  // identical to the previous one
  /* test kenerl vals */
  // float2 mean_ = make_float2(0, 0);
  // float4 cov_ = make_float4(1, 0, 0, 1);
  // float2 query = make_float2(0, 0);
  // float2 xy = query - mean_;
  // float2 tmp =
  //     make_float2(xy.x * cov_.x + xy.y * cov_.y, xy.x * cov_.z + xy.y *
  //     cov_.w);
  // float radial = dot(tmp, xy);
  // printf("radial: %.2f\n", radial);
  // float kernel_val = kernel_gaussian_2d(mean_, cov_, query);
  // printf("kernel val: %f\n", kernel_val);
  // fflush(stdout);
  printf("fill_tiledepth_bsphere_cuda\n");
  int n_tiles = n_tiles_h * n_tiles_w;
  cudaCheck(cudaMemset(tile_n_gaussians, 0, n_tiles * sizeof(int)));
  const int max_num_gaussians_sm = MAX_N_FLOAT_SM / 3;
  uint32_t n_blocks = div_round_up(n_tiles_h * n_tiles_w, (uint32_t)N_THREADS);
  uint32_t N_base = 0;
#pragma unroll
  for (int i = 0; i < N; i += max_num_gaussians_sm) {
    uint32_t num_gaussians_sm = min(max_num_gaussians_sm, N - i);
    printf("num_gaussians_sm: %d\n", num_gaussians_sm);
    fill_tiledepth_bsphere_kernel_sm<<<n_blocks, N_THREADS,
                                       num_gaussians_sm * 3 * sizeof(float)>>>(
        N_base, num_gaussians_sm, gaussian_ids, tiledepth, depth,
        tile_n_gaussians, offset, mean + 2 * i, radius + i, topleft, tile_size,
        n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y);
    cudaError_t last_error;
    checkLastCudaError(last_error);
    cudaDeviceSynchronize();
    N_base += num_gaussians_sm;
  }
  return;
}

void prepare_image_sort_cuda(uint32_t N, uint32_t N_with_dub, int *gaussian_ids,
                             double *tiledepth, float *depth,
                             int *tile_n_gaussians, int *offset, float *mean,
                             float *radius, float *topleft, uint32_t tile_size,
                             uint32_t n_tiles_h, uint32_t n_tiles_w,
                             float pixel_size_x, float pixel_size_y) {
  // N stands for number of gaussians
  // assuimg inclusive scan is not done
  printf("prepare_image_sort_cuda N: %d\n", N);
  cudaError_t last_error;
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  size_t n_tiles = n_tiles_h * n_tiles_w;
  cudaCheck(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          tile_n_gaussians, offset, n_tiles));
  printf("temp_storage_bytes: %d\n", temp_storage_bytes);
  cudaCheck(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
  printf("here\n");
  printf("tiledepth: %p\n", tiledepth);
  cudaCheck(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          tile_n_gaussians, offset, n_tiles));
  int *unsorted_gaussian_ids;
  cudaCheck(
      cudaMalloc((void **)&unsorted_gaussian_ids, N_with_dub * sizeof(int)));
  fill_tiledepth_bsphere_cuda(N, unsorted_gaussian_ids, tiledepth, depth,
                              tile_n_gaussians, offset, mean, radius, topleft,
                              tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
                              pixel_size_y);
  checkLastCudaError(last_error);

  int64_t *sorted_keys;
  cudaCheck(cudaMalloc((void **)&sorted_keys, N_with_dub * sizeof(int64_t)));

  int64_t *keys = reinterpret_cast<int64_t *>(tiledepth);
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  cudaCheck(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, keys, sorted_keys,
      unsorted_gaussian_ids, gaussian_ids, N_with_dub));
  cudaCheck(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
  cudaCheck(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, keys, sorted_keys,
      unsorted_gaussian_ids, gaussian_ids, N_with_dub));

  // printf("N_with_dub: %d\n", N_with_dub);
  // fflush(stdout);

  // int64_t *keys_h = (int64_t *)malloc(N_with_dub * sizeof(int64_t));
  // cudaCheck(cudaMemcpy(keys_h, keys, N_with_dub * sizeof(int64_t),
  //                      cudaMemcpyDeviceToHost));
  // int *tiles_ids_h = reinterpret_cast<int *>(keys_h);
  // for (int i = 0; i < N_with_dub; ++i) {
  //   printf("sorted[%d]: %d\n", i, tiles_ids_h[2 * i + 1]);
  // }

  // int64_t *sorted_keys_h = (int64_t *)malloc(N_with_dub * sizeof(int64_t));
  // int *offset_h = (int *)malloc(n_tiles * sizeof(int));
  // cudaCheck(cudaMemcpy(sorted_keys_h, sorted_keys, N_with_dub *
  // sizeof(int64_t),
  //                      cudaMemcpyDeviceToHost));
  // cudaCheck(cudaMemcpy(offset_h, offset, n_tiles * sizeof(int),
  //                      cudaMemcpyDeviceToHost));
  // // int *sorted_gaussian_ids = reinterpret_cast<int *>(sorted_keys);

  // int *sorted_keys_h_ids = reinterpret_cast<int *>(sorted_keys_h);
  // // for (int i = 0; i < 100; ++i) {
  // //   printf("tile_id[%d]: %d\n", i, sorted_keys_h_ids[offset_h[i] * 2]);
  // // }
  // for (int i = 0; i < n_tiles; ++i) {
  //   printf("tile[%d]: %d\n", i, sorted_keys_h_ids[2 * offset_h[i] + 1]);
  // }

  return;
}