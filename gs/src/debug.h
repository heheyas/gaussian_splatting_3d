#include "common.h"

void check_tiledepth_host(size_t N, int *offset, double *tiledepth) {
  int *tile_ids = reinterpret_cast<int *>(tiledepth);
  float *tile_depth = reinterpret_cast<float *>(tiledepth);
  for (size_t i = 0; i < N; i++) {
    int start = offset[i];
    int end = offset[i + 1];
    for (size_t j = start; j < end - 1; j++) {
      assert(tile_ids[j] == i);
      assert(tile_ids[j + 1] == i);
      assert(tile_depth[j] <= tile_depth[j + 1]);
    }
  }
  printf("[DEBUG] [CUDA] Passed tiledepth check\n");
}

void debug_check_tiledepth(Tensor offset, Tensor tiledepth) {
  // make sure tensors are on host device
  size_t N = offset.size(0);
  check_tiledepth_host(N, offset.data_ptr<int>(), tiledepth.data_ptr<double>());
}
