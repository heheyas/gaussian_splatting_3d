#include "common.h"
#include "culling.h"
#include "data_spec.h"
#include "kernels.h"
#include <stdint.h>

void culling_gaussian_bsphere(Tensor mean, Tensor qvec, Tensor svec,
                              Tensor normal, Tensor pts, Tensor mask,
                              float thresh) {
  CHECK_CUDA(mean);
  CHECK_CUDA(qvec);
  CHECK_CUDA(svec);
  CHECK_CUDA(normal);
  CHECK_CUDA(pts);
  CHECK_CUDA(mask);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(qvec);
  CHECK_CONTIGUOUS(svec);
  CHECK_CONTIGUOUS(normal);
  CHECK_CONTIGUOUS(pts);
  CHECK_CONTIGUOUS(mask);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(qvec);
  CHECK_IS_FLOATING(svec);
  CHECK_IS_FLOATING(normal);
  CHECK_IS_FLOATING(pts);
  CHECK_IS_BOOL(mask);
  uint32_t N = mean.size(0);
  printf("N: %d\n", N);

  culling_gaussian_bsphere_cuda(N, mean.data_ptr<float>(),
                                qvec.data_ptr<float>(), svec.data_ptr<float>(),
                                normal.data_ptr<float>(), pts.data_ptr<float>(),
                                mask.data_ptr<bool>(), thresh);
}

void count_num_gaussians_each_tile(Tensor mean, Tensor cov_inv, Tensor topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, Tensor num_gaussians,
                                   float thresh) {
  CHECK_CUDA(mean);
  CHECK_CUDA(cov_inv);
  CHECK_CUDA(topleft);
  CHECK_CUDA(num_gaussians);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(cov_inv);
  CHECK_CONTIGUOUS(topleft);
  CHECK_CONTIGUOUS(num_gaussians);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(cov_inv);
  CHECK_IS_FLOATING(topleft);
  CHECK_IS_INT(num_gaussians);
  uint32_t N = mean.size(0);
  printf("N: %d\n", N);

  count_tiled_gaussians_cuda_sm(
      N, mean.data_ptr<float>(), cov_inv.data_ptr<float>(),
      topleft.data_ptr<float>(), tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
      pixel_size_y, num_gaussians.data_ptr<int>(), thresh);
}