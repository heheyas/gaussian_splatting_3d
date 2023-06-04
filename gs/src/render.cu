#include "common.h"
#include "culling.h"
#include "data_spec.h"
#include "kernels.h"
#include "tile_ops.h"
#include "vol_render.h"
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
  //   printf("N: %d\n", N);

  culling_gaussian_bsphere_cuda(N, mean.data_ptr<float>(),
                                qvec.data_ptr<float>(), svec.data_ptr<float>(),
                                normal.data_ptr<float>(), pts.data_ptr<float>(),
                                mask.data_ptr<bool>(), thresh);
}

void count_num_gaussians_each_tile(Tensor mean, Tensor cov, Tensor topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, Tensor num_gaussians,
                                   float thresh) {
  CHECK_CUDA(mean);
  CHECK_CUDA(cov);
  CHECK_CUDA(topleft);
  CHECK_CUDA(num_gaussians);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(cov);
  CHECK_CONTIGUOUS(topleft);
  CHECK_CONTIGUOUS(num_gaussians);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(cov);
  CHECK_IS_FLOATING(topleft);
  CHECK_IS_INT(num_gaussians);
  uint32_t N = mean.size(0);
  //   printf("N: %d\n", N);
  //   printf("It seems that it is not recompiled after modifications.\n");

  count_tiled_gaussians_cuda_sm(
      N, mean.data_ptr<float>(), cov.data_ptr<float>(),
      topleft.data_ptr<float>(), tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
      pixel_size_y, num_gaussians.data_ptr<int>(), thresh);
}

void count_num_gaussians_each_tile_bcircle(
    Tensor mean, Tensor radius, Tensor topleft, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, Tensor num_gaussians) {
  CHECK_CUDA(mean);
  CHECK_CUDA(radius);
  CHECK_CUDA(topleft);
  CHECK_CUDA(num_gaussians);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(radius);
  CHECK_CONTIGUOUS(topleft);
  CHECK_CONTIGUOUS(num_gaussians);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(radius);
  CHECK_IS_FLOATING(topleft);
  CHECK_IS_INT(num_gaussians);
  uint32_t N = mean.size(0);
  //   printf("N: %d\n", N);
  //   printf("It seems that it is not recompiled after modifications.\n");

  count_tiled_gaussians_bcircle_cuda_sm(
      N, mean.data_ptr<float>(), radius.data_ptr<float>(),
      topleft.data_ptr<float>(), tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
      pixel_size_y, num_gaussians.data_ptr<int>());
}

void prepare_image_sort(Tensor gaussian_ids, Tensor tiledepth, Tensor depth,
                        Tensor tile_n_gaussians, Tensor offset, Tensor mean,
                        Tensor radius, Tensor topleft, uint32_t tile_size,
                        uint32_t n_tiles_h, uint32_t n_tiles_w,
                        float pixel_size_x, float pixel_size_y) {
  CHECK_CUDA(gaussian_ids);
  CHECK_CUDA(tiledepth);
  CHECK_CUDA(depth);
  CHECK_CUDA(tile_n_gaussians);
  CHECK_CUDA(offset);
  CHECK_CUDA(mean);
  CHECK_CUDA(radius);
  CHECK_CUDA(topleft);
  CHECK_CONTIGUOUS(gaussian_ids);
  CHECK_CONTIGUOUS(tiledepth);
  CHECK_CONTIGUOUS(depth);
  CHECK_CONTIGUOUS(tile_n_gaussians);
  CHECK_CONTIGUOUS(offset);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(radius);
  CHECK_CONTIGUOUS(topleft);
  CHECK_IS_INT(gaussian_ids);
  CHECK_IS_DOUBLE(tiledepth);
  CHECK_IS_FLOATING(depth);
  CHECK_IS_INT(tile_n_gaussians);
  CHECK_IS_INT(offset);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(radius);
  CHECK_IS_FLOATING(topleft);
  uint32_t N = mean.size(0);
  uint32_t N_with_dub = tiledepth.size(0);

  prepare_image_sort_cuda(
      N, N_with_dub, gaussian_ids.data_ptr<int>(), tiledepth.data_ptr<double>(),
      depth.data_ptr<float>(), tile_n_gaussians.data_ptr<int>(),
      offset.data_ptr<int>(), mean.data_ptr<float>(), radius.data_ptr<float>(),
      topleft.data_ptr<float>(), tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
      pixel_size_y);
}

void image_sort(Tensor gaussian_ids, Tensor tiledepth, Tensor depth,
                Tensor tile_n_gaussians, Tensor offset, Tensor mean, Tensor cov,
                Tensor topleft, uint32_t tile_size, uint32_t n_tiles_h,
                uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y,
                float thresh) {
  CHECK_CUDA(gaussian_ids);
  CHECK_CUDA(tiledepth);
  CHECK_CUDA(depth);
  CHECK_CUDA(tile_n_gaussians);
  CHECK_CUDA(offset);
  CHECK_CUDA(mean);
  CHECK_CUDA(cov);
  CHECK_CUDA(topleft);
  CHECK_CONTIGUOUS(gaussian_ids);
  CHECK_CONTIGUOUS(tiledepth);
  CHECK_CONTIGUOUS(depth);
  CHECK_CONTIGUOUS(tile_n_gaussians);
  CHECK_CONTIGUOUS(offset);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(cov);
  CHECK_CONTIGUOUS(topleft);
  CHECK_IS_INT(gaussian_ids);
  CHECK_IS_DOUBLE(tiledepth);
  CHECK_IS_FLOATING(depth);
  CHECK_IS_INT(tile_n_gaussians);
  CHECK_IS_INT(offset);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(cov);
  CHECK_IS_FLOATING(topleft);
  uint32_t N = mean.size(0);
  uint32_t N_with_dub = tiledepth.size(0);

  image_sort_cuda(N, N_with_dub, gaussian_ids.data_ptr<int>(),
                  tiledepth.data_ptr<double>(), depth.data_ptr<float>(),
                  tile_n_gaussians.data_ptr<int>(), offset.data_ptr<int>(),
                  mean.data_ptr<float>(), cov.data_ptr<float>(),
                  topleft.data_ptr<float>(), tile_size, n_tiles_h, n_tiles_w,
                  pixel_size_x, pixel_size_y, thresh);
}

void tile_based_vol_rendering(Tensor mean, Tensor cov, Tensor color,
                              Tensor alpha, Tensor offset, Tensor gaussian_ids,
                              Tensor out, Tensor topleft, uint32_t tile_size,
                              uint32_t n_tiles_h, uint32_t n_tiles_w,
                              float pixel_size_x, float pixel_size_y,
                              uint32_t H, uint32_t W, float thresh) {
  CHECK_CUDA(mean);
  CHECK_CUDA(cov);
  CHECK_CUDA(color);
  CHECK_CUDA(alpha);
  CHECK_CUDA(offset);
  CHECK_CUDA(gaussian_ids);
  CHECK_CUDA(out);
  CHECK_CUDA(topleft);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(cov);
  CHECK_CONTIGUOUS(color);
  CHECK_CONTIGUOUS(alpha);
  CHECK_CONTIGUOUS(offset);
  CHECK_CONTIGUOUS(gaussian_ids);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(topleft);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(cov);
  CHECK_IS_FLOATING(color);
  CHECK_IS_FLOATING(alpha);
  CHECK_IS_INT(offset);
  CHECK_IS_INT(gaussian_ids);
  CHECK_IS_FLOATING(out);
  CHECK_IS_FLOATING(topleft);
  uint32_t N = mean.size(0);
  uint32_t N_with_dub = gaussian_ids.size(0);
  //   printf("tile_based_vol_rendering\n");
  tile_based_vol_rendering_cuda(
      N, N_with_dub, mean.data_ptr<float>(), cov.data_ptr<float>(),
      color.data_ptr<float>(), alpha.data_ptr<float>(), offset.data_ptr<int>(),
      gaussian_ids.data_ptr<int>(), out.data_ptr<float>(),
      topleft.data_ptr<float>(), tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
      pixel_size_y, H, W, thresh);
}

void tile_based_vol_rendering_backward(
    Tensor mean, Tensor cov, Tensor color, Tensor alpha, Tensor offset,
    Tensor gaussian_ids, Tensor out, Tensor grad_mean, Tensor grad_cov,
    Tensor grad_color, Tensor grad_alpha, Tensor grad_out, Tensor topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  CHECK_CUDA(mean);
  CHECK_CUDA(cov);
  CHECK_CUDA(color);
  CHECK_CUDA(alpha);
  CHECK_CUDA(offset);
  CHECK_CUDA(gaussian_ids);
  CHECK_CUDA(out);
  CHECK_CUDA(topleft);
  CHECK_CUDA(grad_mean);
  CHECK_CUDA(grad_cov);
  CHECK_CUDA(grad_color);
  CHECK_CUDA(grad_alpha);
  CHECK_CUDA(grad_out);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(cov);
  CHECK_CONTIGUOUS(color);
  CHECK_CONTIGUOUS(alpha);
  CHECK_CONTIGUOUS(offset);
  CHECK_CONTIGUOUS(gaussian_ids);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(topleft);
  CHECK_CONTIGUOUS(grad_mean);
  CHECK_CONTIGUOUS(grad_cov);
  CHECK_CONTIGUOUS(grad_color);
  CHECK_CONTIGUOUS(grad_alpha);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(cov);
  CHECK_IS_FLOATING(color);
  CHECK_IS_FLOATING(alpha);
  CHECK_IS_INT(offset);
  CHECK_IS_INT(gaussian_ids);
  CHECK_IS_FLOATING(out);
  CHECK_IS_FLOATING(topleft);
  CHECK_IS_FLOATING(grad_mean);
  CHECK_IS_FLOATING(grad_cov);
  CHECK_IS_FLOATING(grad_color);
  CHECK_IS_FLOATING(grad_alpha);
  CHECK_IS_FLOATING(grad_out);
  uint32_t N = mean.size(0);
  uint32_t N_with_dub = gaussian_ids.size(0);
  //   printf("tile_based_vol_rendering_backward\n");
  tile_based_vol_rendering_backward_cuda(
      N, N_with_dub, mean.data_ptr<float>(), cov.data_ptr<float>(),
      color.data_ptr<float>(), alpha.data_ptr<float>(), offset.data_ptr<int>(),
      gaussian_ids.data_ptr<int>(), out.data_ptr<float>(),
      grad_mean.data_ptr<float>(), grad_cov.data_ptr<float>(),
      grad_color.data_ptr<float>(), grad_alpha.data_ptr<float>(),
      grad_out.data_ptr<float>(), topleft.data_ptr<float>(), tile_size,
      n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H, W, thresh);
}