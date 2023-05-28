#include "common.h"

void culling_gaussian_bsphere(Tensor mean, Tensor qvec, Tensor svec,
                              Tensor normal, Tensor pts, Tensor mask,
                              float thresh);

void count_num_gaussians_each_tile(Tensor mean, Tensor cov_inv, Tensor topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, Tensor num_gaussians,
                                   float thresh);

void count_num_gaussians_each_tile_bcircle(
    Tensor mean, Tensor radius, Tensor topleft, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, Tensor num_gaussians);

void prepare_image_sort(Tensor gaussian_ids, Tensor tiledepth, Tensor depth,
                        Tensor tile_n_gaussians, Tensor offset, Tensor mean,
                        Tensor radius, Tensor topleft, uint32_t tile_size,
                        uint32_t n_tiles_h, uint32_t n_tiles_w,
                        float pixel_size_x, float pixel_size_y);

void tile_based_vol_rendering(Tensor mean, Tensor cov, Tensor color,
                              Tensor alpha, Tensor offset, Tensor gaussian_ids,
                              Tensor out, Tensor topleft, uint32_t tile_size,
                              uint32_t n_tiles_h, uint32_t n_tiles_w,
                              float pixel_size_x, float pixel_size_y,
                              uint32_t H, uint32_t W, float thresh);