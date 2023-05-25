#include "common.h"

void culling_gaussian_bsphere(Tensor mean, Tensor qvec, Tensor svec,
                              Tensor normal, Tensor pts, Tensor mask,
                              float thresh);

void count_num_gaussians_each_tile(Tensor mean, Tensor cov_inv, Tensor topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, Tensor num_gaussians,
                                   float thresh)