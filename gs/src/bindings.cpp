#include "debug.h"
#include "render.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("culling_gaussian_bsphere", &culling_gaussian_bsphere,
        "Cull Gaussian with Bounding Sphere");
  m.def("count_num_gaussians_each_tile", &count_num_gaussians_each_tile,
        "Count number of gaussians in each tile");
  m.def("count_num_gaussians_each_tile_bcircle",
        &count_num_gaussians_each_tile_bcircle,
        "Count number of gaussians in each tile with bounding circle");
  m.def("prepare_image_sort", &prepare_image_sort, "Prepare image for sorting");
  m.def("image_sort", &image_sort, "Image radix sort");
  m.def("tile_based_vol_rendering", &tile_based_vol_rendering,
        "Tile based volume rendering");
  m.def("tile_based_vol_rendering_backward", &tile_based_vol_rendering_backward,
        "Tile based volume rendering backward");
  m.def("debug_check_tiledepth", &debug_check_tiledepth,
        "(DEBUG) check tile and depth");
}