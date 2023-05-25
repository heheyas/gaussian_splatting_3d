#include "render.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("culling_gaussian_bsphere", &culling_gaussian_bsphere,
        "Cull Gaussian with Bounding Sphere");
  m.def("count_num_gaussians_each_tile", &count_num_gaussians_each_tile,
        "Count number of gaussians in each tile");
}