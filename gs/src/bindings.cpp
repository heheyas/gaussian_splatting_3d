#include "render.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("culling_gaussian_bsphere", &culling_gaussian_bsphere,
        "Cull Gaussian with Bounding Sphere");
}