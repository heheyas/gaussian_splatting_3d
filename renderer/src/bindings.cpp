#include "renderer.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cull_gaussian", &CullGaussian, "Cull Gaussian");
}