#pragma once

#include <torch/extension.h>

using torch::Tensor;

struct Frustum {
  Tensor normals;
  Tensor distance;
};

struct Gaussian {
  Tensor qvec;
  Tensor svec;
  Tensor mean;
  Tensor color;
  Tensor alpha;
};

struct Camera {
  Tensor c2w;
  uint32_t height;
  uint32_t width;
  float fx, fy, cx, cy;
};

struct RenderOption {
  float background_brightness;
  float sigma_thresh;

  float near_plane;
  float far_plane;
};