#ifndef _RENDERER_H
#define _RENDERER_H

#include "helper_math.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/torch.h>

const float EPS = 1e-5;

using torch::Tensor;

void cull_gaussian(uint32_t N, Tensor mean, Tensor qvec, Tensor svec,
                   Tensor normals, Tensor pts, Tensor mask,
                   float thresh = 0.01f);

#endif