#ifndef _RENDERER_H
#define _RENDERER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/torch.h>

using torch::Tensor;

void CullGaussian(uint32_t N, Tensor mean, Tensor qvec, Tensor svec,
                  Tensor normals, Tensor pts, Tensor mask);

#endif