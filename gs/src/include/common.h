#ifndef _COMMON_H
#define _COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#define GLOBAL __global__
#define DEVICE __device__
#define HOST __host__

#define N_THREADS 256

#define tid_1d(tid) const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_BOOL(x)                                                       \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool,                         \
              #x " must be an bool tensor")
#define CHECK_IS_INT(x)                                                        \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                          \
              #x " must be an int tensor")
#define CHECK_IS_FLOATING(x)                                                   \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float ||                      \
                  x.scalar_type() == at::ScalarType::Half ||                   \
                  x.scalar_type() == at::ScalarType::Double,                   \
              #x " must be a floating tensor")

using torch::Tensor;

const float EPS = 1e-6;
const float gs_coeff_3d = 0.06349363593424097f; // 1 / (2 * pi) ** (3/2)
const float gs_coeff_2d = 0.15915494309189535f; // 1 / (2 * pi)

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

#endif