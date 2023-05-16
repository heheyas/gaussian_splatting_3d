#include "renderer.h"
#include <cub/cub.cuh>
#include <stdint.h>
#include <torch/torch.h>

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

#define TILE 16

// __device__ float3 operator+(const float3 &a, const float3 &b) {
//   return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
// }

// __device__ float3 operator1(const float3 &a, const float3 &b) {
//   return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
// }

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <typename scalar_t>
__device__ inline bool above_plane(scalar_t *query, scalar_t *normals,
                                   scalar_t *pts) {
  bool above = true;
#pragma unroll
  for (int i = 0; i < 6; i++) {
    scalar_t dot_val = 0;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      dot_val += (query[i] - pts[i]) * normals[i];
    }
    above &= (dot_val > 0.0f);
  }
  return above;
}

template <typename scalar_t>
__global__ void naive_cull_gaussian_kernel(uint32_t N, scalar_t *mean,
                                           scalar_t *normals, scalar_t *pts,
                                           bool *mask) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;

  mean += n * 3;
  // normals += n * 3 * 6;
  // pts += n * 3 * 6;
  mask += n;

  *mask = above_plane(mean, normals, pts);
}

template <typename scalar_t>
void naive_cull_gaussian_cuda(uint32_t N, scalar_t *mean, scalar_t *normals,
                              scalar_t *pts, bool *mask) {
  static constexpr uint32_t N_THREAD = 256;
  const uint32_t n_blocks = div_round_up(N, N_THREAD);
  naive_cull_gaussian_kernel<scalar_t>
      <<<n_blocks, N_THREAD>>>(N, mean, normals, pts, mask);
}

void CullGaussian(uint32_t N, Tensor mean, Tensor qvec, Tensor svec,
                  Tensor normals, Tensor pts, Tensor mask) {
  // culling gaussian according to whether the frustum is in its 99% confidence
  // interval return a tensor with index and depth
  // checks
  CHECK_CUDA(mean);
  CHECK_CUDA(qvec);
  CHECK_CUDA(svec);
  CHECK_CUDA(normals);
  CHECK_CUDA(pts);
  CHECK_CUDA(mask);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(qvec);
  CHECK_CONTIGUOUS(svec);
  CHECK_CONTIGUOUS(normals);
  CHECK_CONTIGUOUS(pts);
  CHECK_CONTIGUOUS(mask);
  CHECK_IS_FLOATING(mean);
  CHECK_IS_FLOATING(qvec);
  CHECK_IS_FLOATING(svec);
  CHECK_IS_FLOATING(normals);
  CHECK_IS_FLOATING(pts);
  CHECK_IS_BOOL(mask);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      mean.scalar_type(), "CullGaussian", ([&] {
        naive_cull_gaussian_cuda<scalar_t>(
            N, mean.data_ptr<scalar_t>(), normals.data_ptr<scalar_t>(),
            pts.data_ptr<scalar_t>(), mask.data_ptr<bool>());
      }));
}

// __global__ void ScreenSpaceGaussians() {
//   // project 3d gaussians onto screen plane
// }

// __global__ void DuplicateWithKeys() {
//   // prepare for radix sort, check visibility of gaussians towards each tile
//   and
//   // duplicate keys
// }

// __global__ void IdentiyTileRange() {
//   // find range for each tile on the sorted array
// }

// __global__ void BlendInOrder() {
//   // do volumetric rendering
// }

// __global__ void tiled_rasterize(uint32_t width, uint32_t height, Guassian G)
// {
//   uint32_t n_tile_w = width / TILE + (width % TILE != 0);
//   uint32_t n_tile_h = height / TILE + (height % TILE != 0);
//   uint32_t n_tile = n_tile_h * n_tile_w;
// }