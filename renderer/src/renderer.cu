#include "kernels.cu"
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

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

__device__ inline bool above_plane(float3 *mean, float3 *normals, float3 *pts) {
  return dot(*mean - *pts, *normals) > 0.0f;
}

__device__ bool on_frustum(float *query, float *normals, float *pts) {
  float3 *normals_ = reinterpret_cast<float3 *>(normals);
  float3 *pts_ = reinterpret_cast<float3 *>(pts);
  float3 *query_ = reinterpret_cast<float3 *>(query);
  for (int i = 0; i < 6; ++i) {
    if (dot(query_[0] - pts_[i], normals_[i]) < -EPS) {
      return false;
    }
  }
  return true;
}

__device__ bool on_frustum(float3 *query, float3 *normals, float3 *pts) {
  for (int i = 0; i < 6; ++i) {
    if (dot(query[0] - pts[i], normals[i]) < -EPS) {
      return false;
    }
  }
  return true;
}

__device__ bool intersect_gaussian_plane(float *mean, float *sigma_inv,
                                         float *normal, float *pts,
                                         float thresh = 0.01f) {
  float3 mean_ = reinterpret_cast<float3 *>(mean)[0];
  float3 normal_ = reinterpret_cast<float3 *>(normal)[0];
  float3 pts_ = reinterpret_cast<float3 *>(pts)[0];
  float3 nearest_ = mean_ - dot(normal_, mean_ - pts_) * normal_;
  float3 *sigma_inv_ = reinterpret_cast<float3 *>(sigma_inv);

  return gaussian_kernel_3d_with_inv(&mean_, sigma_inv_, &nearest_) > thresh;
}

__device__ bool intersect_gaussian_plane(float *mean, float *qvec, float *svec,
                                         float *normal, float *pts,
                                         float thresh = 0.01) {

  float3 mean_ = reinterpret_cast<float3 *>(mean)[0];
  float3 normal_ = reinterpret_cast<float3 *>(normal)[0];
  float3 pts_ = reinterpret_cast<float3 *>(pts)[0];
  float3 nearest_ = mean_ - dot(normal_, mean_ - pts_) * normal_;
  float4 *qvec_ = reinterpret_cast<float4 *>(qvec);
  float3 *svec_ = reinterpret_cast<float3 *>(svec);

  // if (!on_frustum(&mean_, &normal_, &pts_)) {
  //   return false;
  // }

  return gaussian_kernel_3d(&mean_, qvec_, svec_, &nearest_) > thresh;
}

__device__ bool intersect_gaussian_plane_aabb(float *mean, float *qvec,
                                              float *svec, float *normal,
                                              float *pts, float thresh = 0.01) {
  return false;
}

template <typename scalar_t>
__device__ bool in_frustum(scalar_t *query, scalar_t *normals, scalar_t *pts) {
  bool above = true;
#pragma unroll
  for (int i = 0; i < 6; i++) {
    scalar_t dot_val = 0;
#pragma unroll
    for (int j = 0; j < 3; j++) {
      dot_val += (query[j] - pts[i * 3 + j]) * normals[i * 3 + j];
    }
    above = above && (dot_val > 0);
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
  // mask += n;

  mask[n] = in_frustum(mean, normals, pts);
}

template <typename scalar_t>
void naive_cull_gaussian_cuda(uint32_t N, scalar_t *mean, scalar_t *normals,
                              scalar_t *pts, bool *mask) {
  static constexpr uint32_t N_THREAD = 256;
  const uint32_t n_blocks = div_round_up(N, N_THREAD);
  naive_cull_gaussian_kernel<scalar_t>
      <<<n_blocks, N_THREAD>>>(N, mean, normals, pts, mask);
}

__global__ void cull_gaussian_kernel(uint32_t N, float *mean, float *qvec,
                                     float *svec, float *normals, float *pts,
                                     bool *mask, float thresh) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;

  mean += n * 3;
  mask += n;
  bool on = on_frustum(reinterpret_cast<float3 *>(mean),
                       reinterpret_cast<float3 *>(normals),
                       reinterpret_cast<float3 *>(pts));
  if (on) {
    printf("there do exists !!!\n");
  }

  bool inside_frustum = in_frustum(mean, normals, pts);
  if (inside_frustum) {
    *mask = true;
    return;
  } else {
    bool intersect = false;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
      bool on = on_frustum(reinterpret_cast<float3 *>(mean),
                           reinterpret_cast<float3 *>(normals),
                           reinterpret_cast<float3 *>(pts));
      if (on) {
        printf("there do exists !!!\n");
      }
      intersect = intersect ||
                  (intersect_gaussian_plane(mean, qvec, svec, normals + 3 * i,
                                            pts + 3 * i, thresh) &&
                   on_frustum(reinterpret_cast<float3 *>(mean),
                              reinterpret_cast<float3 *>(normals),
                              reinterpret_cast<float3 *>(pts)));
    }
    *mask = intersect;
  }
}

void cull_gaussian_cuda(uint32_t N, float *mean, float *qvec, float *svec,
                        float *normals, float *pts, bool *mask, float thresh) {
  static constexpr uint32_t N_THREAD = 256;
  const uint32_t n_blocks = div_round_up(N, N_THREAD);
  cull_gaussian_kernel<<<n_blocks, N_THREAD>>>(N, mean, qvec, svec, normals,
                                               pts, mask, thresh);
}

void cull_gaussian(uint32_t N, Tensor mean, Tensor qvec, Tensor svec,
                   Tensor normals, Tensor pts, Tensor mask, float thresh) {
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
  cull_gaussian_cuda(N, mean.data_ptr<float>(), qvec.data_ptr<float>(),
                     svec.data_ptr<float>(), normals.data_ptr<float>(),
                     pts.data_ptr<float>(), mask.data_ptr<bool>(), thresh);
}

__global__ void ScreenSpaceGaussians(Tensor mean, Tensor qvec, Tensor svec,
                                     Tensor) {
  // project 3d gaussians onto screen plane
}
