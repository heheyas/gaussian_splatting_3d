#include "data_spec.hpp"
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
namespace device {
struct PackedGaussian {
  PackedGaussian(Gaussian &G)
      : qvec(G.qvec.data_ptr<float>()), svec(G.svec.data_ptr<float>()),
        mean(G.mean.data_ptr<float>()), color(G.color.data_ptr<float>()),
        alpha(G.alpha.data_ptr<float>()){};
  float *__restrict__ qvec;
  float *__restrict__ svec;
  float *__restrict__ mean;
  float *__restrict__ color;
  float *__restrict__ alpha;
};

struct Plane {
  float3 pts;
  float3 normal;
};

struct Frustum {
  Plane topFace;
  Plane bottomFace;

  Plane rightFace;
  Plane leftFace;

  Plane farFace;
  Plane nearFace;
};

struct PackedCamera {
  PackedCamera(Camera &cam)
      : c2w(cam.c2w.packed_accessor32<float, 2, torch::RestrictPtrTraits()>()),
        fx(cam.fx), fy(cam.fy), cx(cam.cx), cy(cam.cy), width(cam.width),
        height(cam.height) {}
  /* data */
  const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits()> c2w;
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;
};

Frustum createFrustumFromCamera(const PackedCamera &cam, RenderOption &opt) {
  Frustum frustum;
  const float aspect = cam.height / cam.width;
  const float halfVSide = opt.far_plane * cam.height / (2 * cam.fy);
  const float halfHSide = halfVSide * aspect;

  float3 pos(cam.c2w[0][3], cam.c2w[1][3], cam.c2w[2][3]);
  float3 up(-cam.c2w[0][1], -cam.c2w[1][1], -cam.c2w[2][1]);
  float3 right(cam.c2w[0][0], cam.c2w[1][0], cam.c2w[2][0]);
  float3 front(cam.c2w[0][2], cam.c2w[1][2], cam.c2w[2][2]);

  float3 frontMultFar = opt.near_plane * front;

  frustum.nearFace = {pos + opt.near_plane * front, front};
  frustum.farFace = {pos + frontMultFar, -front};
  frustum.rightFace = {pos, cross(frontMultFar - right * halfHSide, up)};
  frustum.leftFace = {pos, cross(up, frontMultFar + right * halfHSide)};
  frustum.topFace = {pos, cross(right, frontMultFar - up * halfVSide)};
  frustum.bottomFace = {pos, cross(frontMultFar + up * halfVSide, right)};

  return frustum;
}
} // namespace device
} // namespace