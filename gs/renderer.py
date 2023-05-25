from utils.camera import PerspectiveCameras
from utils.transforms import qsvec2rotmat_batched
import numpy as np
from torchtyping import TensorType
import torch


class Renderer:
    def __init__(self, tile_size=16) -> None:
        self.tile_size = tile_size

    def culling3d(self):
        # call _C to cull
        pass

    def project_pts(self, pts: torch.Tensor, c2w: TensorType["N", 3, 4]):
        d = -c2w[..., :3, 3]
        W = torch.transpose(c2w[..., :3, :3], -1, -2)

        return torch.einsum("ij,bj->bi", W, pts + d)

    def jacobian(self, u):
        l = torch.norm(u, dim=-1)
        J = torch.zeros(u.size(0), 3, 3).to(u)
        J[..., 0, 0] = 1.0 / u[..., 2]
        J[..., 2, 0] = u[..., 0] / l
        J[..., 1, 1] = 1.0 / u[..., 2]
        J[..., 2, 1] = u[..., 1] / l
        J[..., 0, 2] = -u[..., 0] / u[..., 2] / u[..., 2]
        J[..., 1, 2] = -u[..., 1] / u[..., 2] / u[..., 2]
        J[..., 2, 2] = u[..., 2] / l

        return J

    def project_gaussian(
        self,
        mean: TensorType["N", 3],
        qvec: TensorType["N", 4],
        svec: TensorType["N", 3],
        camera: PerspectiveCameras,
        idx: int,
    ):
        projected_mean = self.project_pts(mean, camera.c2ws[idx])  # [N, 3]
        rotmat = qsvec2rotmat_batched(qvec, svec)
        # 3d gaussian paper eq (6)
        sigma = rotmat @ torch.transpose(rotmat, -1, -2)

        W = torch.transpose(camera.c2ws[idx][:3, :3], -1, -2)
        J = self.jacobian(mean)
        JW = torch.einsum("bij,jk->bik", J, W)

        projected_cov = torch.bmm(torch.bmm(JW, sigma), torch.transpose(JW, -1, -2))[
            ..., :2, :2
        ]

        return projected_mean, projected_cov, JW

    def tile_partition(self, camera: PerspectiveCameras, idx: int):
        H, W = camera.h, camera.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy
