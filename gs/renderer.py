import cv2
from utils.camera import PerspectiveCameras
from utils.transforms import qsvec2rotmat_batched
from utils.misc import print_info, tic, toc
import numpy as np
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt

try:
    import _gs as _backend
except ImportError:
    from .backend import _backend


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
        print(l.shape)
        print(l.max())
        print(l.min())
        J = torch.zeros(u.size(0), 3, 3).to(u)
        J[..., 0, 0] = 1.0 / u[..., 2]
        J[..., 2, 0] = u[..., 0] / l
        J[..., 1, 1] = 1.0 / u[..., 2]
        J[..., 2, 1] = u[..., 1] / l
        J[..., 0, 2] = -u[..., 0] / u[..., 2] / u[..., 2]
        J[..., 1, 2] = -u[..., 1] / u[..., 2] / u[..., 2]
        J[..., 2, 2] = u[..., 2] / l
        print_info(torch.det(J), "det(J)")

        return J

    def project_gaussian(
        self,
        mean: TensorType["N", 3],
        qvec: TensorType["N", 4],
        svec: TensorType["N", 3],
        camera: PerspectiveCameras,
        idx: int,
    ):
        projected_mean = self.project_pts(mean, camera.c2ws[idx]).contiguous()  # [N, 3]
        print_info(projected_mean[..., 2], "projected_mean_z")

        # test
        # projected_mean /= projected_mean[..., 2:]

        rotmat = qsvec2rotmat_batched(qvec, svec)
        # 3d gaussian paper eq (6)
        sigma = rotmat @ torch.transpose(rotmat, -1, -2)

        print_info(sigma, "sigma")

        W = torch.transpose(camera.c2ws[idx][:3, :3], -1, -2)
        print_info(W, "W")
        J = self.jacobian(projected_mean)
        print_info(J, "J")
        JW = torch.einsum("bij,jk->bik", J, W)

        projected_cov = torch.bmm(torch.bmm(JW, sigma), torch.transpose(JW, -1, -2))[
            ..., :2, :2
        ].contiguous()
        # projected_cov += torch.eye(2).to(projected_cov)
        print_info(projected_cov, "projected_cov")

        depth = projected_mean[..., 2:].clone().contiguous()

        projected_mean = (
            projected_mean[..., :2] / projected_mean[..., 2:]
        ).contiguous()

        return projected_mean, projected_cov, JW, depth

    def tile_partition(
        self, mean, cov_inv, camera: PerspectiveCameras, thresh: float = 0.2
    ):
        H, W = camera.h, camera.w
        print(f"H: {H} and W: {W}")
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w

        print(f"n_tiles_h: {n_tiles_h} and n_tiles_w: {n_tiles_w}")

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        img_bottomright = torch.FloatTensor(
            [(W - camera.cx) / camera.fx, (H - camera.cy) / camera.fy]
        ).to("cuda")

        print(f"img_topleft: {img_topleft}")
        print(f"img_bottomright: {img_bottomright}")

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy

        _backend.count_num_gaussians_each_tile(
            mean,
            cov_inv,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            num_gaussians,
            thresh,
        )
        print(num_gaussians.sum().item())

        print_info(num_gaussians, "num_gaussians")

    def tile_partition_bcircle(self, mean, radius, camera: PerspectiveCameras):
        H, W = camera.h, camera.w
        print(f"H: {H} and W: {W}")
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        n_tiles = n_tiles_h * n_tiles_w

        print(f"n_tiles_h: {n_tiles_h} and n_tiles_w: {n_tiles_w}")

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        img_bottomright = torch.FloatTensor(
            [(W - camera.cx) / camera.fx, (H - camera.cy) / camera.fy]
        ).to("cuda")

        print(f"img_topleft: {img_topleft}")
        print(f"img_bottomright: {img_bottomright}")

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy

        tic()
        _backend.count_num_gaussians_each_tile_bcircle(
            mean,
            radius,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            num_gaussians,
        )
        toc()
        print(num_gaussians.sum().item())

        self.total_gaussians = num_gaussians.sum().item()
        self.num_gaussians = num_gaussians

        self.tiledepth = torch.zeros(
            self.total_gaussians, dtype=torch.float64, device="cuda"
        )

        print(f"total_gaussians: {self.total_gaussians}")
        print_info(num_gaussians, "num_gaussians")

    def image_level_radix_sort(
        self, mean, cov, radius, depth, color, camera: PerspectiveCameras
    ):
        print("=" * 10, "image_level_radix_sort", "=" * 10)
        H, W = camera.h, camera.w
        print(f"H: {H} and W: {W}")
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w

        print(f"n_tiles_h: {n_tiles_h} and n_tiles_w: {n_tiles_w}")

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        img_bottomright = torch.FloatTensor(
            [(W - camera.cx) / camera.fx, (H - camera.cy) / camera.fy]
        ).to("cuda")

        print(f"img_topleft: {img_topleft}")
        print(f"img_bottomright: {img_bottomright}")
        print(self.num_gaussians.shape)

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy
        self.offset = torch.zeros(n_tiles + 1, dtype=torch.int32, device="cuda")
        print_info(self.offset, "offset")
        print_info(self.num_gaussians, "num_gaussians")
        print_info(self.tiledepth, "tiledepth")

        num_gaussians_bkp = self.num_gaussians.clone()
        gaussian_ids = torch.zeros(
            self.total_gaussians, dtype=torch.int32, device="cuda"
        )

        tic()
        _backend.prepare_image_sort(
            gaussian_ids,
            self.tiledepth,
            depth,
            self.num_gaussians,
            self.offset,
            mean,
            radius,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
        )
        toc()
        self.offset[-1] = self.total_gaussians

        print_info(self.offset, "offset")
        print_info(self.num_gaussians, "num_gaussians")
        print(f"original num_gaussians: {num_gaussians_bkp.sum().item()}")
        print(f"sorted num_gaussians: {self.num_gaussians.sum().item()}")
        n_gaussians_check = (
            (self.num_gaussians == num_gaussians_bkp).count_nonzero().item()
        )
        print(f"n_gaussians_check: {n_gaussians_check}")

        print_info(self.tiledepth, "tiledepth")
        print_info(gaussian_ids, "gaussian_ids")
        print_info(self.offset, "offset")
        diff = self.offset[1:] - self.offset[:-1]
        print_info(diff, "diff")

        out = torch.zeros([H * W * 3], dtype=torch.float32, device="cuda")
        # color = torch.ones(
        #     [self.total_gaussians, 3], dtype=torch.float32, device="cuda"
        # )
        alpha = (
            torch.ones([self.total_gaussians], dtype=torch.float32, device="cuda") * 1.0
        )

        print(self.offset[:100])
        print(gaussian_ids[:100])
        self.gaussian_ids = gaussian_ids

        thresh = 0.01
        tic()
        _backend.tile_based_vol_rendering(
            mean,
            # cov,
            torch.inverse(cov).contiguous(),
            color,
            alpha,
            self.offset,
            gaussian_ids,
            out,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        toc()
        # fig, ax = plt.subplots()
        print_info(out, "out")
        print(out.mean())
        print(out.std())
        # ax.imshow(out.reshape(H, W, 3).cpu().numpy())
        # plt.show()
        # fig.savefig("out.png")

        img = (out.reshape(H, W, 3).cpu().numpy() * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("tmp/forward_out.png", img)

        return out

    def render_loop(self):
        pass
