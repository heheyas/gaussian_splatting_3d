from typing import Any
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
    """
    just for test
    """

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
        print_info(projected_cov[..., 0, 0], "projected_cov[..., 0, 0]")

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

    def tile_partition_bcircle(
        self, mean, radius, camera: PerspectiveCameras, cov=None, thresh=0.1
    ):
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
        # _backend.count_num_gaussians_each_tile(
        #     mean,
        #     cov,
        #     img_topleft,
        #     self.tile_size,
        #     n_tiles_h,
        #     n_tiles_w,
        #     pixel_size_x,
        #     pixel_size_y,
        #     num_gaussians,
        #     thresh,
        # )
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
        # _backend.image_sort(
        #     gaussian_ids,
        #     self.tiledepth,
        #     depth,
        #     self.num_gaussians,
        #     self.offset,
        #     mean,
        #     cov,
        #     img_topleft,
        #     self.tile_size,
        #     n_tiles_h,
        #     n_tiles_w,
        #     pixel_size_x,
        #     pixel_size_y,
        #     0.01,
        # )
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

        print(cov.shape)
        print_info(
            cov[..., 0, 0],
            "cov[..., 0, 0]",
        )
        print_info(
            cov[..., 1, 1],
            "cov[..., 1, 1]",
        )
        print_info(torch.det(cov), "det(cov)")

        thresh = 0.001
        tic()
        _backend.tile_based_vol_rendering(
            mean,
            cov,
            # torch.inverse(cov).contiguous(),
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
        print("nans:", torch.count_nonzero(torch.isnan(out)).item())

        return out

    def render_loop(self):
        pass


@torch.no_grad()
def jacobian(u):
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


def project_pts(pts: torch.Tensor, c2w: TensorType["N", 3, 4]):
    d = -c2w[..., :3, 3]
    W = torch.transpose(c2w[..., :3, :3], -1, -2)

    return torch.einsum("ij,bj->bi", W, pts + d)


def project_gaussians(
    mean: TensorType["N", 3],
    qvec: TensorType["N", 4],
    svec: TensorType["N", 3],
    c2w: TensorType[3, 4],
):
    projected_mean = project_pts(mean, c2w)
    rotmat = qsvec2rotmat_batched(qvec, svec)
    sigma = rotmat @ torch.transpose(rotmat, -1, -2)
    W = torch.transpose(c2w[:3, :3], -1, -2)
    J = jacobian(projected_mean)
    JW = torch.einsum("bij,jk->bik", J, W)
    assert JW.grad is None, "JW should not be updated"
    projected_cov = torch.bmm(torch.bmm(JW, sigma), torch.transpose(JW, -1, -2))[
        ..., :2, :2
    ].contiguous()
    depth = projected_mean[..., 2:].clone().contiguous()
    projected_mean = (projected_mean[..., :2] / projected_mean[..., 2:]).contiguous()

    return projected_mean, projected_cov, JW, depth


class _render(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        color,
        alpha,
        offset,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
    ):
        out = torch.empty([H * W * 3], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        ctx.save_for_backward(
            mean, cov, color, alpha, offset, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        mean, cov, color, alpha, offset, gaussian_ids, out, topleft = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        _backend.tile_based_vol_rendering_backward(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


render = _render.apply


class GaussianRenderer(torch.nn.Module):
    def __init__(self, cfg, pts, rgb):
        super().__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.N = pts.shape[0]
        self.mean = torch.nn.parameter.Parameter(pts)
        self.qvec = torch.nn.Parameter(torch.zeros([self.N, 4]))
        self.qvec.data[..., 0] = 1.0
        self.log_svec = torch.nn.Parameter(
            torch.ones([self.N, 3]) * np.log(cfg.svec_init)
        )
        self.color = torch.nn.Parameter(rgb)
        self.alpha = torch.nn.Parameter(torch.ones([self.N]) * cfg.alpha_init)

        self.near_plane = cfg.near_plane
        self.far_plane = cfg.far_plane
        self.tile_size = cfg.tile_size
        self.frustum_culling_radius = cfg.frustum_culling_radius
        self.tile_culling_type = cfg.tile_culling_type
        self.tile_culling_radius = cfg.tile_culling_radius
        self.tile_culling_thresh = cfg.tile_culling_thresh
        self.T_thresh = cfg.T_thresh

    def forward(self, c2w, camera_info):
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        # alpha = self.alpha[mask].contiguous()
        # alpha = torch.nn.functional.sigmoid(self.alpha[mask].contiguous())
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        cov = (cov + cov.transpose(-1, -2)) / 2.0
        with torch.no_grad():
            m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
            p = torch.det(cov)
            radius = torch.sqrt(m + torch.sqrt((m.pow(2) - p).clamp(min=0.0)))

        if self.cfg.debug:
            print_info(radius, "radius")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        tic()
        with torch.no_grad():
            if self.tile_culling_type == "bcircle":
                _backend.count_num_gaussians_each_tile_bcircle(
                    mean,
                    radius * self.tile_culling_radius,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                    num_gaussians,
                )
            elif self.tile_culling_type == "prob":
                _backend.count_num_gaussians_each_tile(
                    mean,
                    cov,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                    num_gaussians,
                    self.tile_culling_thresh,
                )
            else:
                raise NotImplementedError
        toc("tile culling")

        self.total_dub_gaussians = num_gaussians.sum().item()

        tiledepth = torch.zeros(
            self.total_dub_gaussians, dtype=torch.float64, device=self.device
        )
        offset = torch.zeros(n_tiles + 1, dtype=torch.int32, device=self.device)
        gaussian_ids = torch.zeros(
            self.total_dub_gaussians, dtype=torch.int32, device=self.device
        )

        tic()
        with torch.no_grad():
            if self.tile_culling_type == "bcircle":
                _backend.prepare_image_sort(
                    gaussian_ids,
                    tiledepth,
                    depth,
                    num_gaussians,
                    offset,
                    mean,
                    radius * self.tile_culling_radius,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                )
            elif self.tile_culling_type == "prob":
                _backend.image_sort(
                    gaussian_ids,
                    tiledepth,
                    depth,
                    num_gaussians,
                    offset,
                    mean,
                    cov,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                    self.tile_culling_thresh,
                )
            else:
                raise NotImplementedError
        toc("radix sort")
        offset[-1] = self.total_dub_gaussians

        if self.cfg.debug:
            _backend.debug_check_tiledepth(offset.cpu(), tiledepth.cpu())

        tic()
        out = render(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            self.T_thresh,
        ).view(H, W, 3)
        toc("render")

        return out

    def split_gaussians(self):
        pass

    def clone_gaussians(self):
        pass

    def remove_low_alpha_gaussians(self):
        pass

    def adaptive_control(self):
        pass

    def check_grad(self):
        print_info(self.qvec.grad, "grad_qvec")
        print_info(self.log_svec.grad, "grad_svec")

    def check_info(self):
        # print_info(self.qvec, "qvec")
        # print_info(self.svec, "svec")
        pass

    def log_n_gaussian_dub(self, writer, step):
        writer.add_scalar("n_gaussian_dub", self.total_dub_gaussians, step)

    @torch.no_grad()
    def log_grad_bounds(self, writer, step):
        writer.add_scalar("grad_bounds/mean_max", self.mean.grad.max(), step)
        writer.add_scalar("grad_bounds/mean_min", self.mean.grad.min(), step)
        writer.add_scalar("grad_bounds/qvec_max", self.qvec.grad.max(), step)
        writer.add_scalar("grad_bounds/qvec_min", self.qvec.grad.min(), step)
        writer.add_scalar("grad_bounds/log_svec_max", self.log_svec.grad.max(), step)
        writer.add_scalar("grad_bounds/log_svec_min", self.log_svec.grad.min(), step)
        writer.add_scalar("grad_bounds/color_max", self.color.grad.max(), step)
        writer.add_scalar("grad_bounds/color_min", self.color.grad.min(), step)
        writer.add_scalar("grad_bounds/alpha_max", self.alpha.grad.max(), step)
        writer.add_scalar("grad_bounds/alpha_min", self.alpha.grad.min(), step)

    @torch.no_grad()
    def log_info(self, writer, step):
        writer.add_scalar("info/mean_mean", self.mean.abs().mean(), step)
        writer.add_scalar("info/qvec_mean", self.qvec.abs().mean(), step)
        writer.add_scalar("info/svec_mean", self.log_svec.exp().mean(), step)
        writer.add_scalar("info/color_mean", self.color.abs().mean(), step)
        writer.add_scalar("info/alpha_mean", self.alpha.abs().mean(), step)

    def sanity_check(self):
        print()
