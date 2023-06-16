from typing import Any
import cv2
from pathlib import Path
from utils.camera import PerspectiveCameras
from utils.transforms import qsvec2rotmat_batched, qvec2rotmat_batched
from utils.misc import print_info, tic, toc
import numpy as np
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from rich.console import Console
from utils.activations import activations, inv_activations
from gs.culling import tile_culling_aabb_count
from utils.misc import lineprofiler
from timeit import timeit
from time import time
from .initialize import cov_init

console = Console()

try:
    import _gs as _backend
except ImportError:
    from .backend import _backend

from gs.renderer import render_sh, step_check, project_gaussians


sh_base = 0.28209479177387814


@torch.no_grad()
def init_sh_coeffs(cfg, rgb: TensorType["N, 3"], C: int):
    sh_coeffs = torch.zeros((rgb.shape[0], 3, C * C), device=rgb.device)
    sh_coeffs[:, :, 0] = inv_activations["sigmoid"](rgb) / sh_base

    return sh_coeffs


class SHRenderer(torch.nn.Module):
    def __init__(self, cfg, pts=None, rgb=None):
        super().__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.max_C = cfg.sh_order

        self.svec_act = activations[cfg.svec_act]
        self.alpha_act = activations[cfg.alpha_act]

        self.svec_inv_act = inv_activations[cfg.svec_act]
        self.alpha_inv_act = inv_activations[cfg.alpha_act]

        if pts is not None and rgb is not None:
            # self.N = pts.shape[0]
            # self.mean = torch.nn.parameter.Parameter(pts)
            # self.qvec = torch.nn.Parameter(torch.zeros([self.N, 4]))
            # self.qvec.data[..., 0] = 1.0
            # self.sh_coeffs = torch.nn.Parameter(init_sh_coeffs(cfg, rgb, self.max_C))
            # self.svec_before_activation = torch.nn.Parameter(torch.ones([self.N, 3]))
            # self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))
            # self.svec_before_activation.data.fill_(self.svec_inv_act(cfg.svec_init))
            # self.alpha_before_activation.data.fill_(self.alpha_inv_act(cfg.alpha_init))
            self.initialize(cfg, pts, rgb)

        self.now_C = 1

        self.cum_grad = cfg.get("cum_grad", False)
        if self.cum_grad:
            self.register_buffer("max_grad_mean", torch.zeros_like(self.mean[..., 0]))
            # self.register_buffer("max_grad_qvec", torch.zeros_like(self.qvec))
            # self.register_buffer(
            #     "max_grad_svec", torch.zeros_like(self.svec_before_activation)
            # )
            # self.register_buffer(
            #     "max_grad_alpha", torch.zeros_like(self.alpha_before_activation)
            # )
            # self.register_buffer("max_grad_sh_coeffs", torch.zeros_like(self.sh_coeffs))

        self.set_cfg(cfg)

    def initialize(self, cfg, pts, rgb):
        self.N = pts.shape[0]
        self.mean = torch.nn.parameter.Parameter(pts)
        self.qvec = torch.nn.Parameter(torch.zeros([self.N, 4]))
        self.qvec.data[..., 0] = 1.0
        self.sh_coeffs = torch.nn.Parameter(init_sh_coeffs(cfg, rgb, self.max_C))

        svec_init_method = cfg.get("svec_init_method", "nearest")
        if svec_init_method == "fixed":
            self.svec_before_activation = torch.nn.Parameter(torch.ones([self.N, 3]))
            self.svec_before_activation.data.fill_(self.svec_inv_act(cfg.svec_init))
        elif svec_init_method == "nearest":
            init_svec = (cov_init(pts, cfg.get("nearest_k", 3)) * 10).clamp(
                max=0.1, min=0.01
            )
            print_info(init_svec, "init_svec")
            self.svec_before_activation = torch.nn.Parameter(
                self.svec_inv_act(init_svec).unsqueeze(1).repeat(1, 3)
            )
        else:
            raise NotImplementedError

        self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))
        self.alpha_before_activation.data.fill_(self.alpha_inv_act(cfg.alpha_init))

    def set_cfg(self, cfg):
        # camera imaging params
        self.tile_size = cfg.tile_size

        # frustum culling params
        self.frustum_culling_radius = cfg.frustum_culling_radius

        # tile culling params
        self.tile_culling_type = cfg.tile_culling_type
        self.tile_culling_radius = cfg.tile_culling_radius
        self.tile_culling_thresh = cfg.tile_culling_thresh

        # rendering params
        self.T_thresh = cfg.T_thresh

        # adaptive control params
        self.warm_up = cfg.get("warm_up", 1000)
        self.adaptive_control_iteration = cfg.adaptive_control_iteration
        self.pos_grad_thresh = cfg.pos_grad_thresh
        self.split_scale_thresh = cfg.split_scale_thresh
        self.scale_shrink_factor = cfg.scale_shrink_factor
        # !! deprecated
        self.alpha_reset_period = cfg.alpha_reset_period
        self.alpha_reset_val = cfg.alpha_reset_val
        self.alpha_thresh = cfg.alpha_thresh

        self.remove_tiny_period = cfg.get("remove_tiny_period", 500)
        self.remove_tiny = cfg.get("remove_tiny", False)

        # SH
        self.sh_upgrades = cfg.sh_upgrades

    def forward(self, c2w, camera_info):
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.svec,
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.svec[mask].contiguous()
        sh_coeffs = self.sh_coeffs[mask].contiguous()
        alpha = self.alpha[mask].contiguous()

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        self.depth = depth
        self.radius = None

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean,
            cov,
            self.tile_size,
            camera_info,
            self.tile_culling_radius,
        )
        toc("count N with dub")

        self.total_dub_gaussians = N_with_dub

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        start = -torch.ones([n_tiles], dtype=torch.int32, device=self.device)
        end = -torch.ones([n_tiles], dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy
        gaussian_ids = torch.zeros([N_with_dub], dtype=torch.int32, device=self.device)

        tic()
        _backend.tile_culling_aabb_start_end(
            aabb_topleft,
            aabb_bottomright,
            gaussian_ids,
            start,
            end,
            depth,
            n_tiles_h,
            n_tiles_w,
        )
        toc("tile culling aabb")

        if self.cfg.debug:
            print_info(end - start, "num_gaussians_per_tile")

        tic()
        out = render_sh(
            mean,
            cov,
            sh_coeffs[..., : self.now_C * self.now_C].contiguous(),
            alpha,
            start,
            end,
            gaussian_ids,
            img_topleft,
            c2w,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            self.now_C,
            self.T_thresh,
        )
        toc("render sh")

        return out.view(H, W, 3)

    @property
    def svec(self):
        return self.svec_act(self.svec_before_activation)

    @property
    def alpha(self):
        return self.alpha_act(self.alpha_before_activation)

    @torch.no_grad()
    def log(self, writer, step):
        self.log_depth_and_radius(writer, step)
        self.log_bounds(writer, step)
        self.log_info(writer, step)
        self.log_grad_bounds(writer, step)
        self.log_n_gaussian_dub(writer, step)

    @torch.no_grad()
    def log_depth_and_radius(self, writer, step):
        """log the bounds of the parameters"""
        if self.depth is not None:
            writer.add_scalar("bounds/depth_max", self.depth.max(), step)
            writer.add_scalar("bounds/depth_min", self.depth.min(), step)
            writer.add_scalar("bounds/depth_mean", self.depth.mean(), step)

        if self.radius is not None:
            writer.add_scalar("bounds/radius_mean", self.radius.mean(), step)
            writer.add_scalar("bounds/radius_max", self.radius.max(), step)
            writer.add_scalar("bounds/radius_min", self.radius.min(), step)

    @torch.no_grad()
    def log_bounds(self, writer, step):
        """log the bounds of the parameters"""
        writer.add_scalar("bounds/mean_max", self.mean.max(), step)
        writer.add_scalar("bounds/mean_min", self.mean.min(), step)
        writer.add_scalar("bounds/qvec_max", self.qvec.max(), step)
        writer.add_scalar("bounds/qvec_min", self.qvec.min(), step)
        writer.add_scalar("bounds/svec_max", self.svec.max(), step)
        writer.add_scalar("bounds/svec_min", self.svec.min(), step)
        writer.add_scalar(
            "bounds/color_max",
            self.sh_coeffs[..., : self.now_C * self.now_C].max(),
            step,
        )
        writer.add_scalar(
            "bounds/color_min",
            self.sh_coeffs[..., : self.now_C * self.now_C].min(),
            step,
        )
        writer.add_scalar("bounds/alpha_max", self.alpha.max(), step)
        writer.add_scalar("bounds/alpha_min", self.alpha.min(), step)

    @torch.no_grad()
    def log_info(self, writer, step):
        writer.add_scalar("info/mean_mean", self.mean.abs().mean(), step)
        writer.add_scalar("info/qvec_mean", self.qvec.abs().mean(), step)
        writer.add_scalar("info/svec_mean", self.svec.abs().mean(), step)
        writer.add_scalar(
            "info/sh_coeffs_mean",
            self.sh_coeffs[..., : self.now_C * self.now_C].abs().mean(),
            step,
        )
        writer.add_scalar("info/alpha_mean", self.alpha.sigmoid().mean(), step)

    @torch.no_grad()
    def log_grad_bounds(self, writer, step):
        if self.mean.grad is None:
            return
        writer.add_scalar("grad_bounds/mean_max", self.mean.grad.max(), step)
        writer.add_scalar("grad_bounds/mean_min", self.mean.grad.min(), step)
        writer.add_scalar("grad_bounds/qvec_max", self.qvec.grad.max(), step)
        writer.add_scalar("grad_bounds/qvec_min", self.qvec.grad.min(), step)
        writer.add_scalar(
            "grad_bounds/svec_max", self.svec_before_activation.grad.max(), step
        )
        writer.add_scalar(
            "grad_bounds/svec_min", self.svec_before_activation.grad.min(), step
        )
        writer.add_scalar(
            "grad_bounds/sh_coeffs_max",
            self.sh_coeffs.grad[..., : self.now_C * self.now_C].max(),
            step,
        )
        writer.add_scalar(
            "grad_bounds/sh_coeffs_min",
            self.sh_coeffs.grad[..., : self.now_C * self.now_C].min(),
            step,
        )
        writer.add_scalar(
            "grad_bounds/alpha_max", self.alpha_before_activation.grad.max(), step
        )
        writer.add_scalar(
            "grad_bounds/alpha_min", self.alpha_before_activation.grad.min(), step
        )

    def log_n_gaussian_dub(self, writer, step):
        writer.add_scalar("n_gaussian_dub", self.total_dub_gaussians, step)

    def split_gaussians(self):
        assert (
            self.mean.grad is not None
        ), "mean.grad is None while clone or split gaussians are performed according to spatial gradient of mean"
        console.print("[red bold]Splitting Gaussians")

        # all the gaussians need split or clone
        if self.cum_grad:
            mean_mask = self.max_grad_mean > self.pos_grad_thresh
        else:
            mean_mask = self.mean.grad.norm(dim=-1) > self.pos_grad_thresh
        svec_mask = (self.svec.data > self.split_scale_thresh).any(dim=-1)
        # gaussians need split are with large spatial scale
        split_mask = torch.logical_and(mean_mask, svec_mask)
        # gaussians need clone are with small spatial scale
        clone_mask = torch.logical_and(mean_mask, torch.logical_not(split_mask))

        num_split = split_mask.sum().item()
        num_clone = clone_mask.sum().item()

        console.print(f"[red bold]num_split {num_split} num_clone {num_clone}")

        # number of gaussians after split and clone will be increased by num_split + num_clone
        num_new_gaussians = num_split + num_clone

        # split
        split_mean = self.mean.data[split_mask].repeat(2, 1)
        split_qvec = self.qvec.data[split_mask].repeat(2, 1)
        split_svec = self.svec.data[split_mask].repeat(2, 1)
        # split_svec_ba = self.svec_before_activation.data[split_mask].repeat(2, 1)
        split_sh_coeffs = self.sh_coeffs.data[split_mask].repeat(2, 1, 1)
        split_alpha_ba = self.alpha_before_activation.data[split_mask].repeat(2)
        split_rotmat = qvec2rotmat_batched(split_qvec).transpose(-1, -2)

        split_gn = torch.randn(num_split * 2, 3, device=self.mean.device) * split_svec

        split_sampled_mean = split_mean + torch.einsum(
            "bij, bj -> bi", split_rotmat, split_gn
        )

        # check left product or right product

        old_num_gaussians = self.N
        unchanged_gaussians = old_num_gaussians - num_split
        self.N += num_new_gaussians
        console.print(f"[red bold]num gaussians: {self.N}")

        new_mean = torch.zeros([self.N, 3], device=self.device)
        new_qvec = torch.zeros([self.N, 4], device=self.device)
        new_svec = torch.zeros([self.N, 3], device=self.device)
        new_sh_coeffs = torch.zeros(
            [self.N, 3, self.max_C * self.max_C], device=self.device
        )
        new_alpha = torch.zeros([self.N], device=self.device)

        # copy old gaussians (# old_N - num_split)
        new_mean[:unchanged_gaussians] = self.mean.data[~split_mask]
        new_qvec[:unchanged_gaussians] = self.qvec.data[~split_mask]
        new_svec[:unchanged_gaussians] = self.svec_before_activation.data[~split_mask]
        new_sh_coeffs[:unchanged_gaussians] = self.sh_coeffs.data[~split_mask]
        new_alpha[:unchanged_gaussians] = self.alpha_before_activation.data[~split_mask]

        # clone gaussians
        new_mean[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.mean.data[clone_mask]
        new_qvec[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.qvec.data[clone_mask]
        new_svec[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.svec_before_activation.data[clone_mask]
        new_sh_coeffs[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.sh_coeffs.data[clone_mask]
        new_alpha[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.alpha_before_activation.data[clone_mask]

        pts = unchanged_gaussians + num_clone

        new_mean[pts : pts + 2 * num_split] = split_sampled_mean
        new_qvec[pts : pts + 2 * num_split] = split_qvec
        new_sh_coeffs[pts : pts + 2 * num_split] = split_sh_coeffs
        new_alpha[pts : pts + 2 * num_split] = split_alpha_ba
        new_svec[pts : pts + 2 * num_split] = self.svec_inv_act(
            split_svec / self.scale_shrink_factor
        )

        assert pts + 2 * num_split == self.N

        self.mean = torch.nn.Parameter(new_mean)
        self.qvec = torch.nn.Parameter(new_qvec)
        self.svec_before_activation = torch.nn.Parameter(new_svec)
        self.sh_coeffs = torch.nn.Parameter(new_sh_coeffs)
        self.alpha_before_activation = torch.nn.Parameter(new_alpha)

        if self.cum_grad:
            self.max_grad_mean = torch.zeros_like(self.mean[..., 0])

    def remove_low_alpha_gaussians(self):
        mask = self.alpha_act(self.alpha_before_activation.data) >= self.alpha_thresh
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.sh_coeffs = torch.nn.Parameter(self.sh_coeffs.data[mask])
        self.alpha_before_activation = torch.nn.Parameter(
            self.alpha_before_activation.data[mask]
        )
        self.svec_before_activation = torch.nn.Parameter(
            self.svec_before_activation.data[mask]
        )

        removed = self.N - self.mean.shape[0]
        self.N = self.mean.shape[0]
        console.print("[yellow]remove_low_alpha_gaussians[/yellow]")
        console.print(
            f"[yellow]removed {removed} gaussians, remaining {self.N} gaussians"
        )

    def reset_alpha(self):
        console.print("[yellow]reset alpha[/yellow]")
        self.alpha_before_activation.data.fill_(
            self.alpha_inv_act(self.alpha_reset_val)
        )
        with torch.no_grad():
            print_info(self.alpha, "alpha")

    def remove_tiny_gaussians(self):
        mask = (self.svec > self.svec_tiny_thresh).all(dim=-1)
        removed = self.N - mask.sum().item()
        console.print(f"[red bold]removed {removed} gaussians[/red bold]")
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.sh_coeffs = torch.nn.Parameter(self.sh_coeffs.data[mask])
        self.alpha_before_activation = torch.nn.Parameter(
            self.alpha_before_activation.data[mask]
        )
        self.svec_before_activation = torch.nn.Parameter(
            self.svec_before_activation.data[mask]
        )

    def update_max_grads(self):
        self.max_grad_mean = torch.maximum(
            self.max_grad_mean, self.mean.grad.norm(dim=-1)
        )

    def adaptive_control(self, epoch):
        if epoch < self.warm_up:
            return

        if epoch in self.sh_upgrades:
            self.now_C += 1
            self.now_C = min(self.now_C, self.max_C)

        self.update_max_grads()

        if step_check(epoch, self.adaptive_control_iteration):
            self.split_gaussians()

        if step_check(epoch, self.alpha_reset_period):
            self.remove_low_alpha_gaussians()
            self.reset_alpha()

        if self.remove_tiny and step_check(epoch, self.remove_tiny_period):
            self.remove_tiny_gaussians()

    def save(self, path):
        path = Path(path)
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        state_dict = {
            "mean": self.mean.data,
            "qvec": self.qvec.data,
            "svec_before_activation": self.svec_before_activation.data,
            "sh_coeffs": self.sh_coeffs.data,
            "alpha_before_activation": self.alpha_before_activation.data,
            "N": self.N,
            "cfg": self.cfg,
        }

        torch.save(state_dict, path)

    @classmethod
    def load(cls, path):
        state_dict = torch.load(path)

        renderer = cls(state_dict["cfg"])
        renderer.N = state_dict["N"]
        renderer.mean = torch.nn.Parameter(state_dict["mean"])
        renderer.qvec = torch.nn.Parameter(state_dict["qvec"])
        renderer.svec_before_activation = torch.nn.Parameter(
            state_dict["svec_before_activation"]
        )
        renderer.sh_coeffs = torch.nn.Parameter(state_dict["sh_coeffs"])
        renderer.alpha_before_activation = torch.nn.Parameter(
            state_dict["alpha_before_activation"]
        )

        assert renderer.N == renderer.mean.shape[0]

        return renderer

    def get_param_groups(self):
        param_groups = {
            "mean": self.mean,
            "qvec": self.qvec,
            "svec": self.svec_before_activation,
            "sh_coeffs": self.sh_coeffs,
            "alpha": self.alpha_before_activation,
        }
        return param_groups

    def get_optimizer(self):
        lr = self.cfg.lr

        opt_params = []

        param_groups = self.get_param_groups()
        for name, params in param_groups.items():
            opt_params.append({"params": params, "lr": self.cfg.get(f"{name}_lr", lr)})

        return torch.optim.Adam(opt_params, lr=lr)
