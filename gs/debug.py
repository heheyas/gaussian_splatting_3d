from typing import Any
import cv2
from utils.camera import PerspectiveCameras, CameraInfo
from utils.transforms import qsvec2rotmat_batched
from utils.misc import print_info, tic, toc, save_img
from utils.vis.basic import draw_heatmap_of_num_gaussians_per_tile
import numpy as np
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt

try:
    import _gs as _backend
except ImportError:
    from .backend import _backend

from .renderer import render, project_gaussians


def get_c2w_from_up_and_look_at(up, look_at, pos):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    y = np.cross(z, x)

    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos

    return c2w


class MockRenderer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.FloatTensor([[0.0, 0.0, 0.0], [0.1, 0.07, 0.0]]))
        self.qvec = torch.nn.Parameter(torch.FloatTensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
        self.log_svec = torch.nn.Parameter(torch.FloatTensor([[1.0, 1.0, 1.0], [1.1, 1.1, 1.1]]))
        self.log_svec.data *= np.log(80e-3)
        self.color = torch.nn.Parameter(torch.FloatTensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])) 
        self.alpha = torch.nn.Parameter(torch.FloatTensor([10000, 10000]))

        # self.mean = torch.nn.Parameter(self.mean[:1])
        # self.qvec = torch.nn.Parameter(self.qvec[:1])
        # self.color =torch.nn.Parameter(self.color[:1])
        # self.alpha =torch.nn.Parameter(self.alpha[:1])
        # self.log_svec = torch.nn.Parameter(self.log_svec[:1])

        x = torch.linspace(-5, -5, 1)
        y = torch.linspace(-10, 10, 40)
        z = torch.linspace(-10, 10, 50)
        x, y, z = torch.meshgrid(x, y, z)
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        mean = torch.stack([x, y, z], dim=1)
        self.mean = torch.nn.Parameter(mean)
        self.N = self.mean.shape[0]
        self.qvec = torch.nn.Parameter(torch.FloatTensor([[1.0, 0.0, 0.0, 0.0]] * self.N))
        self.log_svec = torch.nn.Parameter(torch.FloatTensor([[1.0, 1.0, 1.0]] * self.N))
        self.log_svec.data *= np.log(120e-3)
        color = torch.randn(self.N, 3).clamp(min=0.0, max=1.0)
        self.color = torch.nn.Parameter(color)
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0.0] * self.N))
        
        
        self.cfg = cfg
        self.device = cfg.device
        self.N = self.mean.shape[0]
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

        self.num_gaussians_bkp = num_gaussians.clone()

        tiledepth = torch.zeros(
            self.total_dub_gaussians, dtype=torch.float64, device=self.device
        )
        offset = torch.zeros(n_tiles + 1, dtype=torch.int32, device=self.device)
        gaussian_ids = torch.zeros(
            self.total_dub_gaussians, dtype=torch.int32, device=self.device
        )

        print("depth", depth)

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
        print("total dub gaussians", self.total_dub_gaussians)
        offset[-1] = self.total_dub_gaussians
        print_info(num_gaussians, "num_gaussians")
        self.offset = offset
        self.num_gaussians = num_gaussians
        self.gaussian_ids = gaussian_ids

        self.mean_2d = mean.detach().clone()
        self.radius_2d = radius.detach().clone() * self.tile_culling_radius
        self.cov_2d = cov.detach().clone()

        if self.cfg.debug:
            _backend.debug_check_tiledepth(offset.cpu(), tiledepth.cpu())

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

        return out

    def test_basic_alias(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        out = self.forward(c2w, camera_info)
        torch.nn.functional.mse_loss(out, torch.zeros_like(out)).backward()

        print_info(out, "out")

        img = (out.cpu().detach().numpy() * 255.).astype(np.uint8)
        
        # color = (255, 255, 255)
        # mean_2d = self.mean_2d.cpu().numpy()
        # radius_2d = (self.radius_2d.cpu().numpy() * camera_info.fx).astype(np.int32)
        
        # for i in range(self.N):
        #     x, y = (int(mean_2d[i, 0] * camera_info.fx + camera_info.cx), int(mean_2d[i, 1] * camera_info.fy + camera_info.cy))
        #     img = cv2.circle(img, (x, y), radius_2d[i], color, 1)

        # # print(img.max())
        # # print(img.min())

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)

        
        # n_tiles = n_tiles_h * n_tiles_w
        
        # num_error_samples = 0

        # for t in range(n_tiles):
        #     start = self.offset[t]
        #     end = self.offset[t + 1]
        #     assert end - start <= 2
        #     if end - start == 2:
        #         if self.gaussian_ids[start] == self.gaussian_ids[start + 1]:
        #             num_error_samples += 1

        # print("num_error_samples", num_error_samples)
        

        heatmap = draw_heatmap_of_num_gaussians_per_tile("test_alias_heatmap.png", self.tile_size, self.num_gaussians.cpu().numpy(), n_tiles_h, n_tiles_w, H, W, False)
        # for i in range(self.N):
        #     x, y = (int(mean_2d[i, 0] * camera_info.fx + camera_info.cx), int(mean_2d[i, 1] * camera_info.fy + camera_info.cy))
        #     heatmap = cv2.circle(heatmap, (x, y), radius_2d[i], color, 1)
        

        cv2.imwrite("./tmp/test_basic_alias.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./tmp/test_basic_alias_heatmap.png", heatmap)

        # print(self.mean_2d[0])
        # print(self.cov_2d[0])

        # print(self.mean_2d[1])
        # print(self.cov_2d[1])
        print_info(self.num_gaussians, "num_gaussians")

        assert (self.num_gaussians == self.num_gaussians_bkp).all()