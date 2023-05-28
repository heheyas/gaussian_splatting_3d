from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .colmap import read_images, read_cameras, read_pts_from_colmap
from .misc import print_info
from .transforms import rotmat2wxyz

"""Using OpenCV coordinates"""


class PerspectiveCameras:
    def __init__(self, c2ws, fx, fy, cx, cy, w, h, distortion=None) -> None:
        self.c2ws = c2ws
        self.n_cams = self.c2ws.shape[0]
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = w / h
        self.distortion = distortion

    def to(self, device):
        self.c2ws = self.c2ws.to(device)
        self.c2ws = self.c2ws.to(torch.float32)

    def get_frustum_pts(self, idx, near_plane, far_plane):
        if idx < 0 or idx >= self.n_cams:
            raise ValueError

        c2w = self.c2ws[idx]

        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = near_plane * lookat + t
        far_point = far_plane * lookat + t
        near_normal = lookat
        far_normal = -lookat

        left_normal = torch.cross(far_point - half_hside * right, up)
        right_normal = torch.cross(far_point + half_hside * right, up)

        up_normal = torch.cross(right, far_point + half_vside * up)
        down_normal = torch.cross(right, far_point - half_vside * up)

        corners = []
        for a in [-1, 1]:
            for b in [-1, 1]:
                corners.append(
                    near_point
                    + a * half_hside * near_plane / far_plane * right
                    + b * half_vside * near_plane / far_plane * up
                )

        for a in [-1, 1]:
            for b in [-1, 1]:
                corners.append(far_point + a * half_hside * right + b * half_vside * up)

        return corners

    # depreatied
    # def get_tile_frustum(self, idx, near_plane, far_plane, tile_size=16):
    #     if not tile_size == 16:
    #         raise NotImplementedError

    def get_frustum(self, idx, near_plane, far_plane):
        if idx < 0 or idx >= self.n_cams:
            raise ValueError

        c2w = self.c2ws[idx]

        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = near_plane * lookat
        far_point = far_plane * lookat
        near_normal = lookat
        far_normal = -lookat

        left_normal = torch.cross(far_point - half_hside * right, up)
        right_normal = torch.cross(up, far_point + half_hside * right)

        up_normal = torch.cross(far_point + half_vside * up, right)
        down_normal = torch.cross(right, far_point - half_vside * up)

        pts = [near_point + t, far_point + t, t, t, t, t]
        normals = [
            near_normal,
            far_normal,
            left_normal,
            right_normal,
            up_normal,
            down_normal,
        ]

        pts = torch.stack(pts, dim=0)
        normals = torch.stack(normals, dim=0)
        normals = F.normalize(normals, dim=-1)

        return normals, pts

    def get_all_rays_o(self, idx):
        xp = (torch.arange(0, self.w, dtype=torch.float32) - self.cx) / self.fx
        yp = (torch.arange(0, self.h, dtype=torch.float32) - self.cy) / self.fy

        xp, yp = torch.meshgrid(xp, yp, indexing="ij")
        xp = xp.reshape(-1)
        yp = yp.reshape(-1)
        padding = torch.ones_like(xp)

        xyz_cam = torch.stack([xp, yp, padding], dim=-1)

        rot = self.c2ws[idx][:3, :3]
        t = self.c2ws[idx][:3][3]

        xyz_world = t + torch.einsum("ij,bj->bi", rot, xyz_cam)

        return xyz_world

    def prepare(self):
        pixel_size_x = 1.0 / self.fx
        pixel_size_y = 1.0 / self.fy

    def get_camera_wxyz(self, idx: int):
        return rotmat2wxyz(self.c2ws[idx][:3, :3].contiguous())

    def get_camera_pos(self, idx: int):
        return self.c2ws[idx][:3, 3]


def get_data(cfg):
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    rot, t, images = read_images(image_bin, cfg.image_dir)
    pts, rgb = read_pts_from_colmap(
        point_bin,
    )
    t = np.expand_dims(t, axis=-1)
    camera = read_cameras(cam_bin)
    params = camera.params
    # print(rot.shape)
    # print(t.shape)
    # print(images.shape)

    c2ws = np.concatenate([rot, t], axis=2)
    c2ws = torch.from_numpy(c2ws)

    cams = None

    if camera.model == "OPENCV":
        cams = PerspectiveCameras(
            c2ws,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
        )
    elif camera.model == "OPENCV_FISHEYE":
        # TODO: add fisheye camera support
        cams = PerspectiveCameras(
            c2ws,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
        )

    print(
        f"camera:\n\tfx: {cams.fx}; fy: {cams.fy}\n\tcx: {cams.cx}; cy: {cams.cy}\n\tH: {cams.h}; W: {cams.w}"
    )

    return cams, images, pts, rgb


def in_frustum(queries, normal, pts):
    is_in = torch.ones_like(queries[..., 0], dtype=torch.bool)
    for i in range(6):
        in_test = torch.einsum("bj,j->b", queries - pts[i], normal[i]) > 0.0
        is_in = torch.logical_and(is_in, in_test)

    return is_in
