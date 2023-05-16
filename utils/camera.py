from pathlib import Path
import numpy as np
import torch
from .colmap import read_images, read_cameras, read_pts_from_colmap

"""Using OpenCV coordinates"""


class PerspectiveCameras:
    def __init__(self, c2ws, fx, fy, cx, cy, w, h) -> None:
        self.c2ws = c2ws
        self.n_cams = self.c2ws.shape[0]
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = w / h

    def to(self, device):
        self.c2ws = self.c2ws.to(device)

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
        right_normal = torch.cross(far_point + half_hside * right, up)

        up_normal = torch.cross(right, far_point + half_vside * up)
        down_normal = torch.cross(right, far_point - half_vside * up)

        pts = [near_point, far_point, t, t, t, t]
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

        return normals, pts


def get_data(cfg):
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    rot, t, images = read_images(image_bin, cfg.image_dir)
    pts, rgb = read_pts_from_colmap(point_bin,)
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
        cams = PerspectiveCameras(c2ws, float(params[0]), float(params[1]), float(params[2]), float(params[3]), camera.width, camera.height)

    return cams, images, pts, rgb