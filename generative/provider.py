import torch
import numpy as np
from utils.camera import CameraInfo


def get_c2w_from_up_and_look_at(up, look_at, pos):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos

    return c2w


class CameraPoseProvider:
    def __init__(self, cfg):
        self.cfg = cfg.camera_pose_provider

        self.center = self.cfg.get("center", [0.0, 0.0, 0.0])
        self.center = np.array(self.center)

        self.vrot_range = self.cfg.get("vrot_range", [0, 90])
        self.hrot_range = self.cfg.get("hrot_range", [0, 360])
        self.radius = self.cfg.get("radius", 1.5)
        self.up = np.array(self.cfg.get("up", [0.0, 0.0, 1.0]))
        self.focal_range = self.cfg.get("focal_range", [0.5, 1.5])
        self.center_jittor_std = self.cfg.get("center_jittor_std", 0.0)
        self.near_plane = self.cfg.get("near_plane", 0.1)
        self.far_plane = self.cfg.get("far_plane", 100.0)

        self.radius_range = self.cfg.get("radius_range", [2.5, 2.5])
        self.real_uniform = self.cfg.get("real_uniform", True)

        self.default_reso = self.cfg.get("default_reso", 512)

        self.horizontal_warmup = self.cfg.get("horizontal_warmup", 0)
        self.horizontal_warmup = max(self.horizontal_warmup, 1)
        print("horizontal_warmup", self.horizontal_warmup)
        print("vrot_range", self.vrot_range)
        print("hrot_range", self.hrot_range)
        print("focal_range", self.focal_range)
        print("real uniform", self.real_uniform)

        self.hrot_bound = (
            lambda x: min(1.0, x / self.horizontal_warmup) * self.hrot_range[1]
        )

    def sample_one(self, e, reso=None):
        if reso is None:
            reso = self.default_reso
        if self.real_uniform:
            sampled_uniform = np.random.rand()
            vertical_rotation = np.arccos(1 - sampled_uniform)
        else:
            sampled_uniform = np.random.uniform(self.vrot_range[0], self.vrot_range[1])
            vertical_rotation = np.deg2rad(sampled_uniform)

        horizontal_rotation = np.deg2rad(
            np.random.uniform(
                0,
                self.hrot_bound(e),
            )
        )

        assert np.abs(horizontal_rotation) <= np.abs(self.hrot_range[1])
        radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
        x = radius * np.sin(vertical_rotation) * np.cos(horizontal_rotation)
        y = radius * np.sin(vertical_rotation) * np.sin(horizontal_rotation)
        z = radius * np.cos(vertical_rotation)

        pos = np.array([x, y, z]) + self.center

        c2w = torch.from_numpy(
            get_c2w_from_up_and_look_at(
                self.up,
                self.center + np.random.randn(3) * self.center_jittor_std,
                pos,
            )
        ).to(torch.float32)

        focal = np.random.uniform(self.focal_range[0], self.focal_range[1]) * reso

        camera_info = CameraInfo(
            focal,
            focal,
            reso / 2.0,
            reso / 2.0,
            reso,
            reso,
            self.near_plane,
            self.far_plane,
        )

        return (
            camera_info,
            c2w,
            np.rad2deg(vertical_rotation),
            np.rad2deg(horizontal_rotation),
        )
