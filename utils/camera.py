from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .colmap import read_images, read_cameras, read_pts_from_colmap, read_images_test
from .misc import print_info
from .transforms import rotmat2wxyz
from rich.console import Console

console = Console()

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

    rot = rot.transpose(0, 2, 1)
    t = -rot @ t
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
    elif camera.model == "PINHOLE":
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
    else:
        print("Not support camera model: ", camera.model)
        raise NotImplementedError

    print(
        f"camera:\n\tfx: {cams.fx}; fy: {cams.fy}\n\tcx: {cams.cx}; cy: {cams.cy}\n\tH: {cams.h}; W: {cams.w}"
    )

    return cams, images, pts, rgb


class CameraInfo:
    def __init__(self, fx, fy, cx, cy, w, h, near_plane, far_plane) -> None:
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = w / h
        self.near_plane = near_plane
        self.far_plane = far_plane

    def downsample(self, scale):
        self.fx /= scale
        self.fy /= scale
        self.cx /= scale
        self.cy /= scale
        self.w //= scale
        self.h //= scale

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = self.w / self.h

    def get_frustum(self, c2w):
        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = self.far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = self.near_plane * lookat
        far_point = self.far_plane * lookat
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

    def print_camera_info(self):
        console.print(
            f"[blue underline]camera:\n\tfx: {self.fx:.2f}; fy: {self.fy:.2f}\n\tcx: {self.cx:.2f}; cy: {self.cy:.2f}\n\tH: {self.h}; W: {self.w}\n\tpixel_size: {1 / self.fx:.4g} pixel_size_y: {1 / self.fy:.4g}"
        )


def in_frustum(queries, normal, pts):
    is_in = torch.ones_like(queries[..., 0], dtype=torch.bool)
    for i in range(6):
        in_test = torch.einsum("bj,j->b", queries - pts[i], normal[i]) > 0.0
        is_in = torch.logical_and(is_in, in_test)

    return is_in


def get_c2ws_and_camera_info(cfg):
    console.print("[bold green]Loading camera info...")
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    points_cached = base / "colmap" / "sparse" / "0" / "points_and_rgb.pt"
    console.print("[bold green]Loading images...")
    rot, t, images = read_images(image_bin, cfg.image_dir)
    console.print("[bold green]Loading points...")
    if points_cached.exists():
        console.print("[bold green]Loading cached points...")
        pts, rgb = torch.load(points_cached)
    else:
        pts, rgb = read_pts_from_colmap(
            point_bin,
        )
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
        torch.save((pts, rgb), points_cached)
    t = np.expand_dims(t, axis=-1)
    camera = read_cameras(cam_bin)
    params = camera.params
    # print(rot.shape)
    # print(t.shape)
    # print(images.shape)

    rot = rot.transpose(0, 2, 1)
    t = -rot @ t
    c2ws = np.concatenate([rot, t], axis=2).astype(np.float32)
    c2ws = torch.from_numpy(c2ws)

    cams = None
    console.print(f"[green bold]camera model: {camera.model}")

    if camera.model == "OPENCV" or camera.model == "PINHOLE":
        camera_info = CameraInfo(
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
            cfg.near_plane,
            cfg.far_plane,
        )
    elif camera.model == "OPENCV_FISHEYE":
        # TODO: add fisheye camera support
        raise NotImplementedError
    else:
        print("Not support camera model: ", camera.model)
        raise NotImplementedError

    console.print(f"[blue underline]downsample: {cfg.downsample}")

    camera_info.downsample(cfg.downsample)

    assert images.shape[0] == c2ws.shape[0]
    if (
        abs(camera_info.w - images.shape[2]) > 1
        or abs(camera_info.h - images.shape[1]) > 1
    ):
        console.print("[red bold]camera image size not match")
        exit(-1)

    if (camera_info.h != images.shape[1]) or (camera_info.w != images.shape[2]):
        console.print(
            "[red bold]camera image size not match due to round caused by downsample"
        )
        camera_info.h = images.shape[1]
        camera_info.w = images.shape[2]

    console.print(
        f"[blue underline]camera:\n\tfx: {camera_info.fx:.2f}; fy: {camera_info.fy:.2f}\n\tcx: {camera_info.cx:.2f}; cy: {camera_info.cy:.2f}\n\tH: {camera_info.h}; W: {camera_info.w}"
    )

    if isinstance(pts, np.ndarray):
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
    c2ws = c2ws.to(cfg.device)

    return c2ws, camera_info, images, pts, rgb


def get_c2ws_and_camera_info_test(cfg):
    console.print("[bold green]Loading camera info...")
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    points_cached = base / "colmap" / "sparse" / "0" / "points_and_rgb.pt"
    console.print("[bold green]Loading images...")
    rot, t, _ = read_images_test(image_bin, cfg.image_dir)
    console.print("[bold green]Loading points...")
    if points_cached.exists():
        console.print("[bold green]Loading cached points...")
        pts, rgb = torch.load(points_cached)
    else:
        pts, rgb = read_pts_from_colmap(
            point_bin,
        )
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
        torch.save((pts, rgb), points_cached)
    t = np.expand_dims(t, axis=-1)
    camera = read_cameras(cam_bin)
    params = camera.params

    rot = rot.transpose(0, 2, 1)
    t = -rot @ t
    c2ws = np.concatenate([rot, t], axis=2).astype(np.float32)
    c2ws = torch.from_numpy(c2ws)

    cams = None
    console.print(f"[green bold]camera model: {camera.model}")

    if camera.model == "OPENCV" or camera.model == "PINHOLE":
        camera_info = CameraInfo(
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
            cfg.near_plane,
            cfg.far_plane,
        )
    elif camera.model == "OPENCV_FISHEYE":
        # TODO: add fisheye camera support
        raise NotImplementedError
    else:
        print("Not support camera model: ", camera.model)
        raise NotImplementedError

    console.print(f"[blue underline]downsample: {cfg.downsample}")

    camera_info.downsample(cfg.downsample)

    N = rot.shape[0]
    images = np.zeros([N, camera_info.h, camera_info.w, 3], dtype=np.float32)

    assert images.shape[0] == c2ws.shape[0]
    if (
        abs(camera_info.w - images.shape[2]) > 1
        or abs(camera_info.h - images.shape[1]) > 1
    ):
        console.print("[red bold]camera image size not match")
        exit(-1)

    if (camera_info.h != images.shape[1]) or (camera_info.w != images.shape[2]):
        console.print(
            "[red bold]camera image size not match due to round caused by downsample"
        )
        camera_info.h = images.shape[1]
        camera_info.w = images.shape[2]

    # console.print(
    #     f"[blue underline]camera:\n\tfx: {camera_info.fx:.2f}; fy: {camera_info.fy:.2f}\n\tcx: {camera_info.cx:.2f}; cy: {camera_info.cy:.2f}\n\tH: {camera_info.h}; W: {camera_info.w}"
    # )
    camera_info.print_camera_info()

    if isinstance(pts, np.ndarray):
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
    c2ws = c2ws.to(cfg.device)

    return c2ws, camera_info, images, pts, rgb
