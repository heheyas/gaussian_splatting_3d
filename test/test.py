import numpy as np
import time
import sys
import torch
import hydra
from hydra import compose, initialize
from utils.camera import get_data, in_frustum
from utils.misc import print_info
import viser
import plotly.graph_objects as go


def config():
    initialize(config_path=".", job_name="test")
    cfg = compose(config_name="test")

    return cfg


def test_get_data():
    get_data(config())


def draw_frustum():
    camera, images, pts, rgb = get_data(config())
    # normals, pts = camera.get_frustum(0, 0.1, 10)
    corners = camera.get_frustum_pts(0, 5, 10)
    x, y, z = torch.stack(corners, dim=0).unbind(dim=-1)
    x = x.numpy()
    y = y.numpy()
    z = z.numpy()

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="markers")])

    fig.show()


def test_culling():
    camera, images, pts, rgb = get_data(config())
    normals, pts = camera.get_frustum(0, 1, 10)
    x = torch.linspace(-20, 20, 200)
    y, z = x.clone(), x.clone()
    x, y, z = torch.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xyz = torch.stack([x, y, z], dim=1)

    is_in = in_frustum(xyz, normals, pts)
    print(is_in.sum())
    xyz = xyz[is_in]
    x, y, z = xyz.unbind(dim=-1)
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                                mode='markers')])

    # fig.show()
    server = viser.ViserServer()

    def vis():
        server.add_point_cloud(
            name="/frustum/pcd",
            points=xyz.numpy(),
            colors=np.zeros_like(xyz.numpy()),
        )

    while True:
        vis()

        time.sleep(1e-3)


def test_culling_on_real():
    camera, images, xyz, rgb = get_data(config())
    normals, pts = camera.get_frustum(1, 0.1, 1000)
    N = pts.shape[0]
    qvec = torch.zeros([N, 4], dtype=torch.float32)
    qvec[..., 0] = 1.0
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.3
    mask = torch.zeros(N, dtype=torch.bool).to("cuda")
    xyz = torch.from_numpy(xyz)
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera = camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")

    is_in = in_frustum(xyz, normals, pts)
    print(is_in.sum().item())
    xyz = xyz[is_in]
    x, y, z = xyz.unbind(dim=-1)
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                                mode='markers')])
    xyz = xyz.cpu()

    # fig.show()
    server = viser.ViserServer()

    def vis():
        server.add_point_cloud(
            name="/frustum/pcd",
            points=xyz.numpy(),
            colors=np.zeros_like(xyz.numpy()),
        )

    while True:
        vis()

        time.sleep(1e-3)


def test_cuda_culling():
    camera, images, pts, rgb = get_data(config())
    normals, pts = camera.get_frustum(0, 1, 10)
    x = torch.linspace(-20, 20, 200)
    y, z = x.clone(), x.clone()
    x, y, z = torch.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xyz = torch.stack([x, y, z], dim=1)

    import _renderer

    mask = torch.ones(xyz.shape[0], dtype=torch.bool).to("cuda")
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera = camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)

    from renderer.backend import _backend

    _backend.cull_gaussian(xyz.shape[0], xyz, xyz, xyz, normals, pts, mask)

    xyz = xyz.cpu()
    mask = mask.cpu()
    print(mask.sum())

    server = viser.ViserServer()

    def vis():
        server.add_point_cloud(
            name="/frustum/pcd",
            points=xyz.numpy(),
            colors=np.zeros_like(xyz.numpy()),
        )

    while True:
        vis()

        time.sleep(1e-3)


def test_gs_culling():
    camera, images, pts, rgb = get_data(config())
    normals, pts = camera.get_frustum(0, 1, 10)
    x = torch.linspace(-20, 20, 200)
    y, z = x.clone(), x.clone()
    x, y, z = torch.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xyz = torch.stack([x, y, z], dim=1)

    N = xyz.shape[0]
    qvec = torch.zeros([N, 4], dtype=torch.float32)
    qvec[..., 0] = 1.0
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.0
    mask = torch.zeros(xyz.shape[0], dtype=torch.bool).to("cuda")
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera = camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")

    from gs.backend import _backend

    _backend.culling_gaussian_bsphere(xyz, qvec, svec, normals, pts, mask, 1.0)

    xyz = xyz.cpu()
    mask = mask.cpu()
    print(mask.sum())

    server = viser.ViserServer()

    def vis():
        server.add_point_cloud(
            name="/frustum/pcd",
            points=xyz.numpy(),
            colors=np.zeros_like(xyz.numpy()),
        )

    while True:
        vis()

        time.sleep(1e-3)


def test_gs_project():
    camera, images, pts, rgb = get_data(config())
    normals, pts = camera.get_frustum(0, 1, 10)

    from gs.renderer import Renderer

    renderer = Renderer()
    x = torch.linspace(-20, 20, 200)
    y, z = x.clone(), x.clone()
    x, y, z = torch.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xyz = torch.stack([x, y, z], dim=1)

    N = xyz.shape[0]
    qvec = torch.zeros([N, 4], dtype=torch.float32)
    qvec[..., 0] = 1.0
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.1
    mask = torch.zeros(xyz.shape[0], dtype=torch.bool).to("cuda")
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")

    mean, cov, JW = renderer.project_gaussian(xyz, qvec, svec, camera, 0)

    cov_inv = torch.inverse(cov)
    print(cov[0])


def gs_project_sanity_check():
    camera, images, pts, rgb = get_data(config())
    normals, pts = camera.get_frustum(0, 1, 10)

    from gs.renderer import Renderer

    renderer = Renderer()
    x = torch.linspace(-20, 20, 200)
    y, z = x.clone(), x.clone()
    x, y, z = torch.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xyz = torch.stack([x, y, z], dim=1)

    N = xyz.shape[0]
    qvec = torch.zeros([N, 4], dtype=torch.float32)
    qvec[..., 0] = 1.0
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.1
    mask = torch.zeros(xyz.shape[0], dtype=torch.bool).to("cuda")
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")
    mean, cov, JW = renderer.project_gaussian(xyz, qvec, svec, camera, 0)

    cov_inv = torch.inverse(cov)

    print(cov[0])
    print(cov_inv[0])


def gs_culling_test_on_llff():
    camera, images, xyz, rgb = get_data(config())
    normals, pts = camera.get_frustum(1, 0.1, 1000)
    N = xyz.shape[0]
    qvec = torch.zeros([N, 4], dtype=torch.float32)
    qvec[..., 0] = 1.0
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.1
    mask = torch.zeros(N, dtype=torch.bool).to("cuda")
    xyz = torch.from_numpy(xyz)
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")

    import _gs as _backend

    _backend.culling_gaussian_bsphere(xyz, qvec, svec, normals, pts, mask, 3.0)

    xyz = xyz.cpu().numpy()
    mask = mask.cpu()
    print(mask.sum())
    culled_xyz = xyz[mask]

    mask = mask.cpu().numpy()
    culled_rgb = rgb[mask]

    server = viser.ViserServer()
    culling = server.add_gui_checkbox("culling", True)

    vis_points = xyz
    vis_colors = rgb
    need_update = True

    @culling.on_update
    def _(_) -> None:
        nonlocal vis_points, vis_colors, need_update
        need_update = True
        if culling.value:
            vis_points = culled_xyz
            vis_colors = culled_rgb
        else:
            vis_points = xyz
            vis_colors = rgb

    def vis():
        server.add_point_cloud(
            name="/frustum_culling/pcd",
            points=vis_points,
            colors=vis_colors,
        )
        server.add_camera_frustum(
            name="/frustum_culling/frustum",
            aspect=camera.aspect,
            fov=camera.yfov,
            position=camera.get_camera_pos(1).cpu().numpy(),
            wxyz=camera.get_camera_wxyz(1).cpu().numpy(),
        )

    while True:
        if need_update:
            need_update = False
            server.reset_scene()
            vis()

        time.sleep(1e-3)


def gs_count_tile():
    from gs.renderer import Renderer

    renderer = Renderer()
    camera, images, xyz, rgb = get_data(config())
    normals, pts = camera.get_frustum(0, 0.1, 1000)
    N = xyz.shape[0]
    qvec = torch.zeros([N, 4], dtype=torch.float32)
    qvec[..., 0] = 1.0
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.03
    mask = torch.zeros(N, dtype=torch.bool).to("cuda")
    xyz = torch.from_numpy(xyz)
    xyz = xyz.to("cuda").to(torch.float32)

    camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")

    # from gs.backend import _backend
    import _gs as _backend

    _backend.culling_gaussian_bsphere(xyz, qvec, svec, normals, pts, mask, 3.0)
    print(mask.sum().item())

    xyz = xyz[mask].contiguous()
    print(xyz.shape)
    print(xyz.is_contiguous())
    qvec = qvec[mask].contiguous()
    svec = svec[mask].contiguous()

    mean, cov, JW, depth = renderer.project_gaussian(xyz, qvec, svec, camera, 0)

    print_info(mean, "mean")
    print_info(cov, "cov")
    print_info(depth, "depth")

    # make sure cov can be diagonalized
    cov = (cov + cov.transpose(-1, -2)) / 2.0
    m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
    p = torch.det(cov)
    radius = torch.sqrt(m + torch.sqrt((m.pow(2) - p).clamp(min=0.0)))
    print_info(radius, "radius")

    cov_inv = torch.inverse(cov).contiguous()
    print_info(cov_inv, "cov_inv")

    # renderer.tile_partition(mean, cov_inv, camera, 1.0)
    color = torch.from_numpy(rgb).to("cuda").to(torch.float32)[mask]
    print_info(color, "color")
    color = torch.zeros_like(color)
    renderer.tile_partition_bcircle(mean, radius * 1.0, camera)

    renderer.image_level_radix_sort(mean, cov, radius * 1.0, depth, color, camera)


def test_pytorch3d_cameras():
    from utils import p3d_helper

    cameras, images, xyz, rgb = p3d_helper.get_data(config())
    p3d_helper.render_pcd(cameras, xyz, rgb, 0, radius=0.03)


if __name__ == "__main__":
    task = sys.argv[1]
    eval(task)()
