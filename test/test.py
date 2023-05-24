import numpy as np
import time
import sys
import torch
import hydra
from hydra import compose, initialize
from utils.camera import get_data, in_frustum
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


def test_cuda_culling_full():
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
    svec = torch.ones([N, 1], dtype=torch.float32) * 100
    mask = torch.zeros(xyz.shape[0], dtype=torch.bool).to("cuda")
    xyz = xyz.to("cuda").to(torch.float32)
    print(xyz.shape)

    camera = camera.to("cuda")
    normals = normals.to("cuda").to(torch.float32)
    pts = pts.to("cuda").to(torch.float32)
    qvec = qvec.to("cuda")
    svec = svec.to("cuda")

    from renderer.backend import _backend

    _backend.cull_gaussian(xyz.shape[0], xyz, qvec, svec, normals, pts, mask, 0.0)

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
    svec = torch.ones([N, 3], dtype=torch.float32) * 0.1
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


if __name__ == "__main__":
    task = sys.argv[1]
    eval(task)()
