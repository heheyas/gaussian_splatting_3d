import numpy as np
from pathlib import Path
from .read import read_points3D_binary, read_cameras_binary, read_images_binary

def read_pts_from_colmap(filename, verbose=True):
    if not isinstance(filename, str):
        filename = str(filename)
    obj = read_points3D_binary(filename)
    pts = []
    rgb = []
    for o in obj.values():
        pts.append(o.xyz)
        rgb.append(rgb)

    if verbose:
        print(f"Read {len(obj)} points from {filename}")

    rgb = rgb.astype(np.float32) / 255.
        
    return np.array(pts), rgb


def colmap_to_open3d(filename):
    import open3d as o3d
    if not isinstance(filename, str):
        filename = str(filename)

    obj = read_points3D_binary(filename)
    
    xyz = []
    rgb = []
    
    for o in obj.values():
        xyz.append(o.xyz)
        rgb.append(o.rgb)

    xyz = np.array(xyz, dtype=np.float32)
    rgb = np.array(rgb, dtype=np.float32) / 255.

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd

def stats(filename):
    if not isinstance(filename, str):
        filename = str(filename)
    obj = read_points3D_binary(filename)

    num_pts = len(obj)
    num_visible_img = []

    for o in obj.values():
        num_visible_img.append(len(o.image_ids)) 
    
    avg_num_visible_img = np.mean(num_visible_img)

    print(f"File: {filename}, #(points): {num_pts}, avg visible times: {avg_num_visible_img:.2f}")

def read_cameras(filename):
    if not isinstance(filename, str):
        filename = str(filename)

    cameras = read_cameras_binary(filename)

def read_images(filename):
    if not isinstance(filename, str):
        filename = str(filename)
    images = read_images_binary(filename)
    rot = []
    T = []
    for img in images.values():
        rot.append(img.qvec2rotmat())
        T.append(img.tvec)

    return np.array(rot), np.array(T)

def read_all(data_dir):
    data_dir = Path(data_dir)
    cam_bin = data_dir / "cameras.bin"
    img_bin = data_dir / "images.bin"
    pts_bin = data_dir / "points3D.bin"

    pos, rgb = read_pts_from_colmap(pts_bin)
    rot, T = read_images(img_bin)
    cam = read_cameras(cam_bin)