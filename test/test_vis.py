from pathlib import Path
from utils.vis.viser import viser_vis_colmap

colmap_dir = Path("./data/floating-tree/colmap/sparse/0")
image_dir = Path("./data/floating-tree/images_8")

def test_viser_colmap():
    viser_vis_colmap(colmap_path=colmap_dir, images_path=image_dir)

if __name__ == "__main__":
    test_viser_colmap()