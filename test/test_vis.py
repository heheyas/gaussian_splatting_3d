from pathlib import Path
from utils.vis.viser import viser_vis_colmap

colmap_dir = Path("./data/garden/colmap/sparse/0")
image_dir = Path("./data/garden/images_8")


def test_viser_colmap():
    viser_vis_colmap(colmap_path=colmap_dir, images_path=image_dir)

def test_coord():
    import random
    import time

    import viser

    server = viser.ViserServer(port=9998)

    while True:
        # Add some coordinate frames to the scene. These will be visualized in the viewer.
        server.add_frame(
            "/tree",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        server.add_frame(
            "/tree/branch",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        leaf = server.add_frame(
            "/tree/branch/leaf",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        time.sleep(5.0)

        # Remove the leaf node from the scene.
        leaf.remove()
        time.sleep(0.5)


if __name__ == "__main__":
    test_viser_colmap()
    # test_coord()
