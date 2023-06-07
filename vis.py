import argparse
from gs.renderer import GaussianRenderer
from utils.viewer.viser_viewer import ViserViewer
from omegaconf import OmegaConf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_model", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=8080)

    opt = parser.parse_args()

    renderer = GaussianRenderer.load(opt.saved_model)

    viewer_conf = {
        "viewer_port": opt.port,
        "device": opt.device,
    }
    viewer_conf = OmegaConf.create(viewer_conf)

    viewer = ViserViewer(viewer_conf)