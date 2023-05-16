import sys
import pytest
import hydra
from hydra import compose, initialize
from utils.camera import get_data

def config():
    initialize(config_path=".", job_name="test")
    cfg = compose(config_name="test")

    return cfg

def test_get_data():
    get_data(config())


def test_culling():
    camera, images, pts, rgb = get_data(config())
    frustum = camera.get_frustum(0, 0.1, 1000)

if __name__ == "__main__":
    task = sys.argv[1]
    eval(task)()