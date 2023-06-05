from threading import Thread
import numpy as np
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from utils.camera import CameraInfo


class RenderThread(Thread):
    pass


class ViserViewer:
    def __init__(self, cfg: OmegaConf):
        self.port = cfg.viewer.port

        self.server = viser.ViserServer(port=self.port)
