from utils.camera import PerspectiveCameras
import numpy as np
import torch


class Render:
    def __init__(self) -> None:
        pass

    def project_pts(self, pts: torch.Tensor, camera: PerspectiveCameras):
        d = -camera.c2ws[..., :3, 3]
        W = torch.transpose(camera.c2ws[..., :3, :3], -1, -2)

        return W @ pts + d

    def project_covariance(self, sigma: torch.Tensor, camera: PerspectiveCameras):
        pass
