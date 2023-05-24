import numpy as np
import torch
import torch.nn as nn

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Gaussian(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.qvec = None
        self.tvec = None
        self.mean = None
        self.color = None
        self.alpha = None

    def initialize(self, pts, rgb, cfg):
        # TODO: add init strategies
        self.n_gaussians = len(pts)
        self.mean = torch.nn.Parameter(pts)
        self.color = torch.nn.Parameter(rgb)
        self.qvec = torch.nn.Parameter([self.n_gaussians, 4])
        self.tvec = torch.nn.Parameter([self.n_gaussians, 3])
        self.alpha = torch.nn.Parameter([self.n_gaussians, 1])

    def is_in_frustum(self, camera):
        pass
