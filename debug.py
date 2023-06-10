import os
import gc
import cv2
import torch
import numpy as np
import hydra
from gs.renderer import GaussianRenderer
from gs.debug import MockRenderer
from utils.camera import get_c2ws_and_camera_info, get_c2ws_and_camera_info_v1
from kornia.losses import ssim_loss
from functools import partial
from utils.misc import print_info, save_img, tic, toc
from utils import misc
from utils.loss import get_loss_fn
import visdom
from collections import deque

from rich.console import Console
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
import datetime

console = Console()


@hydra.main(config_path="conf", config_name="debug")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    renderer = MockRenderer(cfg).to(cfg.device)
    # renderer.test_basic_alias()
    # renderer.test_aabb()
    method = f"renderer.{cfg.fn}"
    eval(method)()


if __name__ == "__main__":
    main()
