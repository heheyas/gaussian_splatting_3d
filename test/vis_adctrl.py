import os
import gc
import cv2
import torch
import time
import numpy as np
import hydra
from gs.renderer import GaussianRenderer
from gs.sh_renderer import SHRenderer
from utils.camera import get_c2ws_and_camera_info, get_c2ws_and_camera_info_v1
from functools import partial
from utils.misc import (
    print_info,
    save_img,
    tic,
    toc,
    lineprofiler,
    step_check,
    average_dicts,
)
from utils.metrics import Metrics
from utils import misc
from utils.loss import get_loss_fn
import visdom

from rich.console import Console
from tqdm import tqdm
import wandb

from torch.utils.tensorboard import SummaryWriter
import datetime
import logging

console = Console()


@hydra.main(config_path="conf", config_name="fern_sh")
@lineprofiler
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    c2ws, camera_info, images, pts, rgb, eval_mask = get_c2ws_and_camera_info_v1(cfg)
    c2ws = c2ws.to(cfg.device).contiguous()
    # renderer = SHRenderer(cfg, pts, rgb).to(cfg.device)

    train_images = images[~eval_mask].contiguous()
    eval_images = images[eval_mask].contiguous()

    eval_mask = eval_mask.to(cfg.device)
    train_c2ws = c2ws[~eval_mask].contiguous()
    eval_c2ws = c2ws[eval_mask].contiguous()

    N_train = train_images.shape[0]
    N_eval = eval_images.shape[0]

    console.print(f"[green bold]#(train images): {N_train} #(eval images): {N_eval}")

    if not cfg.from_ckpt:
        start = 0
        renderer = SHRenderer(cfg, pts, rgb).to(cfg.device)
    else:
        renderer = SHRenderer.load(cfg.ckpt_path, cfg).to(cfg.device)
        start = cfg.start_epoch
    loss_fn = get_loss_fn(cfg)
    metric_meter = Metrics(cfg.device)

    start_epoch = cfg.warm_up + 1

    opt = renderer.get_optimizer()

    if cfg.get("viewer", False):
        from utils.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(cfg)
        viewer.set_renderer(renderer)
        viewer_eval_fps = 1000.0

    e = start_epoch + 1
    i = e % N_train
    tic()
    out = renderer(train_c2ws[i], camera_info)
    gt = train_images[i].to(cfg.device)

    loss = loss_fn(out, gt)
    opt.zero_grad()
    tic()
    with torch.autograd.profiler.profile(enabled=cfg.timing) as prof:
        loss.backward()
    toc("backward")

    if cfg.adapt_ctrl_enabled:
        renderer.adaptive_control(e)

    renderer.vis_grads_gaussians(cfg.vis_grad_thresh)

    out = renderer(train_c2ws[i], camera_info)
    save_img(out, f"./tmp/vis_grads", f"{cfg.data_name}_{cfg.vis_grad_thresh}.png")


if __name__ == "__main__":
    main()
