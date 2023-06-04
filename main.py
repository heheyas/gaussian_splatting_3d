import os
import gc
import cv2
import torch
import numpy as np
import hydra
from gs.renderer import GaussianRenderer
from utils.camera import get_c2ws_and_camera_info
from kornia.losses import ssim_loss
from functools import partial
from utils.misc import print_info, save_img
from utils.loss import get_loss_fn
import visdom
from collections import deque

from rich.console import Console
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
import datetime

console = Console()


@hydra.main(config_path="conf", config_name="garden")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    writer = SummaryWriter(f"./logs/runs/{cfg.data_name}/{timestamp}")
    print(cfg.debug)
    if cfg.debug:
        global debug
        debug = True
        console.print(f"[blue underline]cwd: {os.getcwd()}")
        console.print(f"[red bold]debug mode")

    if cfg.timing:
        global _timing_
        _timing_ = True
        console.print(f"[red bold]timimg enabled")

    c2ws, camera_info, images, pts, rgb = get_c2ws_and_camera_info(cfg)
    renderer = GaussianRenderer(cfg, pts, rgb).to(cfg.device)

    N_images = images.shape[0]
    images = torch.from_numpy(images).to(torch.float32)
    loss_fn = torch.nn.functional.mse_loss
    # loss_fn = get_loss_fn(cfg)

    opt = torch.optim.Adam(renderer.parameters(), lr=cfg.lr)

    with tqdm(total=cfg.max_iteration) as pbar:
        for e in range(cfg.max_iteration):
            i = e % N_images
            out = renderer(c2ws[i], camera_info)
            gt = images[i].to(cfg.device)
            loss = loss_fn(out, gt)
            opt.zero_grad()
            loss.backward()
            if e % 50 == 0:
                save_img(
                    out.cpu(),
                    f"./tmp/{cfg.data_name}",
                    f"train_{e}.png",
                )
                writer.add_scalar("loss", loss.item(), e)
                writer.add_image(
                    "out", out.cpu().moveaxis(-1, 0).clamp(min=0, max=1.0), e
                )
                # renderer.log_grad_bounds(writer, e)
                # renderer.log_info(writer, e)
                # renderer.log_n_gaussian_dub(writer, e)
            # renderer.check_grad()
            opt.step()

            pbar.set_description(f"Iteration: {e}/{cfg.max_iteration}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)


if __name__ == "__main__":
    main()
