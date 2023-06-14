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
from utils.misc import print_info, save_img, tic, toc, lineprofiler
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


@hydra.main(config_path="conf", config_name="garden_sh")
def main(cfg):
    logger = logging.getLogger(__name__)
    if cfg.viewer:
        logger.info("Viewer enabled")
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
        misc._timing_ = True
        console.print(f"[red bold]timimg enabled")
        tic()
        toc("test")

    c2ws, camera_info, images, pts, rgb = get_c2ws_and_camera_info_v1(cfg)
    c2ws = c2ws.to(cfg.device).contiguous()
    renderer = SHRenderer(cfg, pts, rgb).to(cfg.device)

    if cfg.debug:
        avg_radius = np.linalg.norm(c2ws.cpu().numpy()[:, :3, 3], axis=-1).mean()
        console.print(f"[red bold]avg_radius: {avg_radius:.2f}")

    N_images = images.shape[0]
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).to(torch.float32)
    # loss_fn = torch.nn.functional.mse_loss
    loss_fn = get_loss_fn(cfg)

    opt = torch.optim.Adam(renderer.parameters(), lr=cfg.lr)

    if cfg.get("viewer", False):
        from utils.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(cfg)
        viewer.set_renderer(renderer)
        viewer_eval_fps = 1000.0

    log_iteration = cfg.get("log_iteration", 50)
    only_forward = cfg.get("only_forward", False)
    warm_up = cfg.get("warm_up", 1000)

    with tqdm(total=cfg.max_iteration) as pbar:
        for e in range(cfg.max_iteration):
            i = e % N_images
            tic()
            out = renderer(c2ws[i], camera_info)
            toc("whole renderer forward")
            if e == 0:
                print_info(out, "out")
                print("num total gaussian", renderer.total_dub_gaussians)
                if cfg.debug:
                    exit(0)
            if only_forward:
                pbar.update(1)
                continue
            gt = images[i].to(cfg.device)

            loss = loss_fn(out, gt)
            opt.zero_grad()
            tic()
            with torch.autograd.profiler.profile(enabled=cfg.timing) as prof:
                loss.backward()
            toc("backward")
            if e % log_iteration == 0:
                save_img(
                    out.cpu().clamp(min=0.0, max=1.0),
                    f"./tmp/{cfg.data_name}_sh",
                    f"train_{e}.png",
                )
                save_img(
                    gt.cpu().clamp(min=0.0, max=1.0),
                    f"./tmp/{cfg.data_name}_sh",
                    f"gt_{e}.png",
                )
                if cfg.debug:
                    exit(0)
                writer.add_scalar("loss", loss.item(), e)
                writer.add_image(
                    "out", out.cpu().moveaxis(-1, 0).clamp(min=0, max=1.0), e
                )
                renderer.log(writer, e)
            opt.step()

            if e > warm_up:
                renderer.adaptive_control(e)
            opt = torch.optim.Adam(renderer.parameters(), lr=cfg.lr)

            # logger.info(f"Iteration: {e}/{cfg.max_iteration} loss: {loss.item():.4f}")
            pbar.set_description(f"Iteration: {e}/{cfg.max_iteration}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if cfg.viewer:
                while viewer.pause_training:
                    viewer.update()
                    time.sleep(1.0 / viewer_eval_fps)
                if e % viewer.train_viewer_update_period_slider.value == 0:
                    viewer.update()

    if cfg.timing:
        print(prof.key_averages())


if __name__ == "__main__":
    main()
