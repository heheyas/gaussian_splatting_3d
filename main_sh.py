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


@lineprofiler
def train_and_eval(cfg):
    logger = logging.getLogger(__name__)
    if cfg.viewer:
        logger.info("Viewer enabled")
    os.chdir(hydra.utils.get_original_cwd())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    if cfg.wandb:
        wandb.init(
            project="gs",
            config=cfg,
            name=f"{cfg.data_name}_{timestamp}",
            sync_tensorboard=True,
        )
    writer = SummaryWriter(
        f"./logs/runs/{cfg.data_name}/{timestamp}", comment=cfg.comment
    )
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

    opt = renderer.get_optimizer()

    if cfg.get("viewer", False):
        from utils.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(cfg)
        viewer.set_renderer(renderer)
        viewer_eval_fps = 1000.0

    log_iteration = cfg.get("log_iteration", 50)
    only_forward = cfg.get("only_forward", False)

    writer.add_text("cfg", str(cfg))
    writer.add_text("comment", cfg.comment)

    use_train_sample = cfg.get("use_train_sample", False)

    renderer.train()
    with tqdm(total=cfg.max_iteration) as pbar:
        for e in range(start, cfg.max_iteration):
            i = e % N_train
            tic()
            out = renderer(train_c2ws[i], camera_info)
            toc("whole renderer forward")
            if e == 0:
                print_info(out, "out")
                print("num total gaussian", renderer.total_dub_gaussians)
                if cfg.debug:
                    save_img(
                        out.cpu().clamp(min=0.0, max=1.0),
                        f"./tmp/debug",
                        f"train_{e}.png",
                    )
            if only_forward:
                pbar.update(1)
                continue
            gt = train_images[i].to(cfg.device)

            loss = loss_fn(out, gt)
            opt.zero_grad()
            tic()
            with torch.autograd.profiler.profile(enabled=cfg.timing) as prof:
                loss.backward()
            toc("backward")
            if step_check(e, log_iteration, True):
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

            if step_check(e, cfg.eval_iteration):
                renderer.eval()
                metric_dicts = []
                eval_losses = []
                # do eval
                with torch.no_grad():
                    for j in range(N_eval):
                        gt_eval_image = eval_images[j].to(cfg.device)
                        eval_out = renderer(eval_c2ws[j], camera_info)
                        eval_losses.append(loss_fn(eval_out, gt_eval_image).item())
                        metric_dicts.append(metric_meter(eval_out, gt_eval_image))

                    eval_loss = np.mean(eval_losses)
                    eval_metrics = average_dicts(metric_dicts)

                writer.add_scalar("eval/eval_loss", eval_loss, e)
                for key, value in eval_metrics.items():
                    writer.add_scalar(f"eval/{key}", value, e)
                writer.add_image(
                    f"eval/eval_img",
                    eval_out.cpu().moveaxis(-1, 0).clamp(min=0, max=1.0),
                    e,
                )

                info_line = f"[red bold]Iteration {e} Evaluation loss: {eval_loss:.4g}"
                for key, value in eval_metrics.items():
                    info_line += f" {key}: {value:.4g}"
                console.print(info_line)
                renderer.train()

            if cfg.adapt_ctrl_enabled:
                renderer.adaptive_control(e)

            # opt = torch.optim.Adam(renderer.parameters(), lr=cfg.lr)
            opt = renderer.get_optimizer()

            # logger.info(f"Iteration: {e}/{cfg.max_iteration} loss: {loss.item():.4f}")
            pbar.set_description(f"Iteration: {e}/{cfg.max_iteration}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if step_check(e, cfg.save_iteration):
                renderer.save(f"./saved/{timestamp}_{cfg.data_name}_sh/model_{e}.pt")

            if cfg.viewer:
                while viewer.pause_training:
                    viewer.update()
                    time.sleep(1.0 / viewer_eval_fps)
                if e % viewer.train_viewer_update_period_slider.value == 0:
                    viewer.update()


@lineprofiler
def train_only(cfg):
    logger = logging.getLogger(__name__)
    if cfg.viewer:
        logger.info("Viewer enabled")
    os.chdir(hydra.utils.get_original_cwd())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    if cfg.wandb:
        wandb.init(
            project="gs",
            config=cfg,
            name=f"{cfg.data_name}_{timestamp}",
            sync_tensorboard=True,
        )
    writer = SummaryWriter(
        f"./logs/runs/{cfg.data_name}/{timestamp}", comment=cfg.comment
    )
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

    c2ws, camera_info, images, pts, rgb, _ = get_c2ws_and_camera_info_v1(cfg)
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

    opt = renderer.get_optimizer()

    if cfg.get("viewer", False):
        from utils.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(cfg)
        viewer.set_renderer(renderer)
        viewer_eval_fps = 1000.0

    log_iteration = cfg.get("log_iteration", 50)
    only_forward = cfg.get("only_forward", False)

    writer.add_text("cfg", str(cfg))
    writer.add_text("comment", cfg.comment)

    with tqdm(total=cfg.max_iteration) as pbar:
        for e in range(cfg.max_iteration):
            i = e % N_images
            tic()
            out = renderer(c2ws[i], camera_info)
            toc("whole renderer forward")
            if e == 0:
                print_info(out, "out")
                print("num total gaussian", renderer.total_dub_gaussians)
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
            if step_check(e, log_iteration, True):
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

            if cfg.adapt_ctrl_enabled:
                renderer.adaptive_control(e)

            # opt = torch.optim.Adam(renderer.parameters(), lr=cfg.lr)
            del opt
            opt = renderer.get_optimizer()

            # logger.info(f"Iteration: {e}/{cfg.max_iteration} loss: {loss.item():.4f}")
            pbar.set_description(f"Iteration: {e}/{cfg.max_iteration}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if step_check(e, cfg.save_iteration):
                renderer.save(f"./saved/{timestamp}_{cfg.data_name}_sh/model_{e}.pt")

            if cfg.viewer:
                while viewer.pause_training:
                    viewer.update()
                    time.sleep(1.0 / viewer_eval_fps)
                if e % viewer.train_viewer_update_period_slider.value == 0:
                    viewer.update()

    if cfg.timing:
        print(prof.key_averages())


@hydra.main(config_path="conf", config_name="garden_sh")
def main(cfg):
    mode = cfg.get("mode", "train_only")
    if mode == "train_only":
        train_only(cfg)
    elif mode == "train_and_eval":
        train_and_eval(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
