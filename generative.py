import gc
import os
import sys
import tqdm
import time
import torch
import wandb
import hydra
import datetime
import numpy as np
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from gs.renderer import Renderer
from gs.sh_renderer import SHRenderer
from generative.diffusion.stable_diffusion import StableDiffusion
from generative.diffusion.deepfloyd import DeepFloyd
from generative.diffusion.positional_text_embeddings import PositionalTextEmbeddings
from generative.provider import CameraPoseProvider
from utils import misc
from utils.misc import tic, toc, lineprofiler, step_check, reduce_dicts
from utils.viewer.viser_viewer import ViserViewer

from rich.console import Console

console = Console()


class Trainer(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.timestamp = timestamp
        if cfg.wandb:
            wandb.init(
                project="gs",
                config=cfg,
                name=f"gen_{timestamp}",
                sync_tensorboard=True,
            )
        self.writer = SummaryWriter(f"./logs/runs/gen/{timestamp}", comment=cfg.comment)
        if cfg.timing:
            misc._timing_ = True
            console.print(f"[red bold]timimg enabled")
            tic()
            toc("test")

        self.device = cfg.device
        self.set_prompt(cfg)
        self.writer.add_text("prompt", self.prompt, 0)
        self.set_diffusion_model_and_text_embeddings(cfg)
        self.set_hparams(cfg)
        from_ckpt = cfg.get("from_ckpt", False)
        if not from_ckpt:
            self.init_renderer(cfg)
        else:
            self.load_renderer(cfg)
        self.provider = CameraPoseProvider(cfg)

        self.camera_pos = []
        self.set_viewer(cfg)

    def set_viewer(self, cfg):
        if cfg.viewer == False:
            self.viewer_enabled = False
            return

        self.viewer_enabled = True
        self.viewer = ViserViewer(cfg)
        self.viewer.set_renderer(self.renderer)
        self.viewer_eval_fps = 1000.0

    def set_prompt(self, cfg):
        self.prompt = cfg.prompt
        self.cur_prompt = cfg.prompt
        self.top_prompt = cfg.get("top_prompt", ", overhead view")
        self.side_prompt = cfg.get("side_prompt", ", side view")
        self.front_prompt = cfg.get("front_prompt", ", front view")
        self.back_prompt = cfg.get("back_prompt", ", back view")
        self.positional_prompting = cfg.get("positional_prompting", "discrete")

        console.print(f"Prompt: {self.prompt}", style="magenta")

    def set_diffusion_model_and_text_embeddings(self, cfg):
        if cfg.diffusion_model == "stablediffusion":
            self._diffusion_model = StableDiffusion(cfg.device, version=cfg.sd_version)
        elif cfg.diffusion_model == "deepfloyd":
            self._diffusion_model = DeepFloyd(cfg.device)
        else:
            raise NotImplementedError(
                f"Diffusion model {cfg.diffusion_model} not implemented"
            )

        self.text_embeddings = PositionalTextEmbeddings(
            base_prompt=self.cur_prompt,
            top_prompt=self.cur_prompt + self.top_prompt,
            side_prompt=self.cur_prompt + self.side_prompt,
            back_prompt=self.cur_prompt + self.back_prompt,
            front_prompt=self.cur_prompt + self.front_prompt,
            diffusion_model=self._diffusion_model,
            positional_prompting=self.positional_prompting,
        )

    def set_hparams(self, cfg):
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.save_iteration = cfg.save_iteration
        self.log_iteration = cfg.log_iteration
        self.guidance_scale = cfg.get("guidance_scale", 100.0)
        self.alpha_penalty = cfg.get("alpha_penalty", 0.0)
        self.split_cfg = cfg.split

    def init_renderer(self, cfg):
        ## first init pts and rgb
        ## end initialize a renderer

        if cfg.init_method == "random":
            num_init_pts = int(cfg.num_init_pts)
            init_pts_bounds = cfg.init_pts_bounds
            pts = torch.randn((num_init_pts, 3), device=cfg.device) * cfg.init_pts_std
            if cfg.debug:
                torch.save(pts, "./tmp/debug/pts.pt")
            pts = pts.clamp(min=-init_pts_bounds, max=init_pts_bounds)
            rgb = torch.rand((num_init_pts, 3), device=cfg.device).clamp(0.0001, 0.9999)
        elif cfg.init_method == "point_e":
            console.print(f"[red bold]Using point_e init")
            pts, rgb = self.point_e_init(cfg)
            cfg.svec_init_method = "nearest"
            cfg.alpha_init_method = "fixed"
            console.print(f"[red bold]Using nearest init for svec")
        else:
            raise NotImplementedError(f"Unknown init method {cfg.init_method}")
        self.renderer = SHRenderer(cfg, pts, rgb).to(self.device)

        ## TODO: add camera info
        self.camera_info = None
        self.camera_pose_provider = None

        self.random_bg = cfg.get("random_bg", True)
        self.opt = self.renderer.get_optimizer(0)

        self.grad_clip = any(v for v in self.cfg.grad_clip.values())
        if self.grad_clip:
            console.print("[green bold]Using grad clip")
            for field in self.cfg.grad_clip.keys():
                console.print(
                    f"[green bold]{field}: {self.cfg.grad_clip[field]}", end=" "
                )
            console.print("")

    def point_e_init(self, cfg):
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.diffusion.sampler import PointCloudSampler
        from point_e.models.download import load_checkpoint
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        from point_e.util.plotting import plot_point_cloud

        device = self.device
        num_init_pts = int(cfg.num_init_pts)

        base_name = "base40M-textvec"
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

        base_model.load_state_dict(load_checkpoint(base_name, device))
        upsampler_model.load_state_dict(load_checkpoint("upsample", device))

        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024, num_init_pts - 4096],
            aux_channels=["R", "G", "B"],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=(
                "texts",
                "",
            ),  # Do not condition the upsampler at all
        )

        point_e_cfg = cfg.point_e
        if point_e_cfg.prompt == "":
            point_e_prompt = self.prompt
        else:
            point_e_prompt = point_e_cfg.prompt

        samples = None
        for x in tqdm.tqdm(
            sampler.sample_batch_progressive(
                batch_size=1, model_kwargs=dict(texts=[point_e_prompt])
            )
        ):
            samples = x
        pc = sampler.output_to_point_clouds(samples)[0]
        pcd_xyz = torch.from_numpy(pc.coords * point_e_cfg.scale).contiguous()
        pcd_rgb = torch.from_numpy(
            np.stack([pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=1)
        ).contiguous()

        return pcd_xyz, pcd_rgb

    @lineprofiler
    def loss_fn(self, output, text_embedding):
        loss = dict()
        if self.prompt != self.cur_prompt:
            self.cur_prompt = self.prompt
            self.text_embeddings.update_prompt(
                base_prompt=self.cur_prompt,
                top_prompt=self.cur_prompt + self.top_prompt,
                side_prompt=self.cur_prompt + self.side_prompt,
                back_prompt=self.cur_prompt + self.back_prompt,
                front_prompt=self.cur_prompt + self.front_prompt,
            )

        sds_loss = self._diffusion_model.sds_loss(
            text_embedding.to(self.device),
            output.to(self.device),
            guidance_scale=int(self.guidance_scale),
            grad_scaler=None,
        )
        loss["sds_loss"] = sds_loss.to(self.device)
        loss["alpha_loss"] = self.renderer.alpha_penalty(
            self.cfg.loss.alpha_loss_type
        ).to(self.device)

        return loss

    @lineprofiler
    def train_step(self, e):
        if self.random_bg:
            self.renderer.set_bg_rgb(torch.rand(3).to(self.device))
        loss_dicts = []
        for _ in range(self.batch_size):
            (
                camera_info,
                c2w,
                vertical_angle,
                horizontal_angle,
            ) = self.provider.sample_one(e)
            c2w = c2w.to(self.device)
            out = self.renderer(c2w, camera_info).moveaxis(-1, 0).unsqueeze(0)
            if self.prompt != self.cur_prompt:
                self.cur_prompt = self.prompt
                self.text_embeddings.update_prompt(
                    base_prompt=self.cur_prompt,
                    top_prompt=self.cur_prompt + self.top_prompt,
                    side_prompt=self.cur_prompt + self.side_prompt,
                    back_prompt=self.cur_prompt + self.back_prompt,
                    front_prompt=self.cur_prompt + self.front_prompt,
                )

            text_embedding = self.text_embeddings.get_text_embedding(
                vertical_angle=vertical_angle,
                horizontal_angle=horizontal_angle,
            )

            loss_dict = self.loss_fn(out, text_embedding)
            loss_dicts.append(loss_dict)

        loss_dict = reduce_dicts(loss_dicts, torch.mean)
        loss = 0.0
        for loss_name, loss_val in loss_dict.items():
            loss += loss_val * self.cfg.loss.get(f"{loss_name}_mult", 1.0)

        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip:
            self.renderer.clip_grad()
        self.opt.step()

        if step_check(e, self.save_iteration):
            self.renderer.save(f"./saved/{self.timestamp}_gen/model_{e}.pt")

        if self.cfg.adapt_ctrl_enabled:
            self.renderer.adaptive_control(e)

        if step_check(e, self.log_iteration, True):
            self.renderer.log(self.writer, e)
            self.writer.add_image("out", out.cpu()[0].clamp(min=0.0, max=1.0), e)
            for loss_name, loss_val in loss_dict.items():
                self.writer.add_scalar(f"gen_loss/{loss_name}", loss_val.item(), e)

        if e > self.split_cfg.warm_up and e < self.split_cfg.end:
            if step_check(e, self.split_cfg.period) and self.split_cfg.enabled:
                self.renderer.force_split()
                console.print(
                    f"[red bold]Forcing split, now {self.renderer.N} gaussians"
                )

        self.opt = self.renderer.get_optimizer(e)

        return loss

    def train(self):
        cfg = self.cfg
        # self.writer.add_hparams(dict(cfg), dict())
        self.writer.add_text("cfg", OmegaConf.to_yaml(cfg))
        start = 0
        if cfg.from_ckpt:
            start = cfg.start
        with tqdm.tqdm(total=cfg.max_iteration) as pbar:
            for e in range(start, cfg.max_iteration):
                loss = self.train_step(e)

                pbar.set_description(f"Iteration: {e}/{cfg.max_iteration}")
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

                if self.viewer_enabled:
                    while self.viewer.pause_training:
                        self.viewer.update()
                        time.sleep(0.01)
                    if e % self.viewer.train_viewer_update_period_slider.value == 0:
                        self.viewer.update()


@hydra.main(config_path="conf", config_name="generative", version_base="1.1")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
