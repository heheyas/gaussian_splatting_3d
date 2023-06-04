import torch
from kornia.losses.ssim import SSIMLoss, ssim_loss


def get_loss_fn(cfg):
    base_loss_fn = None
    if cfg.loss_fn == "l2":
        base_loss_fn = torch.nn.functional.mse_loss
    elif cfg.loss_fn == "l1":
        base_loss_fn = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError

    def loss_fn(out, gt):
        loss = cfg.ssim_loss_mult * ssim_loss(
            out.moveaxis(-1, 0).unsqueeze(0),
            gt.moveaxis(-1, 0).unsqueeze(0),
            cfg.ssim_loss_win_size,
            reduction="mean",
        ) + (1 - cfg.ssim_loss_mult) * base_loss_fn(out, gt)

        return loss

    return loss_fn
