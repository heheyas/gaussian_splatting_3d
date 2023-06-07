import numpy as np
import torch
import kornia
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Metrics:
    def __init__(self) -> None:
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        
    def evaluate(self, pred, gt):
        pred = pred.clamp(min=0.0, max=1.0)
        gt = gt.clamp(min=0.0, max=1.0)
        psnr = self.psnr(pred, gt)
        ssim = self.ssim(pred, gt, data_range=1.0)
        lpips = self.lpips(pred, gt)
        
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
        }
        
        return metrics