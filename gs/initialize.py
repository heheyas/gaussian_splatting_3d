import numpy as np
import torch
import faiss


def cov_init(pts, k=3):
    ## set cov to mean distance of nearest k points
    if not isinstance(pts, torch.Tensor):
        pts = torch.from_numpy(pts).to("cuda")

    # pts = pts.to("cuda")
    # dist = torch.cdist(pts, pts)
    # topk = torch.topk(dist, k=k, dim=1, largest=False)

    # return topk.mean(axis=1)
    res = faiss.StandardGpuResources()
    # pts = pts.to("cuda")
    index = faiss.IndexFlatL2(pts.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(pts.cpu())
    D, _ = gpu_index_flat.search(pts, k + 1)

    return torch.from_numpy(D[..., 1:].mean(axis=1))
