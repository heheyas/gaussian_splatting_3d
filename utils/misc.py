import cv2
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

_timing_ = False

logger = logging.getLogger(__name__)


def tic():
    import time

    if "_timing_" not in globals() or not _timing_:
        return

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc(name=""):
    import time

    if "_timing_" not in globals() or not _timing_:
        return

    if "startTime_for_tictoc" in globals():
        print("$" * 30)
        print(
            f"{name} Elapsed time is "
            + str(time.time() - startTime_for_tictoc)
            + " seconds."
        )
        print("$" * 30)
        # logger.info(
        #     f"{name} Elapsed time is "
        #     + str(time.time() - startTime_for_tictoc)
        #     + " seconds."
        # )
    else:
        print("Toc: start time not set")


class Timer:
    def __init__(self):
        self.t = 0.0
        self.n = 0

    def tic(self):
        self.t = torch.cuda.Event(enable_timing=True)
        self.t.record()

    def toc(self):
        torch.cuda.synchronize()
        self.t = self.t.elapsed_time()
        self.n += 1

    def avg(self):
        return self.t / self.n


def save_fig(img, filename):
    tmp = Path("tmp")
    if not tmp.exists():
        tmp.mkdir()
    if not isinstance(img, np.ndarray):
        assert isinstance(img, torch.Tensor)
        if img.size(0) == 1:
            img = img.squeeze()
        if img.size(0) == 3:
            img = img.moveaxs(0, -1)

        img = img.cpu().numpy()
    else:
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.savefig(f"tmp/{filename}.png", bbox_inches="tight")


def save_img(img, path, filename):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    img = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path / filename), img)


def float2uint8(img):
    if isinstance(img, torch.Tensor):
        img = (img * 255.0).to(torch.uint8)
    elif isinstance(img, np.ndarray):
        img = (img * 255.0).astype(np.uint8)
    else:
        raise NotImplementedError

    return img


def print_info(var, name):
    print(f"++++++{name}++++++")
    print("contiguous: ", var.is_contiguous())
    print(var.max())
    print(var.min())
    print(var.shape)
    if var.dtype == torch.float32:
        print("mean:", var.mean().item())
    print("nonzero:", torch.count_nonzero(var).item())
    if var.isnan().any():
        print("num nans", torch.count_nonzero(var.isnan()).item())
    print(f"++++++{name}++++++")


def mp_read(filenames):
    pass


def lineprofiler(func):
    try:
        func = profile(func)
    except NameError:
        pass
    return func
