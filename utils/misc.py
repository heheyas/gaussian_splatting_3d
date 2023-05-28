import numpy as np
import torch


def tic():
    import time

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time

    if "startTime_for_tictoc" in globals():
        print("$" * 30)
        print(
            "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
        )
        print("$" * 30)
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


def print_info(var, name):
    print(f"++++++{name}++++++")
    print("contiguous: ", var.is_contiguous())
    print(var.max())
    print(var.min())
    print(var.shape)
    print(f"++++++{name}++++++")
