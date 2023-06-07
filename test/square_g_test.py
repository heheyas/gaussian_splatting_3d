import torch
import numpy as np
import matplotlib.pyplot as plt

def dist_point_lineseg(x, y, x1, x2,
                                    y1, y2):
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1

    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy

    return np.sqrt(dx * dx + dy * dy)

def dist(topleft, tile_x, tile_y, mean):
    rela = mean - topleft
    print(rela)
    # inside = rela[0] > 0 and rela[0] < tile_x and rela[1] > 0 and rela[1] < tile_y
    # if inside:
    #     return -100.

    # nearest = np.array([0, 0])

    # if rela[0] * (tile_x - rela[0]) < 0:
    #     nearest[0] = rela[0]
    # else:
    #     nearest[0] = tile_x if rela[0] > 0 else 0

    # if rela[1] * (tile_y - rela[1]) < 0:
    #     nearest[1] = rela[1]
    # else:
    #     nearest[1] = tile_y if rela[1] > 0 else 0

    # print(nearest)
    # print(rela)

    d1 = dist_point_lineseg(rela[0], rela[1], 0, tile_x, 0, 0)
    d2 = dist_point_lineseg(rela[0], rela[1], 0, tile_x, tile_y, tile_y)
    d3 = dist_point_lineseg(rela[0], rela[1], 0, 0, 0, tile_y)
    d4 = dist_point_lineseg(rela[0], rela[1], tile_x, tile_x, 0, tile_y)

    print(d1, d2, d3, d4)

    d = min(min(d1, d2), min(d3, d4))

    nearest = rela - np.array([tile_x, tile_y])
    print(np.linalg.norm(nearest))
    
    return d


topleft = np.array([-1, -1]) + 1
tile_x, tile_y = 2.0, 2.0

mean = np.array([2, 2]) + 1

print(dist(topleft, tile_x, tile_y, mean))