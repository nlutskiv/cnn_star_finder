import numpy as np

def make_heatmap(size, points_uv, sigma_h=2.0):
    """
    Docstring for make_heatmap
    Generate a gaussian on the truth inside inner 16x16 for training
    Only includes the injected objects
    
    :param size: Description
    :param points_uv: Description
    :param sigma_h: Description

    """
    yy, xx = np.mgrid[0:size, 0:size]          # grid of pixel coords
    H = np.zeros((size, size), dtype=np.float32)

    for (u0, v0) in points_uv:
        G = np.exp(-((xx - u0)**2 + (yy - v0)**2) / (2*sigma_h**2)).astype(np.float32)
        H = np.maximum(H, G)                   # combine peaks

    return H