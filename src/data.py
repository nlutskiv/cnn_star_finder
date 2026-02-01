from astropy.io import fits
import numpy as np
import csv
from matplotlib import pyplot as plt
from .heatmap import make_heatmap


def make_central_mask(size=32):
    m = np.zeros((size, size), dtype=np.float32)
    inner = size // 4          # 8 for size=32
    m[inner:size-inner, inner:size-inner] = 1.0
    return m

def centres(x1,x2,y1,y2, size):
    half = size // 2
    xs = range(x1 + half, x2 - half + 1, size)
    ys = range(y1 + half, y2 - half + 1, size)
    return [(int(xc), int(yc)) for yc in ys for xc in xs]


def get_patch(img, cx, cy, size=32):
    half = size // 2
    return img[cy-half:cy+half, cx-half:cx+half]

def get_inner_inject(cx, cy, true_xy, size=32):
    half = size // 2          # 16
    inner = size // 4         # 8   

    x1 = cx - half
    y1 = cy - half

    xt = true_xy[:, 0]
    yt = true_xy[:, 1]

    # inside full patch
    inside = (xt >= x1) & (xt < x1 + size) & (yt >= y1) & (yt < y1 + size)
    if not np.any(inside):
        return np.empty((0, 2), dtype=np.float32)

    u = xt[inside] - x1
    v = yt[inside] - y1

    # inside central 16x16
    inner_mask = (u >= inner) & (u < size - inner) & (v >= inner) & (v < size - inner)
    u = u[inner_mask]
    v = v[inner_mask]

    return np.column_stack([u, v]).astype(np.float32)



def robust_normalize(patch):
    """
    Docstring for robust_normalize
    Huge range -> has to be normalized for proper training
    
    :param patch: Description
    """
    patch = patch.astype(np.float32)
    med = np.nanmedian(patch)
    mad = np.nanmedian(np.abs(patch - med)) + 1e-6
    patch = (patch - med) / (1.4826 * mad + 1e-6)
    return patch



def sample_item(i, reg, grid_centres, truth_xy, mask, size=32, sigma_h=2.0):
    """
    Docstring for sample_item
    Wraps the dataset creation functionality
    
    :param i: Description
    :param reg: Description
    :param grid_centres: Description
    :param truth_xy: Description
    :param mask: Description
    :param size: Description
    :param sigma_h: Description
    """
    cx, cy = grid_centres[i]
    cx, cy = int(cx), int(cy)        # safety

    patch = get_patch(reg, cx, cy, size=size)
    assert patch.shape == (size, size), patch.shape

    pts = get_inner_inject(cx, cy, true_xy=truth_xy, size=size)
    H = make_heatmap(size, pts, sigma_h=sigma_h)

    x = robust_normalize(patch)[None, :, :]      # (1,32,32)
    y = H[None, :, :]                            # (1,32,32)
    m = mask[None, :, :]                         # (1,32,32)

    return x.astype(np.float32), y.astype(np.float32), m.astype(np.float32)











