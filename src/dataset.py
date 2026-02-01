import numpy as np
import torch
from torch.utils.data import Dataset

from .data import sample_item  # uses your existing function

class HeatmapDataset(Dataset):
    def __init__(self, reg, grid_centres, truth_xy, mask, size=32, sigma_h=2.0):
        self.reg = reg
        self.grid_centres = grid_centres
        self.truth_xy = truth_xy
        self.mask = mask
        self.size = size
        self.sigma_h = sigma_h

    def __len__(self):
        return len(self.grid_centres)

    def __getitem__(self, i):
        x, y, m = sample_item(
            i,
            self.reg,
            self.grid_centres,
            self.truth_xy,
            self.mask,
            size=self.size,
            sigma_h=self.sigma_h
        )
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
