import numpy as np
from astropy.io import fits
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torch
from src.data import centres, make_central_mask
from src.dataset import HeatmapDataset
from src.model import TinyCNN
from src.train import train_overfit

def split_centres(grid_centres, frac_train=0.8, frac_val=0.1, seed=42):
    n = len(grid_centres)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    # keep centres as plain python list of tuples
    train_centres = [grid_centres[i] for i in train_idx]
    val_centres   = [grid_centres[i] for i in val_idx]
    test_centres  = [grid_centres[i] for i in test_idx]
    return train_centres, val_centres, test_centres

# load FITS
with fits.open("data/mosaic_injected.fits") as hdul:
    data = hdul[0].data.astype(np.float32)

# load truth
truth = np.loadtxt("data/truth_injected.csv", skiprows=1, delimiter=',')
truth_x = truth[:, 1]
truth_y = truth[:, 2]

# crop region
X0, X1 = 139, 1000
Y0, Y1 = 1111, 2500
reg = data[Y0:Y1, X0:X1]

# truth -> region coords
truth_xy = np.column_stack([truth_x - X0, truth_y - Y0]).astype(np.float32)

# grid + mask
grid_centres = centres(0, reg.shape[1], 0, reg.shape[0], 32)
mask = make_central_mask(32)

# dataset + small subset
ds = HeatmapDataset(reg, grid_centres, truth_xy, mask, size=32, sigma_h=2.0)
small = Subset(ds, list(range(256)))  # overfit set
dl = DataLoader(small, batch_size=64, shuffle=True, num_workers=0)

train_centres, val_centres, test_centres = split_centres(grid_centres, 0.8, 0.1, seed=42)

ds_train = HeatmapDataset(reg, train_centres, truth_xy, mask, size=32, sigma_h=2.0)
ds_val   = HeatmapDataset(reg, val_centres, truth_xy, mask, size=32, sigma_h=2.0)
ds_test  = HeatmapDataset(reg, test_centres, truth_xy, mask, size=32, sigma_h=2.0)

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=0)
dl_val   = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=0)
dl_test  = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)

# model + train
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyCNN().to(device)

train_overfit(model, dl_train, device=device, lr=1e-3, epochs=500)

# ---- SAVE TRAINED MODEL ----
SAVE_PATH = "models/model.pt"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved trained model to {SAVE_PATH}")



def show_pred(model, ds, idx, device="cpu"):
    model.eval()
    with torch.no_grad():
        x, y, m = ds[idx]
        xb = x.unsqueeze(0).to(device)              # (1,1,32,32)
        logits = model(xb)
        pred = torch.sigmoid(logits)[0,0].cpu().numpy()

    x_np = x[0].numpy()
    y_np = y[0].numpy()
    m_np = m[0].numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    vmin, vmax = np.percentile(x_np, [5, 99])
    plt.imshow(x_np, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Input patch (norm)")

    plt.subplot(1,3,2)
    plt.imshow(y_np, origin="lower", cmap="gray")
    plt.title("Target heatmap")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(pred, origin="lower", cmap="gray")
    plt.title("Predicted heatmap")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


for j in range(len(ds)):
    x, y, m = ds[j]
    if y.max() > 0:
        idx = j
        break

