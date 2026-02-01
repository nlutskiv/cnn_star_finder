import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.io import fits

from src.data import centres, get_patch, robust_normalize
from src.model import TinyCNN


def find_peaks_nms(H, thr=0.5, min_dist=3):
    Hwork = H.copy()
    peaks = []

    while True:
        y, x = np.unravel_index(np.argmax(Hwork), Hwork.shape)
        score = float(Hwork[y, x])
        if score < thr:
            break

        peaks.append((int(x), int(y), score))

        y1 = max(0, y - min_dist)
        y2 = min(Hwork.shape[0], y + min_dist + 1)
        x1 = max(0, x - min_dist)
        x2 = min(Hwork.shape[1], x + min_dist + 1)
        Hwork[y1:y2, x1:x2] = 0.0

    return peaks


def infer_global_heatmap(model, reg, stride=16, size=32, device="cpu"):
    model.eval()
    H_global = np.zeros_like(reg, dtype=np.float32)

    half = size // 2          # 16
    inner = size // 4         # 8
    inner_size = size - 2 * inner  # 16

    x_max = reg.shape[1]
    y_max = reg.shape[0]

    # IMPORTANT: step=stride here (overlap)
    inf_centres = centres(0, x_max, 0, y_max, stride)

    with torch.no_grad():
        for cx, cy in inf_centres:
            patch = get_patch(reg, cx, cy, size=size)
            if patch.shape != (size, size):
                continue

            x = robust_normalize(patch)[None, None, :, :].astype(np.float32)  # (1,1,32,32)
            xb = torch.from_numpy(x).to(device)

            pred = torch.sigmoid(model(xb))[0, 0].cpu().numpy().astype(np.float32)  # (32,32)

            # only trust central 16x16
            pred_inner = pred[inner:inner+inner_size, inner:inner+inner_size]

            gx1 = cx - half + inner
            gx2 = gx1 + inner_size
            gy1 = cy - half + inner
            gy2 = gy1 + inner_size

            H_global[gy1:gy2, gx1:gx2] = np.maximum(H_global[gy1:gy2, gx1:gx2], pred_inner)

    return H_global


if __name__ == "__main__":
    # ----- LOAD REGION (same region as training) -----
    with fits.open("data/mosaic_injected.fits") as hdul:
        data = hdul[0].data.astype(np.float32)

    X0, X1 = 139, 1000
    Y0, Y1 = 1111, 2500
    reg = data[Y0:Y1, X0:X1]

    # ----- LOAD MODEL -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)

    state = torch.load("models/model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded model.pt")

    # ----- INFERENCE -----
    H_global = infer_global_heatmap(model, reg, stride=16, size=32, device=device)

    # ----- PEAKS -----
    peaks = find_peaks_nms(H_global, thr=0.4, min_dist=2)
    pred_xy = np.array([(x, y) for x, y, s in peaks], dtype=np.float32)

    print("Num detections:", len(pred_xy))

    # ----- PLOTS -----
    plt.figure(figsize=(7,6))
    plt.imshow(H_global, origin="lower", cmap="gray")
    plt.title("Global predicted heatmap (stitched inner 16x16)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,6))
    vmin, vmax = np.percentile(reg, [5, 99])
    plt.imshow(reg, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    if len(pred_xy):
        plt.scatter(pred_xy[:, 0], pred_xy[:, 1], s=50,
                    facecolors="none", edgecolors="lime", linewidths=1.5)

    plt.title("Detections on region")
    plt.tight_layout()
    plt.show()

    # ----- SAVE DETECTIONS -----
    np.savetxt("data/detections_region.csv", pred_xy, delimiter=",", header="x,y", comments="")
    print("Saved detections_region.csv")



# ---------- LOAD TRUTH (same as training) ----------
truth = np.loadtxt("data/truth_injected.csv", skiprows=1, delimiter=",")
truth_x = truth[:, 1]
truth_y = truth[:, 2]

# convert to region coords (same crop!)
truth_xy = np.column_stack([truth_x - X0, truth_y - Y0]).astype(np.float32)


# ---------- MATCHING ----------
def match_preds_to_truth(pred_xy, truth_xy, r=3.0):
    """
    Greedy nearest-neighbour matching
    pred_xy: (P,2)
    truth_xy: (T,2)
    """
    if len(pred_xy) == 0:
        return np.zeros(len(truth_xy), dtype=bool), np.array([], dtype=bool)

    pred_xy = np.asarray(pred_xy, dtype=float)
    truth_xy = np.asarray(truth_xy, dtype=float)

    used_pred = np.zeros(len(pred_xy), dtype=bool)
    matched_truth = np.zeros(len(truth_xy), dtype=bool)

    for ti in range(len(truth_xy)):
        dx = pred_xy[:, 0] - truth_xy[ti, 0]
        dy = pred_xy[:, 1] - truth_xy[ti, 1]
        d = np.sqrt(dx*dx + dy*dy)

        j = np.argmin(d)
        if d[j] <= r and not used_pred[j]:
            matched_truth[ti] = True
            used_pred[j] = True

    return matched_truth, used_pred


matched_truth, matched_pred = match_preds_to_truth(pred_xy, truth_xy, r=3.0)

n_truth = len(truth_xy)
n_pred = len(pred_xy)
n_rec = matched_truth.sum()
n_fp = (~matched_pred).sum()

print("\n--- RETRIEVAL METRICS ---")
print(f"Injected sources : {n_truth}")
print(f"Predicted sources: {n_pred}")
print(f"Recovered (≤3 px): {n_rec}")
print(f"Completeness     : {n_rec / n_truth:.3f}")
print(f"False positives  : {n_fp}")
