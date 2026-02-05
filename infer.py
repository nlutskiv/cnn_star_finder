import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.io import fits

from src.data import get_patch, robust_normalize  # don't use centres()
from src.model import TinyCNN


# ---------------------------
# Peak finding (no repeats in a given heatmap)
# ---------------------------
def find_peaks_nms_disk(H, thr=0.4, r_excl=6, max_det=5000):
    """Iterative peak extraction with circular NMS (masks out detected region)."""
    Hwork = H.copy()
    peaks = []

    yy, xx = np.ogrid[:Hwork.shape[0], :Hwork.shape[1]]
    r2 = float(r_excl * r_excl)

    for _ in range(max_det):
        y, x = np.unravel_index(np.argmax(Hwork), Hwork.shape)
        score = float(Hwork[y, x])
        if score < thr:
            break
        peaks.append((int(x), int(y), score))
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r2
        Hwork[mask] = 0.0

    return peaks


# ---------------------------
# Dedupe across stitches
# ---------------------------
def dedup_peaks(peaks, r_merge=5.0):
    """
    Merge detections within r_merge pixels.
    Keeps the highest-score detection per cluster (greedy).
    """
    if not peaks:
        return []

    peaks_sorted = sorted(peaks, key=lambda t: t[2], reverse=True)
    kept = []
    r2 = r_merge * r_merge

    for x, y, s in peaks_sorted:
        ok = True
        for xk, yk, _ in kept:
            dx = x - xk
            dy = y - yk
            if dx * dx + dy * dy <= r2:
                ok = False
                break
        if ok:
            kept.append((x, y, s))

    return kept


# ---------------------------
# Build a sane list of centres (NO explosions)
# ---------------------------
def build_centres_for_offset(H, W, stride, size, xoff, yoff):
    """
    Returns centres (cx,cy) such that the patch [cx-half:cx+half] is fully inside the image.
    Avoids wasting time on invalid patches.
    """
    half = size // 2
    xs = range(half + xoff, W - half, stride)
    ys = range(half + yoff, H - half, stride)
    return [(cx, cy) for cy in ys for cx in xs]


def make_offsets(stride: int, n_offsets: int):
    """
    n_offsets=1 -> [(0,0)]
    n_offsets=4 with stride=16 -> offsets [0,4,8,12] -> 16 stitches
    """
    if n_offsets <= 1:
        return [(0, 0)]
    step = max(1, stride // n_offsets)
    offs = list(range(0, stride, step))[:n_offsets]
    return [(x, y) for y in offs for x in offs]


# ---------------------------
# Heatmap inference for one offset
# ---------------------------
def infer_global_heatmap_offset(model, reg, stride=16, size=32, xoff=0, yoff=0, device="cpu",
                               inner_frac=0.25, max_patches=None, progress_every=2000):
    """
    Sliding-window inference with offset grid and stitched inner region.

    inner_frac=0.25 -> inner = size*0.25 = 8 -> inner_size = 16 (your original)
    inner_frac=0.125 -> inner = 4 -> inner_size = 24 (trust more area; often improves recall)
    """
    model.eval()
    H, W = reg.shape
    H_global = np.zeros((H, W), dtype=np.float32)

    half = size // 2
    inner = int(round(size * inner_frac))
    inner = max(0, min(inner, half - 1))  # keep valid
    inner_size = size - 2 * inner

    centres_list = build_centres_for_offset(H, W, stride, size, xoff, yoff)

    if max_patches is not None:
        centres_list = centres_list[:max_patches]

    with torch.no_grad():
        for i, (cx, cy) in enumerate(centres_list):
            patch = reg[cy - half:cy + half, cx - half:cx + half]
            # patch will always be correct size due to centre construction

            x = robust_normalize(patch)[None, None].astype(np.float32)
            xb = torch.from_numpy(x).to(device)

            pred = torch.sigmoid(model(xb))[0, 0].cpu().numpy().astype(np.float32)

            pred_inner = pred[inner:inner + inner_size, inner:inner + inner_size]

            gx1 = cx - half + inner
            gx2 = gx1 + inner_size
            gy1 = cy - half + inner
            gy2 = gy1 + inner_size

            H_global[gy1:gy2, gx1:gx2] = np.maximum(H_global[gy1:gy2, gx1:gx2], pred_inner)

            if progress_every and (i + 1) % progress_every == 0:
                print(f"  processed {i+1}/{len(centres_list)} patches (offset {xoff},{yoff})")

    return H_global

import numpy as np
from scipy.ndimage import label

def peaks_one_per_component(H, thr=0.2, min_area=10):
    """
    Returns one peak per connected component above thr.
    Choose the (x,y) with max score within each component.
    """
    m = H >= thr
    lab, n = label(m)

    peaks = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if ys.size < min_area:
            continue
        scores = H[ys, xs]
        i = int(np.argmax(scores))
        y = int(ys[i]); x = int(xs[i])
        peaks.append((x, y, float(scores[i])))

    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks

def run_multi_stitch(model, reg, stride=16, size=32, n_offsets=4, device="cpu",
                     inner_frac=0.25, max_patches_per_stitch=None):
    offsets = make_offsets(stride, n_offsets)
    Hs = []
    for k, (xoff, yoff) in enumerate(offsets):
        print(f"Stitch pass {k+1}/{len(offsets)} with offset (xoff={xoff}, yoff={yoff})")
        Hk = infer_global_heatmap_offset(
            model, reg, stride=stride, size=size,
            xoff=xoff, yoff=yoff, device=device,
            inner_frac=inner_frac,
            max_patches=max_patches_per_stitch,
        )
        Hs.append(Hk)
    H_comb = np.maximum.reduce(Hs)
    return Hs, H_comb, offsets

if __name__ == "__main__":
    # ----- LOAD REGION -----
    with fits.open("data/mosaic.fits") as hdul:
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
    print("Loaded model.pt on", device)

    # ----- MULTI-STITCH INFERENCE -----
    stride = 16
    size = 32
    n_offsets = 4          # 4 -> 16 passes, 6 -> 36, 8 -> 64
    inner_frac = 0.125     # trust 24x24 instead of 16x16
    max_patches_per_stitch = None

    Hs, H_comb, offsets = run_multi_stitch(
        model, reg,
        stride=stride, size=size,
        n_offsets=n_offsets,
        device=device,
        inner_frac=inner_frac,
        max_patches_per_stitch=max_patches_per_stitch
    )
    print("Num stitch passes:", len(offsets))

    # ----- OPTIONAL: BRIGHT VETO MASK (kills the '20 peaks on one saturated star' issue) -----
    # Requires scipy.ndimage. If you don't have scipy, skip this block.
    try:
        from scipy.ndimage import binary_dilation

        def bright_veto_from_image(img, q=99.9, dilate_px=30):
            t = np.percentile(img[np.isfinite(img)], q)
            m = img >= t
            structure = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
            return binary_dilation(m, structure=structure)

        veto = bright_veto_from_image(reg, q=99.9, dilate_px=30)
        H_use = H_comb.copy()
        H_use[veto] = 0.0
        print("Applied bright-source veto mask.")
    except Exception:
        H_use = H_comb
        print("No scipy (or veto failed) -> skipping bright-source veto mask.")

    # ----- DEBUG: heatmap distribution -----
    vals = H_use[np.isfinite(H_use)].ravel()
    print("H_use percentiles:", np.percentile(vals, [90, 95, 98, 99, 99.5, 99.9, 99.95, 99.99]))

    # ----- PEAKS: ONE PER CONNECTED COMPONENT (prevents 20 peaks on one big blob) -----
    # Best method; falls back to NMS if scipy missing.
    try:
        from scipy.ndimage import label

        def peaks_one_per_component(H, thr=0.02, min_area=20):
            m = H >= thr
            lab, n = label(m)
            peaks = []
            for k in range(1, n + 1):
                ys, xs = np.where(lab == k)
                if ys.size < min_area:
                    continue
                scores = H[ys, xs]
                i = int(np.argmax(scores))
                y = int(ys[i]); x = int(xs[i])
                peaks.append((x, y, float(scores[i])))
            peaks.sort(key=lambda t: t[2], reverse=True)
            return peaks

        thr = 0.00077     # tune: 0.01–0.10 (use the percentiles above as guide)
        min_area = 4   # tune: 10–50 to kill tiny noisy blobs
        peaks = peaks_one_per_component(H_use, thr=thr, min_area=min_area)
        print(f"Connected-components peaks: {len(peaks)}")
    except Exception:
        # Fallback: your NMS (will still spam big blobs unless r_excl huge)
        thr = 0.01
        r_excl = 16
        peaks = find_peaks_nms_disk(H_use, thr=thr, r_excl=r_excl, max_det=20000)
        print(f"NMS fallback peaks: {len(peaks)}")

    # ----- DEDUP (mostly unnecessary after components, but safe if you later merge multiple lists) -----
    r_merge = 8.0
    peaks_merged = dedup_peaks(peaks, r_merge=r_merge)

    pred_xy = np.array([(x, y) for x, y, s in peaks_merged], dtype=np.float32)
    print("After dedup:", len(pred_xy))

    # ----- PLOTS -----
    plt.figure(figsize=(7, 6))
    plt.imshow(H_use, origin="lower", cmap="gray")
    plt.title("Heatmap used for detection (combined + veto)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    vmin, vmax = np.percentile(reg, [5, 99])
    plt.imshow(reg, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    if len(pred_xy):
        plt.scatter(pred_xy[:, 0], pred_xy[:, 1], s=35,
                    facecolors="none", edgecolors="lime", linewidths=1.2)
    plt.title("Detections (one per blob + optional veto)")
    plt.tight_layout()
    plt.show()

    # ----- SAVE -----
    np.savetxt("data/detections_region.csv", pred_xy, delimiter=",", header="x,y", comments="")
    print("Saved data/detections_region.csv")
