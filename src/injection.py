import numpy as np
from astropy.io import fits
import csv

def add_gaussian_psf(img, x0, y0, flux, sigma):
    r = int(np.ceil(5 * sigma))
    x0i = int(np.floor(x0))
    y0i = int(np.floor(y0))

    x1, x2 = max(0, x0i - r), min(img.shape[1], x0i + r + 1)
    y1, y2 = max(0, y0i - r), min(img.shape[0], y0i + r + 1)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dx = xx - x0
    dy = yy - y0

    norm = 1.0 / (2.0 * np.pi * sigma**2)
    img[y1:y2, x1:x2] += flux * norm * np.exp(-(dx*dx + dy*dy) / (2*sigma*sigma))


def inject_sources(
    data,
    x_min, x_max, y_min, y_max,
    N,
    flux_min, flux_max,
    sigma,
    seed=42
):
    """
    sigma can be:
      - float → fixed size
      - (sigma_min, sigma_max) → uniform random per source
    """
    rng = np.random.default_rng(seed)
    injected = data.astype(float).copy()
    truth = []

    for i in range(N):
        if isinstance(sigma, (tuple, list)):
            sig = rng.uniform(sigma[0], sigma[1])
        else:
            sig = sigma

        margin = int(np.ceil(5 * sig))

        x0 = rng.uniform(x_min + margin, x_max - margin)
        y0 = rng.uniform(y_min + margin, y_max - margin)
        flux = rng.uniform(flux_min, flux_max)

        add_gaussian_psf(injected, x0, y0, flux, sig)
        truth.append((i, x0, y0, flux))

    return injected, truth

with fits.open("src/mosaic.fits") as hdul:
    data = hdul[0].data
    header = hdul[0].header

inj, truth = inject_sources(
    data,
    x_min=139, x_max=1000,
    y_min=1111, y_max=2500,
    N=600,
    flux_min=50000,
    flux_max=100000,
    sigma=(2, 5),  # or just 1.6
    seed=42
)

fits.PrimaryHDU(inj, header=header).writeto("mosaic_injected.fits", overwrite=True)

with open("truth_injected.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "x", "y", "flux"])
    for row in truth:
        w.writerow(row)

