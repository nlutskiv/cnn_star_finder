# CNN Star Finder

This project implements a small convolutional neural network to detect point sources (stars) in astronomical FITS images.

I built this after getting frustrated with the purely mechanical star-finding method proposed in my university lab. Instead of hard-coding what a star *should* look like, I let a CNN learn it directly from the data.

The model predicts a **centre-likelihood heatmap** rather than discrete detections, which makes post-processing flexible and much more robust in crowded or noisy regions.

A screenshot in this repository shows detections overlaid on a real image region (≈580 stars), obtained by tuning a single key hyperparameter: the heatmap threshold.

---

## Method overview

**Training data (injection–recovery)**

* Synthetic point sources are injected into a real FITS background using a Gaussian PSF
* A truth catalogue is generated from the injected source positions
* Targets are **Gaussian heatmaps** centred on each injected source

**Model & prediction**

* The image is split into 32×32 patches
* A lightweight CNN is trained to predict a 32×32 centre-likelihood heatmap
* The network learns *where* a source centre is likely to be, not just whether a pixel is bright

**Inference on large images**

* Overlapping patches are scanned across the image (stride 16)
* Only the central region of each patch is used to avoid edge artefacts
* Predictions are stitched into a global heatmap
* Inference is repeated with multiple offset grids to reduce stitching bias

**Post-processing**

* A threshold is applied to the global heatmap
* Connected components are extracted
* One detection is produced per connected region

The **heatmap threshold** is the most important hyperparameter in the entire pipeline and directly controls the precision–recall trade-off.

---

## Training & inference

### Training

Train the model and save the weights:

```
python main.py
```

This produces:

* `model.pt` – trained CNN weights

Training uses a **masked binary cross-entropy loss**, supervising only the **central region of each patch** to prevent edge artefacts from dominating the loss.

---

### Inference

Run detection using a trained model:

```
python infer.py
```

This outputs:

* `detections_region.csv` – detected source positions (region coordinates)
* Diagnostic plots showing the stitched heatmap and final detections

Threshold selection is done by inspecting the **distribution of heatmap values**, rather than using a fixed heuristic.

---

## Example results

Injection–recovery test on **~600 injected sources**:

* **Completeness (≤ 3 px):** ~0.95 

When run on a real, uninjected image region (just run infer.py for that), the model detects ≈580 stars with visually good localisation.


<img width="416" height="666" alt="Full image" src="https://github.com/user-attachments/assets/82692bfd-c5c1-436c-baad-f78686cde685" />


<img width="820" height="655" alt="zoomed" src="https://github.com/user-attachments/assets/33d04ce4-74b0-4d99-bed2-8facda7e474c" />

---

## Repository structure

* `src/` – core implementation (dataset, model, training, inference utilities)
* `data/` – FITS images and truth catalogues
* `models/` – trained model checkpoints
* `outputs/` – detection catalogues and figures

---

## Possible improvements

* Better handling of saturated and extended sources
* Explicit masking of very bright regions
* Training on multiple injected backgrounds for improved generalisation
* Fully convolutional architectures (e.g. U-Net) to remove the need for stitching

---


