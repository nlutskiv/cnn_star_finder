# CNN Star Finder

This project implements a convolutional neural network to detect point sources in astronomical FITS images.  
The model is trained using injection–recovery simulations, where synthetic sources are added to a real background image and used as supervised labels.

The network predicts a centre-likelihood heatmap rather than discrete detections, allowing flexible post-processing and robust handling of crowded fields.

## Method overview
- Synthetic point sources are injected into a real FITS image using a Gaussian PSF
- A truth catalogue is generated containing injected source positions
- The image is split into 32×32 patches
- A CNN is trained to predict a heatmap of source centres
- During inference, overlapping patches are scanned with stride 16
- Only the central region of each patch is stitched into a global heatmap to avoid edge artefacts
- Peak finding with non-maximum suppression is used to generate a final detection catalogue

## Training & Inference

### Training
Train the model and save the weights:
`python main.py`

This produces:
- `model.pt` – trained CNN weights

Training uses a **masked binary cross-entropy loss** to supervise only the **central region of each patch**.

### Inference
Run detection using a trained model:
`python infer.py`

This outputs:
- `detections_region.csv` – detected source positions in region coordinates
- Diagnostic plots showing the predicted heatmap and detections overlaid on the image

### Example Results
Injection–recovery test on **600 injected sources**:
- **Completeness (≤ 3 px):** ~0.7 to 0.9

- <img width="700" height="600" alt="example_output" src="https://github.com/user-attachments/assets/3009fd57-9010-4368-9be9-7b2784dea70e" />


Some false positives correspond to real pre-existing sources in the image or to saturated or extended objects and also object getting detected more than once due to stitching

### Repository Structure
- `src/` – core implementation (dataset, model, training, inference utilities)
- `data/` – FITS images and truth catalogues
- `models/` – trained model checkpoints
- `outputs/` – detection catalogues and figures

### Possible Improvements
- Stronger non-maximum suppression for extended or saturated sources
- Explicit masking of saturated regions
- Flux-dependent completeness analysis
- Training on multiple injected images for improved generalisation
- Mask out detected objects to avoid redetections

