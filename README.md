# Industrial Defect Inspection

Full-stack computer vision platform for industrial defect inspection using PyTorch, OpenCV, FastAPI, React, ONNX Runtime, SQLite, and Docker, with end-to-end training, inference, evaluation, visualization, and deployment.

## Current Status

Data preparation and PyTorch dataset pipeline are in progress.

## Dataset

This project uses the **Kolektor Surface-Defect Dataset 2 (KolektorSDD2)** for industrial surface-defect segmentation.

The dataset contains an official training split and test split. Raw dataset files are excluded from version control.

## Data Split Strategy

The official test split is treated as a **held-out final evaluation set**.

It is not used for:

- model selection
- hyperparameter tuning
- threshold tuning
- preprocessing decisions

Only the official training split is used during development.

The official training set is divided into:

- **90% training**
- **10% validation**

using a reproducible stratified split:

- `random_state=42`
- stratification based on whether an image contains a defect

This preserves approximately the same defect/normal ratio in the training and validation sets.

Current split:

| Split | Total | Normal | Defect |
|---|---:|---:|---:|
| Training | 2098 | 1877 | 221 |
| Validation | 234 | 209 | 25 |

The generated split files are stored in:

```text
data/processed/train.csv
data/processed/val.csv
```

## Image Preprocessing

KolektorSDD2 images have variable spatial dimensions.

Observed training-image dimensions:

- Height: `602–660 px`
- Width: `184–241 px`

Instead of directly resizing the images, the pipeline preserves the original image geometry and applies **symmetric zero-padding** to a fixed input size of:

```text
672 × 256
```

The same padding is applied to both the image and its segmentation mask.

This approach:

- avoids geometric distortion of small and thin defects
- preserves the original defect shape and scale
- avoids cropping image content
- produces fixed-size tensors that can be batched by PyTorch
- uses dimensions compatible with common CNN downsampling operations

For an image of size `643 × 229`, for example:

```text
Height padding:
672 - 643 = 29
top = 14
bottom = 15

Width padding:
256 - 229 = 27
left = 13
right = 14
```

The resulting image and mask both have dimensions:

```text
672 × 256
```

## Annotation Processing

KolektorSDD2 annotations are stored in Supervisely bitmap format.

The pipeline:

```text
Supervisely JSON
        ↓
Base64 decoding
        ↓
zlib decompression
        ↓
bitmap decoding
        ↓
binary defect mask
        ↓
symmetric padding
        ↓
PyTorch tensor
```

Binary segmentation masks use:

```text
0 = background
1 = defect
```