from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.annotation import load_annotation_json, annotation_to_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "kolektorsdd2-DatasetNinja"


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    image: BGR image from OpenCV
    mask: HxW binary mask
    """
    overlay = image.copy()

    # Mark defect region in red
    overlay[mask == 1] = [0, 0, 255]

    blended = cv2.addWeighted(image, 0.75, overlay, 0.25, 0)
    return blended


def main():
    split = "train"
    file_stem = "10021.png"   # 有缺陷的文件名

    image_path = DATASET_ROOT / split / "img" / file_stem
    annotation_path = DATASET_ROOT / split / "ann" / f"{file_stem}.json"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotation = load_annotation_json(annotation_path)
    mask = annotation_to_mask(annotation)
    overlay = make_overlay(image, mask)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask positive pixels: {mask.sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()