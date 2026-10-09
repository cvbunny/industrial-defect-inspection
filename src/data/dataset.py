from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.annotation import load_annotation_json, annotation_to_mask
from src.data.transforms import pad_to_size


class KolektorSDD2Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split_csv: str | Path,
    ):
        self.dataset_root = Path(dataset_root)
        self.split_csv = Path(split_csv)

        self.image_dir = self.dataset_root / "train" / "img"
        self.annotation_dir = self.dataset_root / "train" / "ann"

        self.samples = pd.read_csv(self.split_csv)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        filename = row["filename"]

        image_path = self.image_dir / filename
        annotation_path = self.annotation_dir / f"{filename}.json"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = load_annotation_json(annotation_path)
        mask = annotation_to_mask(annotation)

        image, mask = pad_to_size(
        image,
        mask,
        target_height=672,
        target_width=256,
        )

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
    