from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kolektorsdd2-DatasetNinja"
)

IMAGE_DIR = DATASET_ROOT / "train" / "img"
ANNOTATION_DIR = DATASET_ROOT / "train" / "ann"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def has_defect(annotation_path: Path) -> int:
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    return int(len(annotation.get("objects", [])) > 0)


def main():
    samples = []

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        annotation_path = ANNOTATION_DIR / f"{image_path.name}.json"

        label = has_defect(annotation_path)

        samples.append(
            {
                "filename": image_path.name,
                "has_defect": label,
            }
        )

    df = pd.DataFrame(samples)

    train_df, val_df = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        stratify=df["has_defect"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = train_df.sort_values("filename").reset_index(drop=True)
    val_df = val_df.sort_values("filename").reset_index(drop=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))

    print("\nTraining label distribution:")
    print(train_df["has_defect"].value_counts())

    print("\nValidation label distribution:")
    print(val_df["has_defect"].value_counts())


if __name__ == "__main__":
    main()