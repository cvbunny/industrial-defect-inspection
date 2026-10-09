from pathlib import Path

import cv2

from src.data.annotation import load_annotation_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kolektorsdd2-DatasetNinja"
)


def validate_split(split: str, show_label_stats: bool):
    image_dir = DATASET_ROOT / split / "img"
    annotation_dir = DATASET_ROOT / split / "ann"

    image_files = sorted(image_dir.glob("*.png"))

    total = len(image_files)
    normal = 0
    defect = 0
    missing_annotations = 0
    size_mismatches = 0

    for image_path in image_files:
        annotation_path = annotation_dir / f"{image_path.name}.json"

        if not annotation_path.exists():
            missing_annotations += 1
            print(f"Missing annotation: {image_path.name}")
            continue

        annotation = load_annotation_json(annotation_path)

        if show_label_stats:
            if len(annotation.get("objects", [])) == 0:
                normal += 1
            else:
                defect += 1

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Failed to read image: {image_path.name}")
            continue

        image_height, image_width = image.shape[:2]

        annotation_height = annotation["size"]["height"]
        annotation_width = annotation["size"]["width"]

        if (
            image_height != annotation_height
            or image_width != annotation_width
        ):
            size_mismatches += 1
            print(
                f"Size mismatch: {image_path.name} | "
                f"image=({image_height}, {image_width}), "
                f"annotation=({annotation_height}, {annotation_width})"
            )

    print()
    print(f"=== {split.upper()} ===")
    print(f"Total images: {total}")
    if show_label_stats:
        print(f"Normal images: {normal}")
        print(f"Defect images: {defect}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Size mismatches: {size_mismatches}")


def main():
    validate_split("train", show_label_stats=True)
    validate_split("test", show_label_stats=False)


if __name__ == "__main__":
    main()