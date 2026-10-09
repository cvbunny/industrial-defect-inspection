from pathlib import Path
from collections import Counter

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kolektorsdd2-DatasetNinja"
    / "train"
    / "img"
)


def main():
    sizes = []

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Failed to read: {image_path.name}")
            continue

        height, width = image.shape[:2]
        sizes.append((height, width))

    size_counts = Counter(sizes)

    print(f"Total images: {len(sizes)}")
    print(f"Unique sizes: {len(size_counts)}")

    print("\nMost common sizes:")
    for size, count in size_counts.most_common(10):
        print(f"{size}: {count}")

    heights = [h for h, w in sizes]
    widths = [w for h, w in sizes]

    print()
    print(f"Height range: {min(heights)} - {max(heights)}")
    print(f"Width range: {min(widths)} - {max(widths)}")


if __name__ == "__main__":
    main()