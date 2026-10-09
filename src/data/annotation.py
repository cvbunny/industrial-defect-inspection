from pathlib import Path
import json
import base64
import zlib

import cv2
import numpy as np


def load_annotation_json(annotation_path: str | Path) -> dict:
    annotation_path = Path(annotation_path)

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)
 
    return annotation


def decode_supervisely_bitmap(bitmap_data: str) -> np.ndarray:
    """
    Decode Supervisely bitmap annotation data into a mask image.

    Returns:
        bitmap_mask: np.ndarray of shape (H, W), values in {0, 1}
    """
    compressed = base64.b64decode(bitmap_data)
    decompressed = zlib.decompress(compressed)

    bitmap_array = np.frombuffer(decompressed, dtype=np.uint8)
    bitmap_image = cv2.imdecode(bitmap_array, cv2.IMREAD_UNCHANGED)

    if bitmap_image is None:
        raise ValueError("Failed to decode Supervisely bitmap data.")

    # Supervisely bitmap often decodes to RGBA image;
    # the alpha channel indicates the object mask.
    if bitmap_image.ndim == 3 and bitmap_image.shape[2] == 4:
        alpha = bitmap_image[:, :, 3]
        bitmap_mask = (alpha > 0).astype(np.uint8)
    elif bitmap_image.ndim == 2:
        bitmap_mask = (bitmap_image > 0).astype(np.uint8)
    else:
        # fallback: use any nonzero pixel
        bitmap_mask = (bitmap_image.sum(axis=-1) > 0).astype(np.uint8)

    return bitmap_mask


def annotation_to_mask(annotation: dict) -> np.ndarray:
    """
    Convert a Supervisely annotation dict into a full-size binary mask.

    Returns:
        full_mask: np.ndarray of shape (height, width), values in {0, 1}
    """
    height = annotation["size"]["height"]
    width = annotation["size"]["width"]

    full_mask = np.zeros((height, width), dtype=np.uint8)

    for obj in annotation.get("objects", []):
        geometry_type = obj.get("geometryType")

        if geometry_type != "bitmap":
            raise NotImplementedError(
                f"Unsupported geometryType: {geometry_type}"
            )

        bitmap_info = obj["bitmap"]
        bitmap_data = bitmap_info["data"]
        origin_x, origin_y = bitmap_info["origin"]

        object_mask = decode_supervisely_bitmap(bitmap_data)

        h, w = object_mask.shape
        y1, y2 = origin_y, origin_y + h
        x1, x2 = origin_x, origin_x + w

        full_mask[y1:y2, x1:x2] = np.maximum(
            full_mask[y1:y2, x1:x2],
            object_mask
        )

    return full_mask