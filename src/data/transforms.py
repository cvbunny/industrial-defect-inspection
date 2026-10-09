import cv2
import numpy as np


def pad_to_size(
    image: np.ndarray,
    mask: np.ndarray,
    target_height: int = 672,
    target_width: int = 256,
):
    """
    Symmetrically zero-pad an image and its mask to a fixed size.

    Args:
        image:
            NumPy array with shape (H, W, C).
        mask:
            NumPy array with shape (H, W).
        target_height:
            Final padded image height.
        target_width:
            Final padded image width.

    Returns:
        padded_image:
            Image with shape (target_height, target_width, C).
        padded_mask:
            Mask with shape (target_height, target_width).
    """
    height, width = image.shape[:2]

    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"Image and mask sizes do not match: "
            f"image={image.shape[:2]}, mask={mask.shape[:2]}"
        )

    if height > target_height or width > target_width:
        raise ValueError(
            f"Input size {(height, width)} exceeds "
            f"target size {(target_height, target_width)}"
        )

    pad_height = target_height - height
    pad_width = target_width - width

    top = pad_height // 2
    bottom = pad_height - top

    left = pad_width // 2
    right = pad_width - left

    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    padded_mask = cv2.copyMakeBorder(
        mask,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded_image, padded_mask