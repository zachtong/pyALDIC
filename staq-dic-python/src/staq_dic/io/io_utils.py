"""Image and mask loading utilities.

Simplified I/O layer for loading images and masks from disk.
Uses OpenCV for image reading.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def load_images(image_dir: str | Path, pattern: str = "*.tif") -> list[NDArray[np.float64]]:
    """Load all images matching a pattern from a directory.

    Images are returned in sorted order as float64 grayscale arrays
    with shape (H, W).

    Args:
        image_dir: Directory containing images.
        pattern: Glob pattern for image files.

    Returns:
        List of (H, W) float64 arrays.
    """
    image_dir = Path(image_dir)
    paths = sorted(image_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No images matching '{pattern}' found in {image_dir}"
        )

    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image: {p}")
        images.append(img.astype(np.float64))
    return images


def load_masks(mask_dir: str | Path, pattern: str = "*.tif") -> list[NDArray[np.bool_]]:
    """Load binary masks from a directory.

    Masks are thresholded at 0.5 and returned as boolean arrays.

    Args:
        mask_dir: Directory containing mask images.
        pattern: Glob pattern for mask files.

    Returns:
        List of (H, W) boolean arrays.
    """
    mask_dir = Path(mask_dir)
    paths = sorted(mask_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No masks matching '{pattern}' found in {mask_dir}"
        )

    masks = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read mask: {p}")
        masks.append(img.astype(np.float64) / 255.0 > 0.5)
    return masks
