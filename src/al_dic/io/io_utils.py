"""Image and mask loading utilities with Unicode-safe I/O and multi-bit-depth support.

All reading uses np.fromfile + cv2.imdecode to handle Windows paths
containing CJK or other non-ASCII characters (cv2.imread fails on those).

Supported bit depths: uint8, uint16, uint32, float32/float64.
Supported formats: tif, tiff, png, bmp, jpg, jpeg, jp2, webp.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────────
# Low-level helpers (shared by load_images, load_masks, read_mask_as_bool)
# ──────────────────────────────────────────────────────────────────


def _read_unchanged(path: str | Path) -> NDArray:
    """Read an image preserving original bit depth and channel count.

    Uses np.fromfile + cv2.imdecode for Unicode-safe path handling.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be decoded.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    buf = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to decode image: {path}")
    return img


def _to_grayscale(img: NDArray) -> NDArray:
    """Convert multi-channel image to single channel, preserving dtype."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _normalize_to_float64(img: NDArray) -> NDArray[np.float64]:
    """Normalize any bit depth to float64 in [0.0, 1.0]."""
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    if img.dtype == np.uint32:
        return img.astype(np.float64) / 4294967295.0
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float64), 0.0, 1.0)
    # Fallback: unsigned int, scale by max
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def _to_uint8(img: NDArray) -> NDArray[np.uint8]:
    """Convert any bit depth to uint8 [0, 255]."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    if img.dtype == np.uint32:
        return (img / 16777216).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    info = np.iinfo(img.dtype)
    return (img.astype(np.float64) / info.max * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────


def load_images(
    image_dir: str | Path, pattern: str = "*.tif"
) -> list[NDArray[np.float64]]:
    """Load all images matching a pattern from a directory.

    Images are returned in sorted order as float64 grayscale arrays
    with shape (H, W) and values in [0.0, 1.0].

    Supports all common bit depths (uint8, uint16, uint32, float)
    and file formats (tif, tiff, png, bmp, jpg, jpeg, jp2, webp).

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

    images: list[NDArray[np.float64]] = []
    for p in paths:
        raw = _read_unchanged(p)
        gray = _to_grayscale(raw)
        images.append(_normalize_to_float64(gray))
    return images


def load_masks(
    mask_dir: str | Path, pattern: str = "*.tif"
) -> list[NDArray[np.bool_]]:
    """Load binary masks from a directory.

    Masks are read at their native bit depth, converted to grayscale
    uint8, and thresholded at 127 (midpoint).

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

    masks: list[NDArray[np.bool_]] = []
    for p in paths:
        raw = _read_unchanged(p)
        gray = _to_grayscale(raw)
        gray_u8 = _to_uint8(gray)
        masks.append(gray_u8 > 127)
    return masks


def read_mask_as_bool(
    path: str | Path,
    target_shape: tuple[int, int] | None = None,
) -> NDArray[np.bool_]:
    """Read a single mask image as a boolean array.

    Supports all bit depths and file formats.  Unicode-safe on Windows.
    Pixels brighter than 50% become True.

    Args:
        path: Path to the mask image file.
        target_shape: Optional (H, W) to resize to (nearest-neighbour)
            if dimensions differ.

    Returns:
        (H, W) boolean mask array.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be decoded.
    """
    raw = _read_unchanged(path)
    gray = _to_grayscale(raw)
    gray_u8 = _to_uint8(gray)
    if target_shape is not None and gray_u8.shape[:2] != target_shape:
        gray_u8 = cv2.resize(
            gray_u8,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return gray_u8 > 127
