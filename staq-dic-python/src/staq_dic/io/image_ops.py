"""Image normalization and gradient computation.

Port of MATLAB io/normalize_img.m and io/img_gradient.m.

IMPORTANT — Coordinate convention difference:
    MATLAB uses transposed images internally: dim1=x (cols), dim2=y (rows).
    Python keeps standard (H, W) layout: dim0=y (rows), dim1=x (cols).
    All functions in this module use Python convention: images are (H, W).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate1d

from ..core.data_structures import GridxyROIRange, ImageGradients


def normalize_images(
    images: list[NDArray[np.float64]],
    roi: GridxyROIRange,
) -> tuple[list[NDArray[np.float64]], GridxyROIRange]:
    """Normalize a list of images using the ROI-based mean/std formula.

    Port of MATLAB normalize_img.m.

    For each image: normalized = (img - mean_roi) / std_roi,
    where mean and std are computed over the ROI sub-region.

    Args:
        images: List of grayscale images, each (H, W) float64.
        roi: Region of interest specifying the normalization region.

    Returns:
        (normalized_images, clamped_roi) where clamped_roi is adjusted
        to fit within image bounds.
    """
    if not images:
        return [], roi

    h, w = images[0].shape

    # Clamp ROI to image bounds (0-based)
    gx0 = max(0, roi.gridx[0])
    gx1 = min(w - 1, roi.gridx[1])
    gy0 = max(0, roi.gridy[0])
    gy1 = min(h - 1, roi.gridy[1])
    clamped_roi = GridxyROIRange(gridx=(gx0, gx1), gridy=(gy0, gy1))

    normalized = []
    for img in images:
        # Extract ROI sub-region for statistics
        # In Python convention: rows = y, cols = x
        roi_patch = img[gy0 : gy1 + 1, gx0 : gx1 + 1]
        avg = np.mean(roi_patch)
        std = np.std(roi_patch, ddof=0)
        if std < 1e-12:
            std = 1.0
        normalized.append((img - avg) / std)

    return normalized, clamped_roi


def compute_image_gradient(
    img_ref: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64] | None = None,
) -> ImageGradients:
    """Compute image gradients using 7-point central finite difference.

    Port of MATLAB img_gradient.m (finite difference branch only).

    Uses the 7-point stencil: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60].
    The gradient is cropped by 3 pixels on each border and multiplied
    by the mask.

    Args:
        img_ref: Reference image (H, W) float64.
        img_ref_mask: Binary mask (H, W) float64. If None, all ones.

    Returns:
        ImageGradients with df_dx, df_dy (cropped), and axis info.
    """
    h, w = img_ref.shape

    if img_ref_mask is None:
        img_ref_mask = np.ones((h, w), dtype=np.float64)

    # 7-point finite difference kernel
    kernel = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])

    # Crop region: 3 pixels from each border (0-based)
    # MATLAB uses DfDxStartx = 4 (1-based) -> Python: row_start = 3 (0-based)
    row_start = 3
    row_end = h - 3  # exclusive
    col_start = 3
    col_end = w - 3  # exclusive

    # Extract padded region for convolution
    I = img_ref[row_start - 3 : row_end + 3, col_start - 3 : col_end + 3]

    # Apply 1D correlation along each axis (matches MATLAB imfilter behavior)
    # axis=0 -> y-derivative (df/dy), axis=1 -> x-derivative (df/dx)
    df_dx_padded = correlate1d(I, kernel, axis=1, mode="nearest")
    df_dy_padded = correlate1d(I, kernel, axis=0, mode="nearest")

    # Crop the 3-pixel border caused by the filter
    df_dx = df_dx_padded[3:-3, 3:-3]
    df_dy = df_dy_padded[3:-3, 3:-3]

    # Apply mask to the cropped region
    mask_crop = img_ref_mask[row_start:row_end, col_start:col_end]
    df_dx = df_dx * mask_crop
    df_dy = df_dy * mask_crop

    return ImageGradients(
        df_dx=df_dx,
        df_dy=df_dy,
        img_ref_mask=img_ref_mask,
        img_size=(h, w),
    )
