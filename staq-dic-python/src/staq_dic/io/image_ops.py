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

    # The 7-point stencil needs 3 neighbors on each side, so the gradient
    # is only valid for pixels [3, H-4] x [3, W-4].  MATLAB stores a
    # cropped (H-6)×(W-6) array and adjusts indices with DfCropWidth in
    # every consumer.  For simplicity, we return FULL-SIZE (H, W) arrays
    # with zeros in the 3-pixel border, so df_dx[y, x] directly gives the
    # gradient at pixel (y, x) without any offset arithmetic.
    crop = 3  # half-width of the 7-point stencil

    # Apply 1D correlation along each axis on the full image
    # axis=0 -> y-derivative (df/dy), axis=1 -> x-derivative (df/dx)
    df_dx_full = correlate1d(img_ref, kernel, axis=1, mode="nearest")
    df_dy_full = correlate1d(img_ref, kernel, axis=0, mode="nearest")

    # Zero out the 3-pixel border where the stencil is unreliable
    df_dx = np.zeros((h, w), dtype=np.float64)
    df_dy = np.zeros((h, w), dtype=np.float64)
    df_dx[crop:h - crop, crop:w - crop] = df_dx_full[crop:h - crop, crop:w - crop]
    df_dy[crop:h - crop, crop:w - crop] = df_dy_full[crop:h - crop, crop:w - crop]

    # Apply mask
    df_dx *= img_ref_mask
    df_dy *= img_ref_mask

    return ImageGradients(
        df_dx=df_dx,
        df_dy=df_dy,
        img_ref_mask=img_ref_mask,
        img_size=(h, w),
    )
