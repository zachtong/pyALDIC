"""Warp a binary mask from reference to deformed configuration.

Given a binary mask in the reference frame and a pixel-level displacement
field u(x,y), v(x,y) (defined at reference coordinates), produce the
corresponding mask in the deformed frame.

Algorithm: iterative inverse mapping with fixed-point iteration.
For each pixel (x', y') in the deformed frame, find (x, y) in the
reference frame such that x' = x + u(x,y), y' = y + v(x,y), then
set mask_def(x', y') = mask_ref(x, y).

The fixed-point iteration converges when max strain < 100%, which is
essentially always true in DIC applications.

Performance: images larger than max_warp_pixels (default 512*512) are
automatically downsampled before warping, then upsampled back, keeping
warp time around ~200ms regardless of image size.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label as ndimage_label, map_coordinates, zoom

# Default area threshold for downsampling (512*512 = 262144 pixels)
DEFAULT_MAX_WARP_PIXELS = 512 * 512


def warp_mask(
    mask: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    n_iter: int = 8,
    min_fragment_ratio: float = 0.01,
    max_warp_pixels: int = DEFAULT_MAX_WARP_PIXELS,
) -> NDArray[np.float64]:
    """Warp binary mask from reference frame to deformed frame.

    Parameters
    ----------
    mask : (H, W) float64 array
        Binary mask in reference frame coordinates (1 = ROI, 0 = outside).
    u : (H, W) float64 array
        Horizontal (x) displacement at each reference pixel.
        Convention: point (x, y) in reference moves to (x + u, y + v).
    v : (H, W) float64 array
        Vertical (y) displacement at each reference pixel.
    n_iter : int
        Number of fixed-point iterations for inverse mapping.
        Default 8 is sufficient for strains up to ~50%.
    min_fragment_ratio : float
        Minimum size of a connected component (as fraction of total mask
        area) to keep. Components smaller than this are removed as
        discretization artifacts. Set to 0 to disable cleanup.
    max_warp_pixels : int
        Maximum pixel count for the warp computation. Images with more
        pixels are downsampled to this level, warped, then upsampled back.
        Set to 0 to disable downsampling.

    Returns
    -------
    mask_warped : (H, W) float64 array
        Binary mask in deformed frame coordinates (1 = ROI, 0 = outside).
    """
    h, w = mask.shape
    if u.shape != (h, w) or v.shape != (h, w):
        raise ValueError(
            f"Shape mismatch: mask {mask.shape}, u {u.shape}, v {v.shape}"
        )

    n_pixels = h * w
    needs_downsample = max_warp_pixels > 0 and n_pixels > max_warp_pixels

    if needs_downsample:
        scale = math.sqrt(max_warp_pixels / n_pixels)
        mask_ds, u_ds, v_ds = _downsample(mask, u, v, scale)
        warped_ds = _warp_core(mask_ds, u_ds, v_ds, n_iter)
        result = _upsample(warped_ds, h, w)
    else:
        result = _warp_core(mask, u, v, n_iter)

    if min_fragment_ratio > 0:
        result = _remove_small_fragments(result, min_fragment_ratio)

    return result


def _downsample(
    mask: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Downsample mask and displacement fields by scale factor.

    Uses stride-based sampling for speed (avoids expensive scipy.zoom on
    large arrays). Displacement values are scaled proportionally.
    """
    step = max(1, round(1.0 / scale))
    mask_ds = mask[::step, ::step].copy()
    u_ds = u[::step, ::step].copy() / step
    v_ds = v[::step, ::step].copy() / step
    return mask_ds, u_ds, v_ds


def _upsample(
    mask_ds: NDArray[np.float64],
    target_h: int,
    target_w: int,
) -> NDArray[np.float64]:
    """Upsample warped mask back to original resolution via repeat."""
    ds_h, ds_w = mask_ds.shape
    step_y = max(1, round(target_h / ds_h))
    step_x = max(1, round(target_w / ds_w))
    upsampled = np.repeat(np.repeat(mask_ds, step_y, axis=0), step_x, axis=1)
    # Crop to exact target size (repeat may overshoot by a few pixels)
    result = upsampled[:target_h, :target_w]
    # Pad if undersized
    if result.shape[0] < target_h or result.shape[1] < target_w:
        padded = np.zeros((target_h, target_w), dtype=np.float64)
        padded[: result.shape[0], : result.shape[1]] = result
        result = padded
    return (result > 0.5).astype(np.float64)


def _warp_core(
    mask: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    n_iter: int,
) -> NDArray[np.float64]:
    """Core inverse-mapping warp (no downsampling, no cleanup)."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Iterative fixed-point to invert the mapping:
    #   x' = x + u(x, y)  =>  x = x' - u(x, y)
    #   Start with x_0 = x' (identity guess)
    #   Iterate: x_{n+1} = x' - u(x_n, y_n)
    src_x = xx.copy()
    src_y = yy.copy()

    for _ in range(n_iter):
        coords = np.array([src_y.ravel(), src_x.ravel()])
        u_src = map_coordinates(
            u, coords, order=1, mode="constant", cval=0.0
        ).reshape(h, w)
        v_src = map_coordinates(
            v, coords, order=1, mode="constant", cval=0.0
        ).reshape(h, w)
        src_x = xx - u_src
        src_y = yy - v_src

    # Nearest-neighbor sampling to preserve binary mask
    coords_final = np.array([src_y.ravel(), src_x.ravel()])
    mask_warped = map_coordinates(
        mask, coords_final, order=0, mode="constant", cval=0.0
    ).reshape(h, w)

    return (mask_warped > 0.5).astype(np.float64)


def _remove_small_fragments(
    mask: NDArray[np.float64],
    min_ratio: float,
) -> NDArray[np.float64]:
    """Remove small connected components from both domains and holes.

    Parameters
    ----------
    mask : (H, W) binary mask (0 or 1).
    min_ratio : Minimum component size as fraction of total mask=1 area.

    Returns
    -------
    Cleaned mask with small fragments removed and small holes filled.
    """
    total_area = mask.sum()
    if total_area == 0:
        return mask

    min_size = max(1, total_area * min_ratio)
    result = mask.copy()

    # Remove small mask=1 fragments
    labeled, n_labels = ndimage_label(result > 0.5)
    for i in range(1, n_labels + 1):
        if (labeled == i).sum() < min_size:
            result[labeled == i] = 0.0

    # Fill small mask=0 holes (internal zero-regions not touching border)
    h, w = result.shape
    labeled_z, n_zero = ndimage_label(result < 0.5)
    border_labels: set[int] = set()
    border_labels.update(labeled_z[0, :].tolist())
    border_labels.update(labeled_z[h - 1, :].tolist())
    border_labels.update(labeled_z[:, 0].tolist())
    border_labels.update(labeled_z[:, w - 1].tolist())
    border_labels.discard(0)

    for i in range(1, n_zero + 1):
        if i not in border_labels and (labeled_z == i).sum() < min_size:
            result[labeled_z == i] = 1.0

    return result
