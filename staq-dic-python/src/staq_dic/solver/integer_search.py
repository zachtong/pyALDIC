"""FFT-based integer-pixel displacement search.

Port of MATLAB solver/integer_search.m + integer_search_kernel.m
(Jin Yang, Caltech).  Redesigned for Python using OpenCV matchTemplate.

Computes the initial displacement guess by normalized cross-correlation
(NCC) between the reference and deformed images.  Each mesh node's
neighbourhood in the reference image is matched against a search region
in the deformed image.

MATLAB/Python differences:
    - MATLAB ``normxcorr2`` -> ``cv2.matchTemplate`` with
      ``cv2.TM_CCOEFF_NORMED``.
    - MATLAB uses transposed images ``f(x, y)``; Python keeps standard
      ``f[y, x]``.  All coordinate handling is adjusted accordingly.
    - Sub-pixel refinement via 9-point quadratic polynomial fit
      (same as MATLAB ``findpeak.m``).
    - The 3-file MATLAB structure is consolidated here.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara, GridxyROIRange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def integer_search(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    para: DICPara,
) -> tuple[
    NDArray[np.float64],  # x0 (M,)
    NDArray[np.float64],  # y0 (N,)
    NDArray[np.float64],  # u  (N, M)
    NDArray[np.float64],  # v  (N, M)
    dict,                 # info
]:
    """Compute initial displacement via FFT cross-correlation.

    Performs normalized cross-correlation between reference and deformed
    images at each grid point to find the best-matching displacement.

    Args:
        f_img: Reference image (H, W), float64.
        g_img: Deformed image (H, W), float64.
        para: DIC parameters.  Uses ``gridxy_roi_range``, ``winsize``,
            ``winstepsize``, ``size_of_fft_search_region``.

    Returns:
        ``(x0, y0, u, v, info)`` where:
            - ``x0``: 1-D grid x-coordinates (M,).
            - ``y0``: 1-D grid y-coordinates (N,).
            - ``u``: x-displacement grid (N, M), float64 (sub-pixel).
            - ``v``: y-displacement grid (N, M), float64 (sub-pixel).
            - ``info``: Dict with ``'cc_max'`` (N, M) peak NCC values,
              ``'qfactors'`` (N, M, 2) quality factors [PCE, PPE],
              ``'search_region_warning'`` flag.
    """
    h, w = f_img.shape
    roi = para.gridxy_roi_range
    winsize = para.winsize
    winstepsize = para.winstepsize
    search = para.size_of_fft_search_region
    half_w = winsize // 2

    # Generate grid points (node centers)
    # Shrink ROI so that subset + search region fits within image
    min_x = max(roi.gridx[0], half_w + search)
    max_x = min(roi.gridx[1], w - 1 - half_w - search)
    min_y = max(roi.gridy[0], half_w + search)
    max_y = min(roi.gridy[1], h - 1 - half_w - search)

    search_region_warning = False
    if min_x >= max_x or min_y >= max_y:
        logger.warning(
            "SizeOfFFTSearchRegion (%d) too large for image (%dx%d) "
            "with winsize=%d. Reducing search region.",
            search, h, w, winsize,
        )
        search_region_warning = True
        # Fall back: use minimal search region
        search = max(1, min(search, min(h, w) // 4 - half_w))
        min_x = max(roi.gridx[0], half_w + search)
        max_x = min(roi.gridx[1], w - 1 - half_w - search)
        min_y = max(roi.gridy[0], half_w + search)
        max_y = min(roi.gridy[1], h - 1 - half_w - search)

    x0 = np.arange(min_x, max_x + 1, winstepsize, dtype=np.float64)
    y0 = np.arange(min_y, max_y + 1, winstepsize, dtype=np.float64)

    if len(x0) == 0 or len(y0) == 0:
        raise ValueError(
            f"No grid points generated. Image ({h}x{w}), winsize={winsize}, "
            f"search={search}, ROI=({roi.gridx}, {roi.gridy})"
        )

    ny, nx = len(y0), len(x0)
    u_grid = np.zeros((ny, nx), dtype=np.float64)
    v_grid = np.zeros((ny, nx), dtype=np.float64)
    cc_max = np.zeros((ny, nx), dtype=np.float64)
    qfactors = np.zeros((ny, nx, 2), dtype=np.float64)

    # Convert to float32 for OpenCV matchTemplate
    f32 = f_img.astype(np.float32)
    g32 = g_img.astype(np.float32)

    subset_size = 2 * half_w + 1  # winsize + 1 for centered subset

    for iy in range(ny):
        for ix in range(nx):
            cx = int(round(x0[ix]))
            cy = int(round(y0[iy]))

            # Extract template from reference image
            t_y0 = cy - half_w
            t_y1 = cy + half_w + 1
            t_x0 = cx - half_w
            t_x1 = cx + half_w + 1

            # Extract search region from deformed image
            s_y0 = t_y0 - search
            s_y1 = t_y1 + search
            s_x0 = t_x0 - search
            s_x1 = t_x1 + search

            # Boundary check
            if s_y0 < 0 or s_x0 < 0 or s_y1 > h or s_x1 > w:
                u_grid[iy, ix] = np.nan
                v_grid[iy, ix] = np.nan
                cc_max[iy, ix] = 0.0
                qfactors[iy, ix] = [np.inf, np.inf]
                continue

            template = f32[t_y0:t_y1, t_x0:t_x1]
            search_img = g32[s_y0:s_y1, s_x0:s_x1]

            # Check for constant template (zero variance)
            if np.std(template) < 1e-6:
                u_grid[iy, ix] = np.nan
                v_grid[iy, ix] = np.nan
                cc_max[iy, ix] = 0.0
                qfactors[iy, ix] = [np.inf, np.inf]
                continue

            # NCC via OpenCV matchTemplate
            # Result size: (2*search + 1, 2*search + 1)
            ncc_map = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)

            # Sub-pixel peak finding
            peak_x, peak_y, peak_val = _findpeak_subpixel(ncc_map)

            # Displacement = peak - center (zero-displacement point)
            center = search  # zero displacement is at index 'search'
            u_grid[iy, ix] = peak_x - center
            v_grid[iy, ix] = peak_y - center
            cc_max[iy, ix] = peak_val

            # Quality factors
            qfactors[iy, ix] = _compute_qfactors(ncc_map, peak_val)

    # Apply reference mask if available
    if para.img_ref_mask is not None:
        mask = para.img_ref_mask
        for iy in range(ny):
            for ix in range(nx):
                cx = int(round(x0[ix]))
                cy = int(round(y0[iy]))
                if 0 <= cy < h and 0 <= cx < w:
                    if mask[cy, cx] < 0.5:
                        u_grid[iy, ix] = np.nan
                        v_grid[iy, ix] = np.nan

    n_valid = np.sum(np.isfinite(u_grid))
    logger.info(
        "Integer search complete: %d/%d valid (%dx%d grid), "
        "mean cc=%.3f",
        n_valid, ny * nx, ny, nx,
        np.nanmean(cc_max),
    )

    info = dict(
        cc_max=cc_max,
        qfactors=qfactors,
        search_region_warning=search_region_warning,
    )
    return x0, y0, u_grid, v_grid, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _findpeak_subpixel(
    ncc: NDArray[np.float32],
) -> tuple[float, float, float]:
    """Find peak in NCC map with sub-pixel precision.

    Port of MATLAB findpeak.m: finds the maximum, then fits a 2nd-order
    polynomial to the 3x3 neighbourhood for sub-pixel refinement.

    Args:
        ncc: 2-D NCC map.

    Returns:
        (x_peak, y_peak, peak_value) with sub-pixel precision.
    """
    # Find integer peak
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc)
    px, py = max_loc  # OpenCV returns (x, y) = (col, row)

    nh, nw = ncc.shape
    # If peak is on the edge, return integer peak
    if px <= 0 or px >= nw - 1 or py <= 0 or py >= nh - 1:
        return float(px), float(py), float(max_val)

    # 9-point quadratic polynomial fit
    # u(x, y) = A0 + A1*x + A2*y + A3*x*y + A4*x^2 + A5*y^2
    patch = ncc[py - 1 : py + 2, px - 1 : px + 2].astype(np.float64).ravel()

    xs = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float64)
    ys = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float64)

    # Design matrix: [1, x, y, x*y, x^2, y^2]
    X = np.column_stack([np.ones(9), xs, ys, xs * ys, xs ** 2, ys ** 2])

    # Least-squares solve
    A, _, _, _ = np.linalg.lstsq(X, patch, rcond=None)

    # Extremum: dU/dx = A1 + A3*y + 2*A4*x = 0
    #           dU/dy = A2 + A3*x + 2*A5*y = 0
    denom = A[3] ** 2 - 4 * A[4] * A[5]
    if abs(denom) < 1e-12:
        return float(px), float(py), float(max_val)

    x_offset = (-A[2] * A[3] + 2 * A[5] * A[1]) / denom
    y_offset = (-A[3] * A[1] + 2 * A[4] * A[2]) / denom

    # If offset exceeds ±1, fall back to integer peak
    if abs(x_offset) > 1.0 or abs(y_offset) > 1.0:
        return float(px), float(py), float(max_val)

    # Round to 1/1000 pixel precision
    x_offset = round(x_offset * 1000) / 1000
    y_offset = round(y_offset * 1000) / 1000

    # Evaluate polynomial at extremum
    peak_val = float(
        A[0]
        + A[1] * x_offset
        + A[2] * y_offset
        + A[3] * x_offset * y_offset
        + A[4] * x_offset ** 2
        + A[5] * y_offset ** 2
    )

    return float(px) + x_offset, float(py) + y_offset, peak_val


def _compute_qfactors(
    ncc: NDArray[np.float32],
    peak_val: float,
) -> NDArray[np.float64]:
    """Compute quality factors for an NCC map.

    Two metrics from Xue (2014) PIV correlation SNR:
        - PCE: peak-to-correlation-energy ratio
        - PPE: peak-to-entropy ratio

    Args:
        ncc: 2-D NCC map.
        peak_val: Peak NCC value.

    Returns:
        [PCE, PPE] array.
    """
    ncc_64 = ncc.astype(np.float64)
    ncc_shifted = ncc_64 - ncc_64.min()

    # PCE: peak^2 / mean(values^2)
    energy = np.mean(ncc_shifted ** 2)
    pce = (peak_val ** 2) / energy if energy > 1e-20 else np.inf

    # PPE: 1 / entropy
    hist, _ = np.histogram(ncc_shifted, bins=30)
    p = hist / hist.sum()
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    ppe = 1.0 / entropy if entropy > 1e-20 else np.inf

    return np.array([pce, ppe], dtype=np.float64)
