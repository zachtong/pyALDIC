"""NCC-based displacement search with optional image pyramid.

Port of MATLAB solver/integer_search.m + integer_search_kernel.m
(Jin Yang, Caltech).  Redesigned for Python using OpenCV matchTemplate.

Computes the initial displacement guess by normalized cross-correlation
(NCC) between the reference and deformed images.  Each mesh node's
neighbourhood in the reference image is matched against a search region
in the deformed image.

Two search strategies:
    - **Direct search** (default): Fixed search region per node.
      Good for small-to-medium displacements (< search_region pixels).
    - **Pyramid search**: Multi-scale coarse-to-fine NCC.
      Handles large displacements without increasing search region.
      Uses ``cv2.pyrDown`` image pyramid; at the coarsest level the
      effective search range is ``search * 2^(n_levels-1)`` pixels.

Sub-pixel refinement uses a 9-point quadratic polynomial fit
(same as MATLAB ``findpeak.m``).
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


def integer_search_pyramid(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    para: DICPara,
    n_levels: int | None = None,
    refine_search: int = 3,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict,
]:
    """Multi-scale NCC search using image pyramid.

    Builds a Gaussian pyramid (via ``cv2.pyrDown``), performs NCC search
    at the coarsest level to capture large displacements, then refines
    at progressively finer levels with a small search window.

    Effective search range at full resolution:
        ``search_region * 2^(n_levels - 1) + refine_search * sum(2^k)``

    For a 512×512 image with search=10 and 3 levels, this covers
    ~40 + 14 = 54 pixels vs the direct method's 10 pixels.

    Args:
        f_img: Reference image (H, W), float64.
        g_img: Deformed image (H, W), float64.
        para: DIC parameters (same as ``integer_search``).
        n_levels: Number of pyramid levels (1 = no pyramid, same as direct).
            Default: auto-determined from image size (up to 4 levels).
        refine_search: Search radius at refinement levels (pixels at that
            level's resolution). Default 3.

    Returns:
        Same as ``integer_search``: ``(x0, y0, u, v, info)``.
    """
    h, w = f_img.shape
    winsize = para.winsize
    search = para.size_of_fft_search_region

    # Auto-determine pyramid levels: stop when image is < 64px or winsize < 5px
    if n_levels is None:
        n_levels = 1
        test_h, test_w, test_win = h, w, winsize
        while test_h >= 128 and test_w >= 128 and test_win >= 10 and n_levels < 4:
            n_levels += 1
            test_h //= 2
            test_w //= 2
            test_win //= 2

    if n_levels <= 1:
        return integer_search(f_img, g_img, para)

    # Build image pyramids
    f_pyr = [f_img]
    g_pyr = [g_img]
    for _ in range(n_levels - 1):
        f_pyr.append(cv2.pyrDown(f_pyr[-1].astype(np.float32)).astype(np.float64))
        g_pyr.append(cv2.pyrDown(g_pyr[-1].astype(np.float32)).astype(np.float64))

    logger.info(
        "Pyramid search: %d levels, sizes %s",
        n_levels,
        [f.shape for f in f_pyr],
    )

    # Coarsest level: full search with scaled parameters
    scale = 2 ** (n_levels - 1)
    coarse_h, coarse_w = f_pyr[-1].shape
    from dataclasses import replace
    coarse_para = replace(
        para,
        winsize=max(5, winsize // scale),
        winstepsize=max(1, para.winstepsize // scale),
        size_of_fft_search_region=max(3, search),
        gridxy_roi_range=GridxyROIRange(
            gridx=(max(0, para.gridxy_roi_range.gridx[0] // scale),
                   min(coarse_w - 1, para.gridxy_roi_range.gridx[1] // scale)),
            gridy=(max(0, para.gridxy_roi_range.gridy[0] // scale),
                   min(coarse_h - 1, para.gridxy_roi_range.gridy[1] // scale)),
        ),
        img_size=(coarse_h, coarse_w),
        img_ref_mask=None,  # Mask applied at finest level only
    )

    x0_c, y0_c, u_c, v_c, info_c = integer_search(
        f_pyr[-1], g_pyr[-1], coarse_para,
    )

    # Refine through intermediate levels
    for level in range(n_levels - 2, 0, -1):
        scale_l = 2 ** level
        img_f = f_pyr[level]
        img_g = g_pyr[level]
        lh, lw = img_f.shape

        # Scale up coordinates and displacements from previous level
        x0_l = np.clip(x0_c * 2, 0, lw - 1)
        y0_l = np.clip(y0_c * 2, 0, lh - 1)
        u_l = u_c * 2
        v_l = v_c * 2

        # Refine at this level: small search around predicted position
        u_l, v_l, cc_l = _refine_at_level(
            img_f, img_g, x0_l, y0_l, u_l, v_l,
            half_w=max(2, winsize // scale_l // 2),
            search=refine_search,
        )
        x0_c, y0_c, u_c, v_c = x0_l, y0_l, u_l, v_l

    # Final refinement at full resolution
    x0_full = np.clip(x0_c * 2, 0, w - 1)
    y0_full = np.clip(y0_c * 2, 0, h - 1)
    u_full = u_c * 2
    v_full = v_c * 2

    half_w = winsize // 2
    u_full, v_full, cc_full = _refine_at_level(
        f_img, g_img, x0_full, y0_full, u_full, v_full,
        half_w=half_w,
        search=refine_search,
    )

    # Snap grid to the standard winstepsize grid expected by mesh_setup
    # The pyramid may produce non-standard coordinates; remap to standard grid
    x0_std, y0_std, u_std, v_std, cc_std = _remap_to_standard_grid(
        x0_full, y0_full, u_full, v_full, cc_full,
        para, h, w,
    )

    ny, nx = len(y0_std), len(x0_std)
    qfactors = np.zeros((ny, nx, 2), dtype=np.float64)  # Not computed for pyramid

    # Apply reference mask
    if para.img_ref_mask is not None:
        mask = para.img_ref_mask
        for iy in range(ny):
            for ix in range(nx):
                cx = int(round(x0_std[ix]))
                cy = int(round(y0_std[iy]))
                if 0 <= cy < h and 0 <= cx < w:
                    if mask[cy, cx] < 0.5:
                        u_std[iy, ix] = np.nan
                        v_std[iy, ix] = np.nan

    n_valid = np.sum(np.isfinite(u_std))
    logger.info(
        "Pyramid search complete: %d/%d valid (%dx%d grid), "
        "mean cc=%.3f",
        n_valid, ny * nx, ny, nx,
        np.nanmean(cc_std),
    )

    info = dict(
        cc_max=cc_std,
        qfactors=qfactors,
        search_region_warning=False,
    )
    return x0_std, y0_std, u_std, v_std, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _refine_at_level(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    u_init: NDArray[np.float64],
    v_init: NDArray[np.float64],
    half_w: int,
    search: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Refine displacement at a single pyramid level via NCC.

    For each grid node, extracts a template from the reference at (x0, y0)
    and searches in the deformed image at (x0 + u_init, y0 + v_init) ± search.

    Args:
        f_img: Reference at this level.
        g_img: Deformed at this level.
        x0: Grid x-coords (M,).
        y0: Grid y-coords (N,).
        u_init: Initial u-displacement (N, M).
        v_init: Initial v-displacement (N, M).
        half_w: Half subset size.
        search: Search radius.

    Returns:
        (u_refined, v_refined, cc_max) all shape (N, M).
    """
    h, w = f_img.shape
    ny, nx = len(y0), len(x0)
    u_out = u_init.copy()
    v_out = v_init.copy()
    cc_out = np.zeros((ny, nx), dtype=np.float64)

    f32 = f_img.astype(np.float32)
    g32 = g_img.astype(np.float32)

    for iy in range(ny):
        for ix in range(nx):
            cx = int(round(x0[ix]))
            cy = int(round(y0[iy]))

            # Template from reference
            t_y0 = cy - half_w
            t_y1 = cy + half_w + 1
            t_x0 = cx - half_w
            t_x1 = cx + half_w + 1

            if t_y0 < 0 or t_x0 < 0 or t_y1 > h or t_x1 > w:
                u_out[iy, ix] = np.nan
                v_out[iy, ix] = np.nan
                continue

            template = f32[t_y0:t_y1, t_x0:t_x1]
            if np.std(template) < 1e-6:
                u_out[iy, ix] = np.nan
                v_out[iy, ix] = np.nan
                continue

            # Search in deformed image centered at predicted position
            pred_cx = int(round(cx + u_init[iy, ix]))
            pred_cy = int(round(cy + v_init[iy, ix]))
            s_y0 = pred_cy - half_w - search
            s_y1 = pred_cy + half_w + 1 + search
            s_x0 = pred_cx - half_w - search
            s_x1 = pred_cx + half_w + 1 + search

            # Clamp to image bounds
            s_y0_c = max(0, s_y0)
            s_y1_c = min(h, s_y1)
            s_x0_c = max(0, s_x0)
            s_x1_c = min(w, s_x1)

            search_img = g32[s_y0_c:s_y1_c, s_x0_c:s_x1_c]
            if (search_img.shape[0] < template.shape[0] or
                    search_img.shape[1] < template.shape[1]):
                # Search region too small after clamping
                continue

            ncc_map = cv2.matchTemplate(
                search_img, template, cv2.TM_CCOEFF_NORMED,
            )

            if ncc_map.size == 0:
                continue

            peak_x, peak_y, peak_val = _findpeak_subpixel(ncc_map)

            # Convert peak position to displacement
            # Peak (0,0) in ncc_map means template top-left is at search_img[0,0]
            # Template center is at (half_w, half_w) relative to template top-left
            # Search image starts at (s_x0_c, s_y0_c) in global coords
            match_x = s_x0_c + peak_x + half_w
            match_y = s_y0_c + peak_y + half_w
            u_out[iy, ix] = match_x - cx
            v_out[iy, ix] = match_y - cy
            cc_out[iy, ix] = peak_val

    return u_out, v_out, cc_out


def _remap_to_standard_grid(
    x0_pyr: NDArray[np.float64],
    y0_pyr: NDArray[np.float64],
    u_pyr: NDArray[np.float64],
    v_pyr: NDArray[np.float64],
    cc_pyr: NDArray[np.float64],
    para: DICPara,
    img_h: int,
    img_w: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Remap pyramid grid results to the standard winstepsize grid.

    The pyramid may produce grid coordinates at slightly different positions
    than the direct search. This function interpolates the pyramid results
    onto the standard grid expected by mesh_setup.
    """
    from scipy.interpolate import RegularGridInterpolator

    roi = para.gridxy_roi_range
    winsize = para.winsize
    winstepsize = para.winstepsize
    half_w = winsize // 2
    search = para.size_of_fft_search_region

    # Standard grid (same as direct integer_search would produce)
    min_x = max(roi.gridx[0], half_w + 1)
    max_x = min(roi.gridx[1], img_w - 1 - half_w - 1)
    min_y = max(roi.gridy[0], half_w + 1)
    max_y = min(roi.gridy[1], img_h - 1 - half_w - 1)

    x0_std = np.arange(min_x, max_x + 1, winstepsize, dtype=np.float64)
    y0_std = np.arange(min_y, max_y + 1, winstepsize, dtype=np.float64)

    if len(x0_std) == 0 or len(y0_std) == 0:
        raise ValueError("Cannot generate standard grid from pyramid results.")

    ny_std, nx_std = len(y0_std), len(x0_std)

    # If pyramid grid matches standard grid, no interpolation needed
    if (len(x0_pyr) == nx_std and len(y0_pyr) == ny_std and
            np.allclose(x0_pyr, x0_std) and np.allclose(y0_pyr, y0_std)):
        return x0_std, y0_std, u_pyr, v_pyr, cc_pyr

    # Interpolate pyramid results onto standard grid
    # Replace NaN with nearest valid value for interpolation
    u_filled = u_pyr.copy()
    v_filled = v_pyr.copy()
    nan_mask = np.isnan(u_filled) | np.isnan(v_filled)
    if np.all(nan_mask):
        # All NaN — return NaN grid
        u_std = np.full((ny_std, nx_std), np.nan)
        v_std = np.full((ny_std, nx_std), np.nan)
        cc_std = np.zeros((ny_std, nx_std))
        return x0_std, y0_std, u_std, v_std, cc_std

    if np.any(nan_mask):
        from scipy.ndimage import generic_filter
        for arr in [u_filled, v_filled]:
            mask = np.isnan(arr)
            if np.any(mask):
                mean_val = np.nanmean(arr)
                arr[mask] = mean_val

    try:
        interp_u = RegularGridInterpolator(
            (y0_pyr, x0_pyr), u_filled,
            method="linear", bounds_error=False, fill_value=None,
        )
        interp_v = RegularGridInterpolator(
            (y0_pyr, x0_pyr), v_filled,
            method="linear", bounds_error=False, fill_value=None,
        )
        yy, xx = np.meshgrid(y0_std, x0_std, indexing="ij")
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        u_std = interp_u(pts).reshape(ny_std, nx_std)
        v_std = interp_v(pts).reshape(ny_std, nx_std)
    except ValueError:
        # Interpolation failed — fall back to nearest
        u_std = np.full((ny_std, nx_std), np.nanmean(u_filled))
        v_std = np.full((ny_std, nx_std), np.nanmean(v_filled))

    cc_std = np.zeros((ny_std, nx_std), dtype=np.float64)
    return x0_std, y0_std, u_std, v_std, cc_std


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
