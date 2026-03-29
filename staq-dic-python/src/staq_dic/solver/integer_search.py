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

Performance notes:
    - Vectorized boundary check and template std via integral images.
    - Batch sub-pixel fitting via pre-computed pseudo-inverse.
    - Threaded matchTemplate for large node counts (OpenCV releases GIL).
    - Vectorized quality factor computation.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara, GridxyROIRange

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-computed constant for batch sub-pixel fitting (3x3 patch → 6 coeffs)
# ---------------------------------------------------------------------------
_XS = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float64)
_YS = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float64)
_X_DESIGN = np.column_stack([np.ones(9), _XS, _YS, _XS * _YS, _XS ** 2, _YS ** 2])
_X_PINV = np.linalg.pinv(_X_DESIGN)  # (6, 9) — computed once at import

# Offsets for extracting 3x3 patch (row-major: top-left → bottom-right)
_ROW_OFFSETS = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.intp)
_COL_OFFSETS = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.intp)


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
    """Compute initial displacement via NCC cross-correlation.

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

    # Vectorized NCC search
    u_grid, v_grid, cc_max, qfactors = _batch_ncc_search(
        f_img, g_img, x0, y0, half_w, search,
    )

    # Vectorized mask application
    if para.img_ref_mask is not None:
        _apply_mask_vectorized(u_grid, v_grid, x0, y0, para.img_ref_mask, h, w)

    n_valid = np.sum(np.isfinite(u_grid))

    # Detect peaks clipped by search region boundary.
    # If |u| or |v| >= search - 0.5, the true NCC peak may lie outside
    # the search window, indicating an insufficient search region.
    finite_mask = np.isfinite(u_grid) & np.isfinite(v_grid)
    clip_threshold = search - 0.5
    if np.any(finite_mask):
        clipped_nodes = finite_mask & (
            (np.abs(u_grid) >= clip_threshold)
            | (np.abs(v_grid) >= clip_threshold)
        )
        n_clipped = int(np.sum(clipped_nodes))
        max_abs_disp = float(max(
            np.nanmax(np.abs(u_grid)) if np.any(finite_mask) else 0.0,
            np.nanmax(np.abs(v_grid)) if np.any(finite_mask) else 0.0,
        ))
        peak_clipped = n_clipped > 0
    else:
        peak_clipped = False
        n_clipped = 0
        max_abs_disp = 0.0

    if peak_clipped:
        logger.warning(
            "FFT search: %d/%d nodes have peaks at search boundary "
            "(max |disp|=%.1f, search=%d). Consider increasing "
            "size_of_fft_search_region.",
            n_clipped, n_valid, max_abs_disp, search,
        )

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
        peak_clipped=peak_clipped,
        n_clipped=n_clipped,
        max_abs_disp=max_abs_disp,
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

    For a 512x512 image with search=10 and 3 levels, this covers
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
    x0_std, y0_std, u_std, v_std, cc_std = _remap_to_standard_grid(
        x0_full, y0_full, u_full, v_full, cc_full,
        para, h, w,
    )

    ny, nx = len(y0_std), len(x0_std)
    qfactors = np.zeros((ny, nx, 2), dtype=np.float64)

    # Apply reference mask (vectorized)
    if para.img_ref_mask is not None:
        _apply_mask_vectorized(u_std, v_std, x0_std, y0_std,
                               para.img_ref_mask, h, w)

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
# Vectorized batch NCC search (core of the optimization)
# ---------------------------------------------------------------------------


def _batch_ncc_search(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    half_w: int,
    search: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Vectorized NCC search for all grid nodes.

    Optimizations over the naive per-node loop:
    1. Integral images for O(1) boundary check + template std.
    2. Threaded cv2.matchTemplate (GIL-free) for N >= 500.
    3. Batch sub-pixel fitting via pre-computed pseudo-inverse.
    4. Vectorized quality factor computation.

    Returns:
        (u_grid, v_grid, cc_max, qfactors) all shape (ny, nx) or (ny,nx,2).
    """
    h, w = f_img.shape
    ny, nx = len(y0), len(x0)
    f32 = f_img.astype(np.float32)
    g32 = g_img.astype(np.float32)

    # --- Stage 1: Vectorized validity check ---
    cx_all = np.round(x0).astype(np.intp)  # (nx,)
    cy_all = np.round(y0).astype(np.intp)  # (ny,)
    CX, CY = np.meshgrid(cx_all, cy_all)   # (ny, nx) each

    # Boundary: search region must fit in image
    boundary_ok = (
        (CY - half_w - search >= 0) &
        (CX - half_w - search >= 0) &
        (CY + half_w + 1 + search <= h) &
        (CX + half_w + 1 + search <= w)
    )

    # Template std via integral images (O(1) per node, O(H*W) total)
    f32_sq = f32 ** 2
    int_f = cv2.integral(f32)      # (H+1, W+1), float64
    int_f2 = cv2.integral(f32_sq)  # (H+1, W+1), float64

    T_y0 = CY - half_w
    T_y1 = CY + half_w + 1
    T_x0 = CX - half_w
    T_x1 = CX + half_w + 1
    n_pix = float((2 * half_w + 1) ** 2)

    # Compute variance only for boundary-ok nodes
    ok_mask = boundary_ok
    ok_idx = np.where(ok_mask)
    var_f = np.zeros((ny, nx), dtype=np.float64)

    if len(ok_idx[0]) > 0:
        ty0 = T_y0[ok_idx]
        ty1 = T_y1[ok_idx]
        tx0 = T_x0[ok_idx]
        tx1 = T_x1[ok_idx]
        s_f = (int_f[ty1, tx1] - int_f[ty0, tx1]
               - int_f[ty1, tx0] + int_f[ty0, tx0])
        s_f2 = (int_f2[ty1, tx1] - int_f2[ty0, tx1]
                - int_f2[ty1, tx0] + int_f2[ty0, tx0])
        var_f[ok_idx] = np.maximum(s_f2 / n_pix - (s_f / n_pix) ** 2, 0.0)

    valid = ok_mask & (np.sqrt(var_f) > 1e-6)

    # --- Stage 2: matchTemplate for valid nodes ---
    valid_iy, valid_ix = np.where(valid)
    n_valid = len(valid_iy)

    ncc_h = 2 * search + 1
    ncc_w = 2 * search + 1

    # Pre-allocate output arrays (shared across threads)
    ncc_maps = np.zeros((n_valid, ncc_h, ncc_w), dtype=np.float32)
    ipeak_x = np.zeros(n_valid, dtype=np.intp)
    ipeak_y = np.zeros(n_valid, dtype=np.intp)
    ipeak_val = np.zeros(n_valid, dtype=np.float64)

    if n_valid > 0:
        if n_valid >= 500:
            _threaded_match(
                n_valid, valid_iy, valid_ix, cx_all, cy_all,
                f32, g32, half_w, search,
                ncc_maps, ipeak_x, ipeak_y, ipeak_val,
            )
        else:
            _sequential_match(
                n_valid, valid_iy, valid_ix, cx_all, cy_all,
                f32, g32, half_w, search,
                ncc_maps, ipeak_x, ipeak_y, ipeak_val,
            )

    # --- Stage 3: Batch sub-pixel peak finding ---
    sub_x, sub_y, sub_val = _batch_subpixel(
        ncc_maps, ipeak_x, ipeak_y, ipeak_val, ncc_h, ncc_w,
    )

    # --- Stage 4: Vectorized quality factors ---
    pce, ppe = _batch_qfactors(ncc_maps, sub_val, n_valid)

    # --- Assemble output grids ---
    u_grid = np.full((ny, nx), np.nan, dtype=np.float64)
    v_grid = np.full((ny, nx), np.nan, dtype=np.float64)
    cc_max = np.zeros((ny, nx), dtype=np.float64)
    qfactors = np.full((ny, nx, 2), np.inf, dtype=np.float64)

    if n_valid > 0:
        center = float(search)
        u_grid[valid_iy, valid_ix] = sub_x - center
        v_grid[valid_iy, valid_ix] = sub_y - center
        cc_max[valid_iy, valid_ix] = sub_val
        qfactors[valid_iy, valid_ix, 0] = pce
        qfactors[valid_iy, valid_ix, 1] = ppe

    return u_grid, v_grid, cc_max, qfactors


# ---------------------------------------------------------------------------
# matchTemplate backends
# ---------------------------------------------------------------------------


def _sequential_match(
    n_valid, valid_iy, valid_ix, cx_all, cy_all,
    f32, g32, half_w, search,
    ncc_maps, ipeak_x, ipeak_y, ipeak_val,
):
    """Sequential matchTemplate for small node counts."""
    for k in range(n_valid):
        _match_one(
            k, valid_iy, valid_ix, cx_all, cy_all,
            f32, g32, half_w, search,
            ncc_maps, ipeak_x, ipeak_y, ipeak_val,
        )


def _threaded_match(
    n_valid, valid_iy, valid_ix, cx_all, cy_all,
    f32, g32, half_w, search,
    ncc_maps, ipeak_x, ipeak_y, ipeak_val,
):
    """Threaded matchTemplate. OpenCV releases GIL → true parallelism."""
    n_workers = min(os.cpu_count() or 4, max(1, n_valid // 100))
    chunk = max(1, (n_valid + n_workers - 1) // n_workers)

    def _worker(start, end):
        for k in range(start, end):
            _match_one(
                k, valid_iy, valid_ix, cx_all, cy_all,
                f32, g32, half_w, search,
                ncc_maps, ipeak_x, ipeak_y, ipeak_val,
            )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for i in range(0, n_valid, chunk):
            futures.append(pool.submit(_worker, i, min(i + chunk, n_valid)))
        for fut in futures:
            fut.result()


def _match_one(
    k, valid_iy, valid_ix, cx_all, cy_all,
    f32, g32, half_w, search,
    ncc_maps, ipeak_x, ipeak_y, ipeak_val,
):
    """Compute matchTemplate + integer peak for a single node."""
    iy, ix = valid_iy[k], valid_ix[k]
    cx = cx_all[ix]
    cy = cy_all[iy]

    template = f32[cy - half_w: cy + half_w + 1,
                   cx - half_w: cx + half_w + 1]
    search_img = g32[cy - half_w - search: cy + half_w + 1 + search,
                     cx - half_w - search: cx + half_w + 1 + search]

    ncc_map = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    ncc_maps[k] = ncc_map

    _, max_val, _, max_loc = cv2.minMaxLoc(ncc_map)
    ipeak_x[k] = max_loc[0]  # column
    ipeak_y[k] = max_loc[1]  # row
    ipeak_val[k] = max_val


# ---------------------------------------------------------------------------
# Batch sub-pixel peak finding
# ---------------------------------------------------------------------------


def _batch_subpixel(
    ncc_maps: NDArray[np.float32],
    ipeak_x: NDArray[np.intp],
    ipeak_y: NDArray[np.intp],
    ipeak_val: NDArray[np.float64],
    ncc_h: int,
    ncc_w: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized sub-pixel peak finding for all nodes at once.

    Uses pre-computed pseudo-inverse of the 9-point quadratic design matrix
    instead of per-node ``np.linalg.lstsq``.
    """
    n = len(ipeak_x)
    sub_x = ipeak_x.astype(np.float64).copy()
    sub_y = ipeak_y.astype(np.float64).copy()
    sub_val = ipeak_val.copy()

    if n == 0:
        return sub_x, sub_y, sub_val

    # Nodes where peak is not on the edge (can refine)
    can_refine = (
        (ipeak_x > 0) & (ipeak_x < ncc_w - 1) &
        (ipeak_y > 0) & (ipeak_y < ncc_h - 1)
    )
    refine_idx = np.where(can_refine)[0]
    n_ref = len(refine_idx)

    if n_ref == 0:
        return sub_x, sub_y, sub_val

    # Extract 3x3 patches around integer peaks (vectorized fancy indexing)
    k_arr = refine_idx
    px_arr = ipeak_x[refine_idx]
    py_arr = ipeak_y[refine_idx]

    rows = py_arr[:, None] + _ROW_OFFSETS[None, :]  # (n_ref, 9)
    cols = px_arr[:, None] + _COL_OFFSETS[None, :]   # (n_ref, 9)
    k_exp = k_arr[:, None].repeat(9, axis=1)          # (n_ref, 9)

    patches = ncc_maps[k_exp, rows, cols].astype(np.float64)  # (n_ref, 9)

    # Batch least-squares: A = patches @ X_pinv.T  →  (n_ref, 6)
    A = patches @ _X_PINV.T

    # Sub-pixel offset from quadratic extremum
    denom = A[:, 3] ** 2 - 4.0 * A[:, 4] * A[:, 5]
    safe = np.abs(denom) > 1e-12
    denom_safe = np.where(safe, denom, 1.0)

    x_off = np.where(
        safe,
        (-A[:, 2] * A[:, 3] + 2.0 * A[:, 5] * A[:, 1]) / denom_safe,
        0.0,
    )
    y_off = np.where(
        safe,
        (-A[:, 3] * A[:, 1] + 2.0 * A[:, 4] * A[:, 2]) / denom_safe,
        0.0,
    )

    # Clamp: if offset exceeds +-1, fall back to integer peak
    ok = safe & (np.abs(x_off) <= 1.0) & (np.abs(y_off) <= 1.0)
    x_off = np.where(ok, np.round(x_off * 1000.0) / 1000.0, 0.0)
    y_off = np.where(ok, np.round(y_off * 1000.0) / 1000.0, 0.0)

    # Evaluate polynomial at sub-pixel extremum
    refined_val = (
        A[:, 0]
        + A[:, 1] * x_off
        + A[:, 2] * y_off
        + A[:, 3] * x_off * y_off
        + A[:, 4] * x_off ** 2
        + A[:, 5] * y_off ** 2
    )

    sub_x[refine_idx] = ipeak_x[refine_idx].astype(np.float64) + x_off
    sub_y[refine_idx] = ipeak_y[refine_idx].astype(np.float64) + y_off
    sub_val[refine_idx] = refined_val

    return sub_x, sub_y, sub_val


# ---------------------------------------------------------------------------
# Batch quality factors
# ---------------------------------------------------------------------------


def _batch_qfactors(
    ncc_maps: NDArray[np.float32],
    peak_vals: NDArray[np.float64],
    n_valid: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized quality factor computation.

    PCE (peak-to-correlation-energy) is fully vectorized.
    PPE (peak-to-entropy) uses a fast variance-based proxy to avoid
    per-node histogram computation.
    """
    if n_valid == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Reshape to (n_valid, ncc_h * ncc_w)
    flat = ncc_maps.reshape(n_valid, -1).astype(np.float64)
    ncc_min = flat.min(axis=1, keepdims=True)
    shifted = flat - ncc_min

    # PCE: peak^2 / mean(shifted^2)
    energy = np.mean(shifted ** 2, axis=1)
    pce = np.where(energy > 1e-20, peak_vals ** 2 / energy, np.inf)

    # PPE proxy: 1 / normalized_variance (higher = sharper peak)
    ncc_var = np.var(shifted, axis=1)
    ncc_max = shifted.max(axis=1)
    norm_var = np.where(ncc_max > 1e-20, ncc_var / (ncc_max ** 2), 0.0)
    ppe = np.where(norm_var > 1e-20, 1.0 / norm_var, np.inf)

    return pce, ppe


# ---------------------------------------------------------------------------
# Vectorized mask application
# ---------------------------------------------------------------------------


def _apply_mask_vectorized(
    u_grid: NDArray[np.float64],
    v_grid: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    mask: NDArray,
    h: int,
    w: int,
) -> None:
    """Apply reference mask to displacement grids (in-place)."""
    cx_int = np.round(x0).astype(np.intp)
    cy_int = np.round(y0).astype(np.intp)
    CX, CY = np.meshgrid(cx_int, cy_int)

    in_bounds = (CY >= 0) & (CY < h) & (CX >= 0) & (CX < w)
    # Use clipped indices for safe indexing (out-of-bounds already excluded)
    cy_safe = np.clip(CY, 0, h - 1)
    cx_safe = np.clip(CX, 0, w - 1)
    masked = in_bounds & (mask[cy_safe, cx_safe] < 0.5)

    u_grid[masked] = np.nan
    v_grid[masked] = np.nan


# ---------------------------------------------------------------------------
# Legacy per-node helpers (used by pyramid refinement)
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
    and searches in the deformed image at (x0 + u_init, y0 + v_init) +- search.

    Optimised with the same batch pattern as ``_batch_ncc_search``:
    integral-image template std, threaded ``matchTemplate``, and batch
    sub-pixel fitting — replacing the previous serial nested loop.

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

    # --- Vectorized validity check ---
    cx_all = np.round(x0).astype(np.intp)  # (nx,)
    cy_all = np.round(y0).astype(np.intp)  # (ny,)
    CX, CY = np.meshgrid(cx_all, cy_all)   # (ny, nx)

    # Template must fit in reference image
    template_ok = (
        (CY - half_w >= 0) & (CX - half_w >= 0)
        & (CY + half_w + 1 <= h) & (CX + half_w + 1 <= w)
    )

    # Exclude nodes with NaN initial displacement
    nan_init = np.isnan(u_init) | np.isnan(v_init)

    # Predicted search centre per node
    pred_CX = np.round(CX + np.where(nan_init, 0, u_init)).astype(np.intp)
    pred_CY = np.round(CY + np.where(nan_init, 0, v_init)).astype(np.intp)

    # Full search region must fit in deformed image
    search_ok = (
        (pred_CY - half_w - search >= 0)
        & (pred_CX - half_w - search >= 0)
        & (pred_CY + half_w + 1 + search <= h)
        & (pred_CX + half_w + 1 + search <= w)
    )

    # Template std via integral images — O(H*W) build, O(1) per node
    f32_sq = f32 ** 2
    int_f = cv2.integral(f32)      # (H+1, W+1)
    int_f2 = cv2.integral(f32_sq)

    T_y0 = CY - half_w
    T_y1 = CY + half_w + 1
    T_x0 = CX - half_w
    T_x1 = CX + half_w + 1
    n_pix = float((2 * half_w + 1) ** 2)

    pre_ok = template_ok & search_ok & ~nan_init
    ok_idx = np.where(pre_ok)
    var_f = np.zeros((ny, nx), dtype=np.float64)

    if len(ok_idx[0]) > 0:
        ty0 = T_y0[ok_idx]
        ty1 = T_y1[ok_idx]
        tx0 = T_x0[ok_idx]
        tx1 = T_x1[ok_idx]
        s_f = (int_f[ty1, tx1] - int_f[ty0, tx1]
               - int_f[ty1, tx0] + int_f[ty0, tx0])
        s_f2 = (int_f2[ty1, tx1] - int_f2[ty0, tx1]
                - int_f2[ty1, tx0] + int_f2[ty0, tx0])
        var_f[ok_idx] = np.maximum(s_f2 / n_pix - (s_f / n_pix) ** 2, 0.0)

    valid = pre_ok & (np.sqrt(var_f) > 1e-6)

    # Mark all invalid nodes as NaN
    invalid = ~valid
    u_out[invalid] = np.nan
    v_out[invalid] = np.nan

    # --- Threaded matchTemplate for valid nodes ---
    valid_iy, valid_ix = np.where(valid)
    n_valid = len(valid_iy)

    if n_valid == 0:
        return u_out, v_out, cc_out

    ncc_h = 2 * search + 1
    ncc_w = 2 * search + 1

    ncc_maps = np.zeros((n_valid, ncc_h, ncc_w), dtype=np.float32)
    ipeak_x = np.zeros(n_valid, dtype=np.intp)
    ipeak_y = np.zeros(n_valid, dtype=np.intp)
    ipeak_val = np.zeros(n_valid, dtype=np.float64)

    pred_cx_valid = pred_CX[valid_iy, valid_ix]
    pred_cy_valid = pred_CY[valid_iy, valid_ix]

    if n_valid >= 500:
        _threaded_match_refine(
            n_valid, valid_iy, valid_ix, cx_all, cy_all,
            pred_cx_valid, pred_cy_valid,
            f32, g32, half_w, search,
            ncc_maps, ipeak_x, ipeak_y, ipeak_val,
        )
    else:
        for k in range(n_valid):
            _match_one_refine(
                k, valid_iy, valid_ix, cx_all, cy_all,
                pred_cx_valid, pred_cy_valid,
                f32, g32, half_w, search,
                ncc_maps, ipeak_x, ipeak_y, ipeak_val,
            )

    # --- Batch sub-pixel peak finding (reuse existing vectorised path) ---
    sub_x, sub_y, sub_val = _batch_subpixel(
        ncc_maps, ipeak_x, ipeak_y, ipeak_val, ncc_h, ncc_w,
    )

    # --- Vectorized displacement conversion ---
    cx_valid = cx_all[valid_ix].astype(np.float64)
    cy_valid = cy_all[valid_iy].astype(np.float64)
    s_x0_valid = pred_cx_valid.astype(np.float64) - half_w - search
    s_y0_valid = pred_cy_valid.astype(np.float64) - half_w - search

    u_out[valid_iy, valid_ix] = s_x0_valid + sub_x + half_w - cx_valid
    v_out[valid_iy, valid_ix] = s_y0_valid + sub_y + half_w - cy_valid
    cc_out[valid_iy, valid_ix] = sub_val

    return u_out, v_out, cc_out


def _match_one_refine(
    k, valid_iy, valid_ix, cx_all, cy_all,
    pred_cx, pred_cy,
    f32, g32, half_w, search,
    ncc_maps, ipeak_x, ipeak_y, ipeak_val,
):
    """matchTemplate for a single node with offset search centre."""
    iy, ix = valid_iy[k], valid_ix[k]
    cx = cx_all[ix]
    cy = cy_all[iy]
    pcx = pred_cx[k]
    pcy = pred_cy[k]

    template = f32[cy - half_w: cy + half_w + 1,
                   cx - half_w: cx + half_w + 1]
    search_img = g32[pcy - half_w - search: pcy + half_w + 1 + search,
                     pcx - half_w - search: pcx + half_w + 1 + search]

    ncc_map = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    ncc_maps[k] = ncc_map

    _, max_val, _, max_loc = cv2.minMaxLoc(ncc_map)
    ipeak_x[k] = max_loc[0]
    ipeak_y[k] = max_loc[1]
    ipeak_val[k] = max_val


def _threaded_match_refine(
    n_valid, valid_iy, valid_ix, cx_all, cy_all,
    pred_cx, pred_cy,
    f32, g32, half_w, search,
    ncc_maps, ipeak_x, ipeak_y, ipeak_val,
):
    """Threaded matchTemplate for pyramid refinement nodes."""
    n_workers = min(os.cpu_count() or 4, max(1, n_valid // 100))
    chunk = max(1, (n_valid + n_workers - 1) // n_workers)

    def _worker(start, end):
        for k in range(start, end):
            _match_one_refine(
                k, valid_iy, valid_ix, cx_all, cy_all,
                pred_cx, pred_cy,
                f32, g32, half_w, search,
                ncc_maps, ipeak_x, ipeak_y, ipeak_val,
            )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for i in range(0, n_valid, chunk):
            futures.append(pool.submit(_worker, i, min(i + chunk, n_valid)))
        for fut in futures:
            fut.result()


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
    u_filled = u_pyr.copy()
    v_filled = v_pyr.copy()
    nan_mask = np.isnan(u_filled) | np.isnan(v_filled)
    if np.all(nan_mask):
        u_std = np.full((ny_std, nx_std), np.nan)
        v_std = np.full((ny_std, nx_std), np.nan)
        cc_std = np.zeros((ny_std, nx_std))
        return x0_std, y0_std, u_std, v_std, cc_std

    if np.any(nan_mask):
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
    patch = ncc[py - 1: py + 2, px - 1: px + 2].astype(np.float64).ravel()

    # Batch solve: A = patch @ X_pinv.T (uses pre-computed pseudo-inverse)
    A = _X_PINV @ patch  # (6,)

    # Extremum: dU/dx = A1 + A3*y + 2*A4*x = 0
    #           dU/dy = A2 + A3*x + 2*A5*y = 0
    denom = A[3] ** 2 - 4 * A[4] * A[5]
    if abs(denom) < 1e-12:
        return float(px), float(py), float(max_val)

    x_offset = (-A[2] * A[3] + 2 * A[5] * A[1]) / denom
    y_offset = (-A[3] * A[1] + 2 * A[4] * A[2]) / denom

    # If offset exceeds +-1, fall back to integer peak
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
