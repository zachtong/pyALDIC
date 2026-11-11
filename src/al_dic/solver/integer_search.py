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
# Chunked NCC search: configuration
# ---------------------------------------------------------------------------
#
# `_batch_ncc_search` allocates an `ncc_maps` array of shape
# (n_valid, 2*search+1, 2*search+1) float32. When `search` auto-expands
# (e.g. from 16 to 350 on a large-displacement scenario) this single
# array can balloon past 100 GB and OOM the host before any computation
# starts. The chunked path below caps the per-iteration ncc_maps to
# CHUNK_TARGET_GB; processing the same nodes in K chunks keeps peak
# memory at ~CHUNK_TARGET_GB regardless of search radius.

# Soft cap on the per-chunk ncc_maps allocation. 4 GB is comfortable
# even on 16 GB-RAM dev machines; bumping this reduces chunking
# overhead for cases that wouldn't OOM anyway.
CHUNK_TARGET_GB = 4.0

# Tests set this to a small integer to force chunking on inputs that
# would otherwise fit in a single batch — the only way to exercise the
# chunk-loop code path on synthetic small N. ``None`` = auto-size from
# CHUNK_TARGET_GB.
CHUNK_SIZE_OVERRIDE: int | None = None


def _decide_chunk_size(n_valid: int, ncc_h: int, ncc_w: int) -> int:
    """Pick a chunk size so each chunk's ncc_maps stays under the budget.

    Floor at 500 because below that the OpenCV ThreadPoolExecutor
    overhead dominates per-call wall time (matchTemplate amortizes well
    only for batches of a few hundred templates).
    """
    if CHUNK_SIZE_OVERRIDE is not None:
        return max(1, int(CHUNK_SIZE_OVERRIDE))
    bytes_per_node = ncc_h * ncc_w * 4   # float32
    budget_bytes = CHUNK_TARGET_GB * 1e9
    chunk = int(budget_bytes // bytes_per_node)
    chunk = max(500, chunk)
    return min(chunk, n_valid)


# ---------------------------------------------------------------------------
# Grid centering helper
# ---------------------------------------------------------------------------


def _centered_arange(lo: int, hi: int, step: int) -> NDArray[np.float64]:
    """Generate evenly-spaced grid points centered within [lo, hi].

    ``np.arange(lo, hi+1, step)`` starts at *lo* and leaves the
    remainder on the right, producing asymmetric ROI coverage.
    This function distributes the remainder evenly so that the gap
    between the grid and each boundary is approximately equal.

    Args:
        lo: Minimum valid coordinate (inclusive).
        hi: Maximum valid coordinate (inclusive).
        step: Grid spacing.

    Returns:
        1-D float64 array of grid coordinates.
    """
    span = hi - lo
    n_steps = span // step
    offset = (span - n_steps * step) // 2
    return np.arange(lo + offset, hi + 1, step, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def integer_search(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    para: DICPara,
    *,
    clamp_border: bool = True,
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
        clamp_border: Search nodes near the border with a window clamped to
            the image (default). False skips them as before, for images
            whose border is not real data (coarse pyramid levels).

    Returns:
        ``(x0, y0, u, v, info)`` where:
            - ``x0``: 1-D grid x-coordinates (M,).
            - ``y0``: 1-D grid y-coordinates (N,).
            - ``u``: x-displacement grid (N, M), float64 (sub-pixel).
            - ``v``: y-displacement grid (N, M), float64 (sub-pixel).
            - ``info``: Dict with ``'cc_max'`` (N, M) peak NCC values,
              ``'qfactors'`` (N, M, 2) quality factors [PCE, PPE].

    Memory characteristics (post v0.5.1 fixes):
        Peak RAM scales roughly as ``H*W * float64 * (3-4)`` plus
        ``CHUNK_TARGET_GB`` (currently 4 GB) for the per-chunk ncc_maps.
        On a 4096² image with ~64K nodes (step=16) the peak is ~12 GB.

        Memory-frugal patches across v0.5.x (in chronological order):
          1. ``_batch_qfactors`` no longer doubles ncc_maps to float64
             nor allocates ``shifted ** 2`` at full size — streaming
             einsum reductions on the float32 NCC view (v0.5).
          2. Pipeline caches the 6-DOF local_icgn precompute per
             (ref_idx, mesh_hash), so frames sharing a reference no
             longer rebuild a 3-GB array on top of the still-alive
             2-DOF subpb1 cache (v0.5).
          3. Pipeline shares one ``g_img`` buffer between the FFT
             and IC-GN call sites instead of making two identical
             copies per frame (v0.5).
          4. ``_batch_qfactors`` derives shifted statistics via the
             algebraic identity ``sum((X-min)²) = sum(X²) - 2·min·sum(X)
             + K·min²``, dropping even the single ``shifted`` buffer
             (v0.5.1).
          5. ``_batch_ncc_search`` processes nodes in chunks of
             ``CHUNK_TARGET_GB`` worth of ncc_maps at a time, capping
             peak allocation regardless of how large ``search``
             auto-expanded to. Pre-fix, search=300 with ~30K nodes
             tried to allocate ~43 GB in one go (v0.5.1).

        Cumulative effect on a 4096²×3 frame benchmark:
          v0.4.x baseline:  37 GB peak / 90s wall
          after v0.5:       17 GB peak / 95s wall
          after v0.5.1:     12 GB peak / 82s wall (chunking + streaming)

        And the 5472×3648 + search→350 case that OOMed even after
        v0.5 now runs in 16.6 GB.
    """
    h, w = f_img.shape
    roi = para.gridxy_roi_range
    winsize = para.winsize
    winstepsize = para.winstepsize
    search = para.size_of_fft_search_region
    half_w = winsize // 2

    # Generate grid points (node centers)
    # Mesh padding only requires `half_w` (full subset fits in image).
    # Nodes closer to the border than `half_w + search` are searched with
    # a window clamped to the image by `_batch_ncc_search`; only those whose
    # best match sits at the image edge are left NaN for `_inpaint_nans`.
    min_x = max(roi.gridx[0], half_w)
    max_x = min(roi.gridx[1], w - 1 - half_w)
    min_y = max(roi.gridy[0], half_w)
    max_y = min(roi.gridy[1], h - 1 - half_w)

    # Distinguish the two ways this goes empty: an unset/degenerate ROI (the
    # common scripting mistake -- gridxy_roi_range defaults to a zero-size
    # box) versus a winsize the image genuinely cannot fit. Blaming winsize
    # for an empty ROI sends callers chasing the wrong parameter.
    roi_span_x = roi.gridx[1] - roi.gridx[0]
    roi_span_y = roi.gridy[1] - roi.gridy[0]
    if roi_span_x <= 0 or roi_span_y <= 0:
        reason = (
            f"ROI is empty: gridxy_roi_range spans {roi_span_x}x{roi_span_y} px "
            f"(gridx={roi.gridx}, gridy={roi.gridy}). Set "
            f"para.gridxy_roi_range to the region to correlate -- it defaults "
            f"to a zero-size box, so it must be set explicitly when calling "
            f"run_aldic() directly."
        )
    else:
        reason = (
            f"winsize ({winsize}) is too large for the ROI "
            f"({roi_span_x}x{roi_span_y} px) within image ({h}x{w}); each node "
            f"needs a {winsize}x{winsize} subset inside the image."
        )

    if min_x >= max_x or min_y >= max_y:
        logger.warning("Cannot generate any mesh nodes. %s", reason)

    x0 = _centered_arange(min_x, max_x, winstepsize)
    y0 = _centered_arange(min_y, max_y, winstepsize)

    if len(x0) == 0 or len(y0) == 0:
        raise ValueError(
            f"No grid points generated. {reason} "
            f"(image {h}x{w}, winsize={winsize}, winstepsize={winstepsize}, "
            f"search={search})"
        )

    ny, nx = len(y0), len(x0)

    # Pre-compute template coverage: skip NCC entirely for nodes whose
    # template window has low mask coverage.  These boundary nodes produce
    # false NCC peaks (edge feature dominates speckle texture).  Skipping
    # them lets init_disp fill from reliable neighbours instead.
    skip_mask = None
    n_skipped = 0
    if para.img_ref_mask is not None:
        coverage = _compute_grid_coverage(x0, y0, para.img_ref_mask, half_w)
        skip_mask = coverage < 0.75
        n_skipped = int(np.sum(skip_mask))
        if n_skipped > 0:
            logger.info(
                "Coverage pre-filter: skipping NCC for %d/%d nodes "
                "(coverage < 75%%)",
                n_skipped, ny * nx,
            )

    # Vectorized NCC search (low-coverage nodes excluded via skip_mask)
    u_grid, v_grid, cc_max, qfactors = _batch_ncc_search(
        f_img, g_img, x0, y0, half_w, search, skip_mask=skip_mask,
        clamp_border=clamp_border,
    )

    # Mask application: NaN for nodes whose center is outside the mask
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

    # --- Reliability filtering ---
    # NaN unreliable nodes so init_disp only interpolates from "seeds"
    # that truly have accurate NCC results.
    n_rejected = 0

    # (a) Clipped peaks: NCC peak at search boundary → likely wrong
    if peak_clipped:
        u_grid[clipped_nodes] = np.nan
        v_grid[clipped_nodes] = np.nan
        n_rejected += n_clipped
        logger.warning(
            "FFT search: %d/%d nodes have peaks at search boundary "
            "(max |disp|=%.1f, search=%d) -> NaN'd for interpolation.",
            n_clipped, n_valid, max_abs_disp, search,
        )

    # (b) Low NCC quality: weak/ambiguous correlation peak.
    # PCE (Peak Correlation Energy) < threshold → NCC landscape is
    # flat or has multiple competing peaks → unreliable.
    finite_after_clip = np.isfinite(u_grid) & np.isfinite(v_grid)
    pce_vals = qfactors[:, :, 0]
    finite_pce = pce_vals[finite_after_clip]
    if len(finite_pce) > 0:
        # Adaptive threshold: nodes significantly below median PCE
        pce_median = np.median(finite_pce)
        pce_thresh = max(pce_median * 0.1, 1e-3)
        low_quality = finite_after_clip & (pce_vals < pce_thresh)
        n_low_q = int(np.sum(low_quality))
        if n_low_q > 0:
            u_grid[low_quality] = np.nan
            v_grid[low_quality] = np.nan
            n_rejected += n_low_q
            logger.info(
                "Quality filter: %d nodes with PCE < %.3f (median=%.3f) "
                "-> NaN'd",
                n_low_q, pce_thresh, pce_median,
            )

    n_seeds = int(np.sum(np.isfinite(u_grid)))
    logger.info(
        "Integer search complete: %d seeds / %d total (%dx%d grid), "
        "%d rejected (coverage=%d, clipped=%d, quality=%d), "
        "mean cc=%.3f",
        n_seeds, ny * nx, ny, nx,
        n_skipped + n_rejected, n_skipped, n_clipped,
        n_rejected - n_clipped,
        np.nanmean(cc_max),
    )

    info = dict(
        cc_max=cc_max,
        qfactors=qfactors,
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

    # Downsample mask through pyramid levels so coverage filtering
    # applies at every level (not just finest).  Without this, coarse
    # levels match bubble/hole edges instead of speckle texture.
    mask_pyr: list[NDArray[np.float64] | None] = [para.img_ref_mask]
    if para.img_ref_mask is not None:
        for _ in range(n_levels - 1):
            prev = mask_pyr[-1]
            down = cv2.pyrDown(prev.astype(np.float32)).astype(np.float64)
            # Re-binarize: treat anything < 0.5 as outside mask
            mask_pyr.append((down >= 0.5).astype(np.float64))
    else:
        mask_pyr = [None] * n_levels

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
        img_ref_mask=mask_pyr[-1],
    )

    # cv2.pyrDown reflects the border, so a coarse level holds artificial
    # content near its edges: border nodes are searched only at full
    # resolution (clamp_border=False on the coarse levels).
    x0_c, y0_c, u_c, v_c, info_c = integer_search(
        f_pyr[-1], g_pyr[-1], coarse_para, clamp_border=False,
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
            clamp_border=False,
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
        _apply_template_coverage_filter(
            u_std, v_std, x0_std, y0_std, para.img_ref_mask, half_w,
        )

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
    skip_mask: NDArray[np.bool_] | None = None,
    clamp_border: bool = True,
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

    # Full search box inside the image: the batch path below. A node whose
    # box crosses the border (but whose template fits) is searched with a
    # clamped window instead of being skipped (see _clamped_window_search).
    boundary_ok = (
        (CY - half_w - search >= 0) &
        (CX - half_w - search >= 0) &
        (CY + half_w + 1 + search <= h) &
        (CX + half_w + 1 + search <= w)
    )
    template_ok = (
        (CY - half_w >= 0) & (CX - half_w >= 0)
        & (CY + half_w + 1 <= h) & (CX + half_w + 1 <= w)
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

    # Compute variance for every node whose template fits
    ok_mask = boundary_ok
    ok_idx = np.where(template_ok)
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

    textured = np.sqrt(var_f) > 1e-6
    if skip_mask is not None:
        textured = textured & ~skip_mask
    valid = ok_mask & textured
    clamped = template_ok & ~ok_mask & textured & clamp_border

    # --- Stage 2: matchTemplate for valid nodes (chunked) ---
    valid_iy, valid_ix = np.where(valid)
    n_valid = len(valid_iy)

    ncc_h = 2 * search + 1
    ncc_w = 2 * search + 1

    # Pre-allocate FULL-N result arrays (cheap — only ipeak indices and
    # final per-node aggregates, not the per-node NCC maps).
    sub_x_all = np.empty(n_valid, dtype=np.float64)
    sub_y_all = np.empty(n_valid, dtype=np.float64)
    sub_val_all = np.empty(n_valid, dtype=np.float64)
    pce_all = np.empty(n_valid, dtype=np.float64)
    ppe_all = np.empty(n_valid, dtype=np.float64)

    if n_valid > 0:
        # Pick chunk size: keep ncc_maps_chunk under MEMORY_TARGET_GB so
        # the peak allocation in this stage is bounded regardless of how
        # large `search` got from auto-expand. Tests can pin via the
        # CHUNK_SIZE_OVERRIDE module attribute below.
        chunk_size = _decide_chunk_size(n_valid, ncc_h, ncc_w)

        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            n_chunk = end - start

            # Per-chunk allocations — released at end of iteration.
            ncc_maps_c = np.zeros(
                (n_chunk, ncc_h, ncc_w), dtype=np.float32,
            )
            ipeak_x_c = np.zeros(n_chunk, dtype=np.intp)
            ipeak_y_c = np.zeros(n_chunk, dtype=np.intp)
            ipeak_val_c = np.zeros(n_chunk, dtype=np.float64)

            # Slice the global node-index arrays so _match_one's
            # valid_iy[k]/valid_ix[k] lookups index chunk-relative k.
            iy_c = valid_iy[start:end]
            ix_c = valid_ix[start:end]

            if n_chunk >= 500:
                _threaded_match(
                    n_chunk, iy_c, ix_c, cx_all, cy_all,
                    f32, g32, half_w, search,
                    ncc_maps_c, ipeak_x_c, ipeak_y_c, ipeak_val_c,
                )
            else:
                _sequential_match(
                    n_chunk, iy_c, ix_c, cx_all, cy_all,
                    f32, g32, half_w, search,
                    ncc_maps_c, ipeak_x_c, ipeak_y_c, ipeak_val_c,
                )

            sx, sy, sv = _batch_subpixel(
                ncc_maps_c, ipeak_x_c, ipeak_y_c, ipeak_val_c, ncc_h, ncc_w,
            )
            pc, pp = _batch_qfactors(ncc_maps_c, sv, n_chunk)

            sub_x_all[start:end] = sx
            sub_y_all[start:end] = sy
            sub_val_all[start:end] = sv
            pce_all[start:end] = pc
            ppe_all[start:end] = pp
            # Drop big allocations explicitly so the next iteration starts
            # at low memory pressure (Python ref-counts will free at scope
            # exit anyway, but explicit del helps when sibling temporaries
            # in the loop body live longer than expected).
            del ncc_maps_c, ipeak_x_c, ipeak_y_c, ipeak_val_c

    # --- Assemble output grids ---
    u_grid = np.full((ny, nx), np.nan, dtype=np.float64)
    v_grid = np.full((ny, nx), np.nan, dtype=np.float64)
    cc_max = np.zeros((ny, nx), dtype=np.float64)
    qfactors = np.full((ny, nx, 2), np.inf, dtype=np.float64)

    if n_valid > 0:
        center = float(search)
        u_grid[valid_iy, valid_ix] = sub_x_all - center
        v_grid[valid_iy, valid_ix] = sub_y_all - center
        cc_max[valid_iy, valid_ix] = sub_val_all
        qfactors[valid_iy, valid_ix, 0] = pce_all
        qfactors[valid_iy, valid_ix, 1] = ppe_all

    # --- Border nodes: clamped search window ---
    c_iy, c_ix = np.where(clamped)
    if len(c_iy) > 0:
        tcx, tcy = cx_all[c_ix], cy_all[c_iy]
        px, py, pv, pc, pp = _clamped_window_search(
            f32, g32, tcx, tcy, tcx, tcy, half_w, search,
        )
        found = np.isfinite(px)
        iy, ix = c_iy[found], c_ix[found]
        u_grid[iy, ix] = px[found] - float(search)
        v_grid[iy, ix] = py[found] - float(search)
        cc_max[iy, ix] = pv[found]
        qfactors[iy, ix, 0] = pc[found]
        qfactors[iy, ix, 1] = pp[found]
        logger.info(
            "NCC search: %d node(s) near the image border searched with a "
            "clamped window; %d of them gave no reliable match and are left "
            "for the fill.",
            len(c_iy), int(np.sum(~found)),
        )

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


# Acceptance tests for a peak found in a clamped window (see
# _clamped_window_search). On synthetic shifts with the true match outside
# the image for some nodes, reverse consistency alone kept ~20% of the
# spurious peaks and a primary/secondary peak ratio alone kept even more;
# together (1 px, 1.3) they kept 1-3 of 100-178 spurious peaks while keeping
# 99.5-100% of the true ones. 1.3 sits in the usual PIV range of 1.2-1.5.
_REVERSE_TOL_PX = 1
_MIN_PEAK_RATIO = 1.3


def _reverse_consistent(
    f32: NDArray[np.float32],
    g32: NDArray[np.float32],
    tcx: int,
    tcy: int,
    mcx: int,
    mcy: int,
    half_w: int,
    search: int,
) -> bool:
    """True if the deformed subset centred at (mcx, mcy), searched back in
    the reference within ``search`` of (tcx, tcy), lands on (tcx, tcy).

    A spurious match is a piece of the deformed image whose real origin is
    elsewhere, so the reverse search usually finds that origin instead.
    """
    h, w = f32.shape
    template = g32[mcy - half_w: mcy + half_w + 1, mcx - half_w: mcx + half_w + 1]
    x0, x1 = max(0, tcx - half_w - search), min(w, tcx + half_w + 1 + search)
    y0, y1 = max(0, tcy - half_w - search), min(h, tcy + half_w + 1 + search)
    ncc = cv2.matchTemplate(f32[y0:y1, x0:x1], template, cv2.TM_CCOEFF_NORMED)
    _, _, _, (px, py) = cv2.minMaxLoc(ncc)
    return (abs(x0 + px + half_w - tcx) <= _REVERSE_TOL_PX
            and abs(y0 + py + half_w - tcy) <= _REVERSE_TOL_PX)


def _clamped_window_search(
    f32: NDArray[np.float32],
    g32: NDArray[np.float32],
    tcx: NDArray[np.intp],
    tcy: NDArray[np.intp],
    scx: NDArray[np.intp],
    scy: NDArray[np.intp],
    half_w: int,
    search: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """NCC search for nodes whose search box crosses the image border.

    The box ``search centre +- (half_w + search)`` is clamped to the deformed
    image, so the node still tries every displacement whose subset lies in
    the image (an asymmetric window), as ``seed_propagation`` does for a
    Starting Point. Templates (centred at ``tcx, tcy``) must fit in the
    reference image; search centres are ``scx, scy``.

    Returns per node ``(peak_x, peak_y, peak_val, pce, ppe)``. The peak is
    sub-pixel and in FULL-map coordinates, the frame of the unclamped search
    (index ``search`` = zero offset from the search centre). PCE/PPE come
    from the node's own map with the usual formulas.

    A clamped window may have lost the true match (it lies outside the image)
    and then holds only a spurious peak, so a peak is kept only if it passes
    three tests; otherwise it is NaN and the node is filled as before:

    - along an axis whose window was clamped, it is more than half a pixel
      from either end of the map. This includes the far, unclamped end: a
      node whose true match left the image often puts a spurious peak
      there, so an end peak on such an axis is unreliable and must not be
      reported as clipped either, or the search would widen for nothing.
      Along an unclamped axis a radius-edge peak is left to the caller's
      usual clipped test;
    - searching the matched deformed subset back in the reference, with the
      same radius, returns to the template within ``_REVERSE_TOL_PX``;
    - the peak is at least ``_MIN_PEAK_RATIO`` times the best value farther
      than ``half_w`` from it on the same map.

    The peak is also NaN when the window is smaller than the template or the
    best correlation is not positive.
    """
    h, w = g32.shape
    n = len(tcx)
    peak_x = np.full(n, np.nan)
    peak_y = np.full(n, np.nan)
    peak_val = np.zeros(n)
    pce = np.full(n, np.inf)
    ppe = np.full(n, np.inf)
    tpl = 2 * half_w + 1

    def _one(k: int) -> None:
        asked_x0 = int(scx[k]) - half_w - search
        asked_y0 = int(scy[k]) - half_w - search
        asked_x1 = int(scx[k]) + half_w + 1 + search
        asked_y1 = int(scy[k]) + half_w + 1 + search
        x0, x1 = max(0, asked_x0), min(w, asked_x1)
        y0, y1 = max(0, asked_y0), min(h, asked_y1)
        if x1 - x0 < tpl or y1 - y0 < tpl:
            return
        template = f32[tcy[k] - half_w: tcy[k] + half_w + 1,
                       tcx[k] - half_w: tcx[k] + half_w + 1]
        ncc = cv2.matchTemplate(g32[y0:y1, x0:x1], template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, (px, py) = cv2.minMaxLoc(ncc)
        if max_val <= 0.0:  # no positive correlation anywhere: nothing to keep
            return
        nh, nw = ncc.shape
        sx, sy, sv = _batch_subpixel(
            ncc[None], np.array([px], dtype=np.intp), np.array([py], dtype=np.intp),
            np.array([max_val], dtype=np.float64), nh, nw,
        )
        x_clamped = x0 > asked_x0 or x1 < asked_x1
        y_clamped = y0 > asked_y0 or y1 < asked_y1
        x_end = sx[0] <= 0.5 or sx[0] >= nw - 1.5
        y_end = sy[0] <= 0.5 or sy[0] >= nh - 1.5
        if (x_clamped and x_end) or (y_clamped and y_end):
            return
        # A radius-edge peak on an open axis is handled as in a full window:
        # the caller reports it as clipped (wider search) and drops it.
        if not (x_end or y_end):
            if not _reverse_consistent(f32, g32, tcx[k], tcy[k], x0 + px + half_w,
                                       y0 + py + half_w, half_w, search):
                return
            far = np.ones_like(ncc, dtype=bool)
            far[max(0, py - half_w): py + half_w + 1,
                max(0, px - half_w): px + half_w + 1] = False
            if far.any() and max_val < _MIN_PEAK_RATIO * float(ncc[far].max()):
                return
        pc, pp = _batch_qfactors(ncc[None], sv, 1)
        peak_x[k] = sx[0] + (x0 - asked_x0)
        peak_y[k] = sy[0] + (y0 - asked_y0)
        peak_val[k] = sv[0]
        pce[k] = pc[0]
        ppe[k] = pp[0]

    if n >= 500:
        n_workers = min(os.cpu_count() or 4, max(1, n // 100))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_one, range(n)))
    else:
        for k in range(n):
            _one(k)
    return peak_x, peak_y, peak_val, pce, ppe


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

    # Memory-frugal streaming rewrite (was 4× ncc_maps in float64 in
    # v0.4; v0.5 brought it to 1× via a `shifted` buffer; v0.5.x drops
    # even that buffer using the algebraic identity:
    #   shifted = X - min(X)
    #   sum(shifted)    = sum(X) - K * min
    #   sum(shifted^2)  = sum(X^2) - 2 * min * sum(X) + K * min^2
    # so all per-row statistics can be derived from sum(X), sum(X^2),
    # min(X), max(X) — four reductions over the float32 NCC view, no
    # full-size intermediate. At 4K² + auto-expanded search this saves
    # the entire ncc_maps duplicate (~10-100 GB for large displacement
    # scenarios). Numerics: float32 accumulation in einsum can lose ~1e-3
    # relative precision on rows with ~5e5 elements; we promote to
    # float64 BEFORE the final per-row arithmetic to bound the error.
    flat32 = ncc_maps.reshape(n_valid, -1)              # (n_valid, K) float32 view
    K = flat32.shape[1]
    inv_K = 1.0 / K

    # Four streaming reductions over flat32 (no full-size intermediate)
    ncc_min = flat32.min(axis=1)                                       # (n_valid,)
    ncc_max = flat32.max(axis=1)                                       # (n_valid,)
    sum_orig = flat32.sum(axis=1)                                      # (n_valid,)
    sum_orig_sq = np.einsum("ij,ij->i", flat32, flat32, optimize=True)  # (n_valid,)

    # Promote per-row aggregates to float64 BEFORE applying the shift
    # identity. min/max/sum on float32 inputs yield float32; sum-of-
    # squares of large positive values especially needs float64 here
    # because the shift formula has a cancellation term (sum(X²) and
    # 2*min*sum(X) can be of similar magnitude).
    ncc_min64 = ncc_min.astype(np.float64)
    ncc_max64 = ncc_max.astype(np.float64)
    sum_orig64 = sum_orig.astype(np.float64)
    sum_orig_sq64 = sum_orig_sq.astype(np.float64)
    peak64 = peak_vals.astype(np.float64, copy=False)

    # Apply the identity: derive the same statistics that would have
    # come from `shifted = flat32 - ncc_min[:, None]` without ever
    # materializing it.
    sum_x = sum_orig64 - ncc_min64 * K
    sum_x2 = sum_orig_sq64 - 2.0 * ncc_min64 * sum_orig64 + K * ncc_min64 * ncc_min64
    # Numerical floor — sum_x2 should be >= 0 mathematically; tiny
    # negative values can leak in from cancellation. Clamp.
    sum_x2 = np.maximum(sum_x2, 0.0)

    mean_x = sum_x * inv_K
    energy = sum_x2 * inv_K
    var_x = np.maximum(energy - mean_x * mean_x, 0.0)
    ncc_max_row = ncc_max64 - ncc_min64                                # max(shifted)

    pce = np.where(energy > 1e-20, peak64 * peak64 / energy, np.inf)
    norm_var = np.where(
        ncc_max_row > 1e-20, var_x / (ncc_max_row * ncc_max_row), 0.0,
    )
    ppe = np.where(norm_var > 1e-20, 1.0 / norm_var, np.inf)

    return pce, ppe


# ---------------------------------------------------------------------------
# Vectorized mask application
# ---------------------------------------------------------------------------


def _compute_grid_coverage(
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    mask: NDArray,
    half_w: int,
) -> NDArray[np.float64]:
    """Compute template coverage fraction for each grid node.

    Returns (ny, nx) array of coverage fractions in [0, 1].
    Uses the same integral-image approach as the other coverage functions.
    """
    h, w = mask.shape
    mask_f = (mask > 0.5).astype(np.float64)
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(mask_f, axis=0, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    win = 2 * half_w + 1
    n_pix = win * win

    cx_int = np.round(x0).astype(np.intp)
    cy_int = np.round(y0).astype(np.intp)
    CX_g, CY_g = np.meshgrid(cx_int, cy_int)

    y1 = np.clip(CY_g - half_w, 0, h)
    y2 = np.clip(CY_g + half_w + 1, 0, h)
    x1 = np.clip(CX_g - half_w, 0, w)
    x2 = np.clip(CX_g + half_w + 1, 0, w)

    s = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return s / n_pix


def _apply_template_coverage_filter(
    u_grid: NDArray[np.float64],
    v_grid: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    mask: NDArray,
    half_w: int,
    min_coverage: float = 0.75,
) -> None:
    """Reject nodes whose template window has low mask coverage (in-place).

    Nodes near mask boundaries (e.g. bubble edges) produce false NCC peaks
    because the gray/speckle step edge dominates over actual speckle texture.
    This filter computes the fraction of template pixels inside the mask and
    sets u/v to NaN when coverage < *min_coverage*.

    Uses an integral-image approach: O(H*W) precompute, O(1) per node.
    """
    h, w = mask.shape
    # Integral image of the binary mask (float64 for precision)
    mask_f = (mask > 0.5).astype(np.float64)
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(mask_f, axis=0, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    # Template window size
    win = 2 * half_w + 1
    n_pix = win * win

    cx_int = np.round(x0).astype(np.intp)
    cy_int = np.round(y0).astype(np.intp)
    CX_g, CY_g = np.meshgrid(cx_int, cy_int)

    # Clamp template bounds to image
    y1 = np.clip(CY_g - half_w, 0, h)
    y2 = np.clip(CY_g + half_w + 1, 0, h)
    x1 = np.clip(CX_g - half_w, 0, w)
    x2 = np.clip(CX_g + half_w + 1, 0, w)

    # Sum of mask pixels in each template window via integral image
    s = (integral[y2, x2] - integral[y1, x2]
         - integral[y2, x1] + integral[y1, x1])
    coverage = s / n_pix

    low = coverage < min_coverage
    u_grid[low] = np.nan
    v_grid[low] = np.nan


def compute_template_coverage(
    coordinates: NDArray[np.float64],
    mask: NDArray,
    half_w: int,
) -> NDArray[np.float64]:
    """Compute fraction of template window inside mask for each node.

    Works with arbitrary (n, 2) coordinate arrays [x, y],
    unlike ``_apply_template_coverage_filter`` which requires regular grids.

    Uses an integral-image approach: O(H*W) precompute, O(1) per node.

    Args:
        coordinates: (n, 2) array of node coordinates [x, y].
        mask: (H, W) binary mask (1 = inside ROI).
        half_w: Half-window size (template is 2*half_w+1 square).

    Returns:
        (n,) array of coverage fractions in [0, 1].
    """
    h, w = mask.shape
    mask_f = (mask > 0.5).astype(np.float64)
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(mask_f, axis=0, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    win = 2 * half_w + 1
    n_pix = win * win

    cx = np.clip(np.round(coordinates[:, 0]).astype(np.intp), 0, w - 1)
    cy = np.clip(np.round(coordinates[:, 1]).astype(np.intp), 0, h - 1)

    y1 = np.clip(cy - half_w, 0, h)
    y2 = np.clip(cy + half_w + 1, 0, h)
    x1 = np.clip(cx - half_w, 0, w)
    x2 = np.clip(cx + half_w + 1, 0, w)

    s = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return s / n_pix


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
    clamp_border: bool = True,
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
        clamp_border: Search border nodes with a clamped window (default);
            False drops them, for levels whose border is not real data.

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

    # Full search region inside the deformed image: the batch path below.
    # A node whose region crosses the border is searched with a clamped
    # window instead of being dropped (see _clamped_window_search).
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
    ok_idx = np.where(template_ok & ~nan_init)
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

    textured = np.sqrt(var_f) > 1e-6
    valid = pre_ok & textured
    # A border node is searched only if its predicted match still lies in
    # the deformed image; otherwise it cannot be measured at this level.
    pred_fits = (
        (pred_CY - half_w >= 0) & (pred_CX - half_w >= 0)
        & (pred_CY + half_w + 1 <= h) & (pred_CX + half_w + 1 <= w)
    )
    clamped = template_ok & ~search_ok & ~nan_init & textured & pred_fits & clamp_border

    # Mark all invalid nodes as NaN
    invalid = ~valid
    u_out[invalid] = np.nan
    v_out[invalid] = np.nan

    # --- Border nodes: clamped search window around the prediction ---
    c_iy, c_ix = np.where(clamped)
    if len(c_iy) > 0:
        tcx, tcy = cx_all[c_ix], cy_all[c_iy]
        pcx, pcy = pred_CX[c_iy, c_ix], pred_CY[c_iy, c_ix]
        px, py, pv, _, _ = _clamped_window_search(
            f32, g32, tcx, tcy, pcx, pcy, half_w, search,
        )
        found = np.isfinite(px)
        iy, ix = c_iy[found], c_ix[found]
        # Same conversion as the batch path: window origin + peak - template.
        u_out[iy, ix] = (pcx[found] - half_w - search) + px[found] + half_w - tcx[found]
        v_out[iy, ix] = (pcy[found] - half_w - search) + py[found] + half_w - tcy[found]
        cc_out[iy, ix] = pv[found]

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

    # Standard grid (same as direct integer_search would produce).
    # Mesh padding is `half_w` so partial-edge IC-GN nodes can be created;
    # NCC for those edge nodes is left as NaN and inpainted downstream.
    min_x = max(roi.gridx[0], half_w)
    max_x = min(roi.gridx[1], img_w - 1 - half_w)
    min_y = max(roi.gridy[0], half_w)
    max_y = min(roi.gridy[1], img_h - 1 - half_w)

    x0_std = _centered_arange(min_x, max_x, winstepsize)
    y0_std = _centered_arange(min_y, max_y, winstepsize)

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
