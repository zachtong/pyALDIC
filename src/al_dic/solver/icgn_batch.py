"""Batch-vectorized IC-GN solvers for all nodes simultaneously.

Replaces the per-node Python loop with fully vectorized NumPy operations:
- Single map_coordinates call per iteration (instead of N calls)
- Batched ZNSSD computation across all active nodes
- Batched np.linalg.solve for Hessian systems
- Vectorized compose_warp

Used as default backend when Numba is not available, and for
pre-computation even when Numba handles the iteration loop.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-computation (shared by batch and Numba backends)
# ---------------------------------------------------------------------------

def precompute_subsets_6dof(
    coords: NDArray[np.float64],
    img_ref: NDArray[np.float64],
    df_dx: NDArray[np.float64],
    df_dy: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    winsize: int,
) -> dict:
    """Pre-extract reference subsets, masks, gradients, Hessians for all nodes.

    Uses Numba prange for N >= 50 nodes, falls back to Python loop otherwise.

    Returns dict with arrays of shape (N, Sy, Sx) or (N, 6, 6) etc.
    """
    N = coords.shape[0]
    half_w = winsize // 2
    Sy = Sx = winsize + 1
    h, w = img_ref.shape

    # --- Try Numba parallel backend ---
    if N >= 50:
        try:
            from .numba_kernels import precompute_subsets_6dof_numba, HAS_NUMBA
            if HAS_NUMBA:
                (ref_all, gx_all, gy_all, mask_all,
                 XX_all, YY_all, H_all, meanf_all, bottomf_all,
                 valid, mark_hole) = precompute_subsets_6dof_numba(
                    coords, img_ref, df_dx, df_dy, img_ref_mask,
                    half_w, Sy, Sx,
                )
                return {
                    "ref_all": ref_all, "gx_all": gx_all, "gy_all": gy_all,
                    "mask_all": mask_all, "XX_all": XX_all, "YY_all": YY_all,
                    "H_all": H_all, "meanf_all": meanf_all,
                    "bottomf_all": bottomf_all,
                    "valid": valid, "mark_hole": mark_hole,
                    "Sy": Sy, "Sx": Sx, "img_h": h, "img_w": w,
                }
        except Exception:
            logger.warning(
                "Numba precompute_6dof failed, using Python fallback.",
                exc_info=True,
            )

    # --- Fallback: Python loop ---
    return _precompute_subsets_6dof_python(
        coords, img_ref, df_dx, df_dy, img_ref_mask, winsize,
    )


def _precompute_subsets_6dof_python(
    coords, img_ref, df_dx, df_dy, img_ref_mask, winsize,
):
    """Python-loop fallback for precompute_subsets_6dof."""
    N = coords.shape[0]
    half_w = winsize // 2
    Sy = Sx = winsize + 1
    h, w = img_ref.shape

    ref_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    gx_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    gy_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    mask_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    XX_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    YY_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    H_all = np.zeros((N, 6, 6), dtype=np.float64)
    meanf_all = np.zeros(N, dtype=np.float64)
    bottomf_all = np.ones(N, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)
    mark_hole = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        x0, y0 = coords[i]
        x_lo = int(round(x0) - half_w)
        x_hi = int(round(x0) + half_w)
        y_lo = int(round(y0) - half_w)
        y_hi = int(round(y0) + half_w)

        if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
            mark_hole[i] = True
            continue

        mask_patch = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ref_patch_raw = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1]

        # Connected component containing subset center (mask-based validity)
        bw = _connected_center_mask(mask_patch > 0.5)
        n_connected = int(np.sum(bw > 0.5))
        if n_connected < int(0.5 * mask_patch.size):
            mark_hole[i] = True
            continue

        # Apply connected component mask to raw pixels and gradients
        ref_sub = ref_patch_raw * bw
        gx_sub = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw
        gy_sub = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw

        xx_rel = np.arange(x_lo, x_hi + 1, dtype=np.float64) - round(x0)
        yy_rel = np.arange(y_lo, y_hi + 1, dtype=np.float64) - round(y0)
        XX = np.broadcast_to(xx_rel[np.newaxis, :], (Sy, Sx)).copy()
        YY = np.broadcast_to(yy_rel[:, np.newaxis], (Sy, Sx)).copy()

        H = _build_hessian_6dof(XX, YY, gx_sub, gy_sub)

        # Hessian conditioning check — reject if ill-conditioned (thin strips)
        try:
            np.linalg.cholesky(H + 1e-12 * np.eye(6))
        except np.linalg.LinAlgError:
            mark_hole[i] = True
            continue

        # Statistics from mask-valid pixels (not pixel-value proxy)
        valid_px = bw > 0.5
        n_valid = int(np.sum(valid_px))
        if n_valid < 4:
            continue

        meanf = np.mean(ref_sub[valid_px])
        varf = np.var(ref_sub[valid_px])
        bottomf = np.sqrt(max((n_valid - 1) * varf, 1e-30))

        ref_all[i] = ref_sub
        gx_all[i] = gx_sub
        gy_all[i] = gy_sub
        mask_all[i] = bw
        XX_all[i] = XX
        YY_all[i] = YY
        H_all[i] = H
        meanf_all[i] = meanf
        bottomf_all[i] = bottomf
        valid[i] = True

    return {
        "ref_all": ref_all, "gx_all": gx_all, "gy_all": gy_all,
        "mask_all": mask_all, "XX_all": XX_all, "YY_all": YY_all,
        "H_all": H_all, "meanf_all": meanf_all, "bottomf_all": bottomf_all,
        "valid": valid, "mark_hole": mark_hole,
        "Sy": Sy, "Sx": Sx, "img_h": h, "img_w": w,
    }


def precompute_subsets_2dof(
    coords: NDArray[np.float64],
    img_ref: NDArray[np.float64],
    df_dx: NDArray[np.float64],
    df_dy: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    winsize_x_arr: NDArray[np.int64],
    winsize_y_arr: NDArray[np.int64],
) -> dict:
    """Pre-extract subsets for 2-DOF IC-GN (subpb1).

    Uses Numba prange for N >= 50 nodes, falls back to Python loop otherwise.
    Handles per-node winsize by using the max winsize and zero-padding smaller subsets.
    """
    N = coords.shape[0]
    max_wx = int(winsize_x_arr.max())
    max_wy = int(winsize_y_arr.max())
    Sx = max_wx + 1
    Sy = max_wy + 1
    h, w = img_ref.shape

    # --- Try Numba parallel backend ---
    if N >= 50:
        try:
            from .numba_kernels import precompute_subsets_2dof_numba, HAS_NUMBA
            if HAS_NUMBA:
                (ref_all, gx_all, gy_all, mask_all,
                 XX_all, YY_all, H2_img_all,
                 meanf_all, bottomf_all, valid) = precompute_subsets_2dof_numba(
                    coords, img_ref, df_dx, df_dy, img_ref_mask,
                    winsize_x_arr, winsize_y_arr, Sy, Sx,
                )
                return {
                    "ref_all": ref_all, "gx_all": gx_all, "gy_all": gy_all,
                    "mask_all": mask_all, "XX_all": XX_all, "YY_all": YY_all,
                    "H2_img_all": H2_img_all,
                    "meanf_all": meanf_all, "bottomf_all": bottomf_all,
                    "valid": valid, "Sy": Sy, "Sx": Sx, "img_h": h, "img_w": w,
                }
        except Exception:
            logger.warning(
                "Numba precompute_2dof failed, using Python fallback.",
                exc_info=True,
            )

    # --- Fallback: Python loop ---
    return _precompute_subsets_2dof_python(
        coords, img_ref, df_dx, df_dy, img_ref_mask,
        winsize_x_arr, winsize_y_arr,
    )


def _precompute_subsets_2dof_python(
    coords, img_ref, df_dx, df_dy, img_ref_mask,
    winsize_x_arr, winsize_y_arr,
):
    """Python-loop fallback for precompute_subsets_2dof."""
    N = coords.shape[0]
    max_wx = int(winsize_x_arr.max())
    max_wy = int(winsize_y_arr.max())
    Sx = max_wx + 1
    Sy = max_wy + 1
    h, w = img_ref.shape

    ref_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    gx_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    gy_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    mask_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    XX_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    YY_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    H2_img_all = np.zeros((N, 2, 2), dtype=np.float64)
    meanf_all = np.zeros(N, dtype=np.float64)
    bottomf_all = np.ones(N, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        x0, y0 = coords[i]
        half_wx = int(winsize_x_arr[i]) // 2
        half_wy = int(winsize_y_arr[i]) // 2
        sy = int(winsize_y_arr[i]) + 1
        sx = int(winsize_x_arr[i]) + 1

        x_lo = int(round(x0) - half_wx)
        x_hi = int(round(x0) + half_wx)
        y_lo = int(round(y0) - half_wy)
        y_hi = int(round(y0) + half_wy)

        if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
            continue

        mask_patch = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ref_patch_raw = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1]

        # Connected component containing subset center (mask-based validity)
        bw = _connected_center_mask(mask_patch > 0.5)
        n_connected = int(np.sum(bw > 0.5))
        if n_connected < int(0.5 * mask_patch.size):
            continue

        # Apply connected component mask to raw pixels and gradients
        ref_sub = ref_patch_raw * bw
        gx_sub = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw
        gy_sub = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw

        xx_rel = np.arange(x_lo, x_hi + 1, dtype=np.float64) - round(x0)
        yy_rel = np.arange(y_lo, y_hi + 1, dtype=np.float64) - round(y0)
        XX = np.broadcast_to(xx_rel[np.newaxis, :], (sy, sx)).copy()
        YY = np.broadcast_to(yy_rel[:, np.newaxis], (sy, sx)).copy()

        ref_all[i, :sy, :sx] = ref_sub
        gx_all[i, :sy, :sx] = gx_sub
        gy_all[i, :sy, :sx] = gy_sub
        mask_all[i, :sy, :sx] = bw
        XX_all[i, :sy, :sx] = XX
        YY_all[i, :sy, :sx] = YY

        gx2 = gx_sub ** 2
        gy2 = gy_sub ** 2
        gxgy = gx_sub * gy_sub
        H2 = np.array([
            [np.sum(gx2), np.sum(gxgy)],
            [np.sum(gxgy), np.sum(gy2)],
        ], dtype=np.float64)

        # Hessian conditioning check — reject if ill-conditioned
        if np.linalg.det(H2) < 1e-12:
            continue

        H2_img_all[i] = H2

        # Statistics from mask-valid pixels (not pixel-value proxy)
        valid_px = bw[:sy, :sx] > 0.5
        n_valid = int(np.sum(valid_px))
        if n_valid < 4:
            continue
        meanf_all[i] = np.mean(ref_sub[valid_px])
        varf = np.var(ref_sub[valid_px])
        bottomf_all[i] = np.sqrt(max((n_valid - 1) * varf, 1e-30))
        valid[i] = True

    return {
        "ref_all": ref_all, "gx_all": gx_all, "gy_all": gy_all,
        "mask_all": mask_all, "XX_all": XX_all, "YY_all": YY_all,
        "H2_img_all": H2_img_all,
        "meanf_all": meanf_all, "bottomf_all": bottomf_all,
        "valid": valid, "Sy": Sy, "Sx": Sx, "img_h": h, "img_w": w,
    }


# ---------------------------------------------------------------------------
# Batch 6-DOF IC-GN solver
# ---------------------------------------------------------------------------

def icgn_6dof_batch(
    coords: NDArray[np.float64],
    U0_2d: NDArray[np.float64],
    img_ref: NDArray[np.float64],
    img_def: NDArray[np.float64],
    df_dx: NDArray[np.float64],
    df_dy: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    winsize: int,
    tol: float,
    max_iter: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Batch IC-GN 6-DOF solver for all nodes simultaneously.

    Args:
        coords: (N, 2) node positions [x, y].
        U0_2d: (N, 2) initial displacements [u, v].
        img_ref, img_def: Reference and deformed images (H, W).
        df_dx, df_dy: Reference image gradients (H, W).
        img_ref_mask: Binary mask (H, W).
        winsize: Subset size.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        (U, F, conv_iter, mark_hole) where:
            U: (N, 2) displacements
            F: (N, 4) deformation gradients
            conv_iter: (N,) iteration counts
            mark_hole: (N,) bool mask for hole-marked nodes
    """
    pre = precompute_subsets_6dof(
        coords, img_ref, df_dx, df_dy, img_ref_mask, winsize,
    )
    return _iterate_6dof_batch(coords, U0_2d, img_def, pre, tol, max_iter)


def _iterate_6dof_batch(
    coords: NDArray,
    U0_2d: NDArray,
    img_def: NDArray,
    pre: dict,
    tol: float,
    max_iter: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Run batched Gauss-Newton iteration for 6-DOF IC-GN."""
    N = coords.shape[0]
    h, w = pre["img_h"], pre["img_w"]
    Sy, Sx = pre["Sy"], pre["Sx"]

    ref_all = pre["ref_all"]
    gx_all = pre["gx_all"]
    gy_all = pre["gy_all"]
    mask_all = pre["mask_all"]
    XX_all = pre["XX_all"]
    YY_all = pre["YY_all"]
    H_all = pre["H_all"]
    meanf_all = pre["meanf_all"]
    bottomf_all = pre["bottomf_all"]
    valid = pre["valid"]

    # Initialize parameters
    P = np.zeros((N, 6), dtype=np.float64)
    P[:, 4] = U0_2d[:, 0]
    P[:, 5] = U0_2d[:, 1]

    active = valid.copy()
    conv_iter = np.full(N, max_iter + 2, dtype=np.int64)
    conv_iter[valid] = max_iter + 1  # Not-converged default for valid nodes
    norm_init = np.full(N, -1.0, dtype=np.float64)

    x0s = np.round(coords[:, 0])
    y0s = np.round(coords[:, 1])

    for step in range(1, max_iter + 1):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break

        Na = len(idx)
        Pa = P[idx]  # (Na, 6)

        # --- Batch warp coordinates ---
        u22 = ((1.0 + Pa[:, 0, None, None]) * XX_all[idx]
               + Pa[:, 2, None, None] * YY_all[idx]
               + x0s[idx, None, None] + Pa[:, 4, None, None])
        v22 = (Pa[:, 1, None, None] * XX_all[idx]
               + (1.0 + Pa[:, 3, None, None]) * YY_all[idx]
               + y0s[idx, None, None] + Pa[:, 5, None, None])

        # --- Boundary check ---
        margin = 2.5
        oob = ((u22.reshape(Na, -1).min(axis=1) < margin) |
               (u22.reshape(Na, -1).max(axis=1) > w - 1 - margin) |
               (v22.reshape(Na, -1).min(axis=1) < margin) |
               (v22.reshape(Na, -1).max(axis=1) > h - 1 - margin))

        if oob.any():
            active[idx[oob]] = False
            still = ~oob
            idx = idx[still]
            u22 = u22[still]
            v22 = v22[still]
            Na = len(idx)
            if Na == 0:
                continue

        # --- SINGLE batch map_coordinates call ---
        all_g = map_coordinates(
            img_def, [v22.ravel(), u22.ravel()],
            order=3, mode="constant", cval=0.0,
        )
        tempg = all_g.reshape(Na, Sy, Sx)

        # --- Masking ---
        g_valid = np.abs(tempg) > 1e-10
        combined = (mask_all[idx] > 0.5) & g_valid
        valid_counts = combined.sum(axis=(1, 2))

        too_few = valid_counts < 4
        if too_few.any():
            active[idx[too_few]] = False
            keep = ~too_few
            idx, tempg, combined, valid_counts = (
                idx[keep], tempg[keep], combined[keep], valid_counts[keep],
            )
            Na = len(idx)
            if Na == 0:
                continue

        # Apply masks
        tempf_m = ref_all[idx] * combined
        gx_m = gx_all[idx] * combined
        gy_m = gy_all[idx] * combined
        tempg_m = tempg * combined

        # --- Batch ZNSSD ---
        g_sum = np.sum(tempg_m, axis=(1, 2))
        meang = g_sum / valid_counts
        tempg_c = tempg_m - meang[:, None, None] * combined
        varg = np.sum(tempg_c ** 2, axis=(1, 2)) / valid_counts
        bottomg = np.sqrt(np.maximum((valid_counts - 1) * varg, 1e-30))

        mf = meanf_all[idx]
        bf = bottomf_all[idx]
        residual = ((tempf_m - mf[:, None, None]) / bf[:, None, None]
                    - (tempg_m - meang[:, None, None]) / bottomg[:, None, None])

        # --- Batch gradient assembly ---
        XX_a = XX_all[idx]
        YY_a = YY_all[idx]
        b = np.empty((Na, 6), dtype=np.float64)
        b[:, 0] = np.sum(XX_a * gx_m * residual, axis=(1, 2))
        b[:, 1] = np.sum(XX_a * gy_m * residual, axis=(1, 2))
        b[:, 2] = np.sum(YY_a * gx_m * residual, axis=(1, 2))
        b[:, 3] = np.sum(YY_a * gy_m * residual, axis=(1, 2))
        b[:, 4] = np.sum(gx_m * residual, axis=(1, 2))
        b[:, 5] = np.sum(gy_m * residual, axis=(1, 2))
        b *= bf[:, None]

        # --- Convergence check ---
        norm_abs = np.linalg.norm(b, axis=1)
        first = norm_init[idx] < 0
        norm_init[idx[first]] = norm_abs[first]
        ni = norm_init[idx]
        norm_rel = np.where(ni > tol, norm_abs / ni, 0.0)

        converged = (norm_rel < tol) | (norm_abs < tol)
        if converged.any():
            active[idx[converged]] = False
            conv_iter[idx[converged]] = step

        still = ~converged
        idx = idx[still]
        b = b[still]
        Na = len(idx)
        if Na == 0:
            continue

        # --- Batch solve ---
        # Use explicit (m,m),(m,1)->(m,1) form for NumPy 2.x compatibility
        try:
            delta_P = np.linalg.solve(
                H_all[idx], (-b)[:, :, np.newaxis],
            )[:, :, 0]
        except np.linalg.LinAlgError:
            # Fallback: solve individually
            delta_P = np.zeros_like(b)
            for k in range(Na):
                try:
                    delta_P[k] = np.linalg.solve(H_all[idx[k]], -b[k])
                except np.linalg.LinAlgError:
                    active[idx[k]] = False
            continue

        # Supplementary convergence: delta_P too small
        dp_norm = np.linalg.norm(delta_P, axis=1)
        small = dp_norm < tol
        if small.any():
            active[idx[small]] = False
            conv_iter[idx[small]] = step

        # --- Batch compose warp ---
        to_update = ~small
        if to_update.any():
            idx_u = idx[to_update]
            P_new, singular = _compose_warp_batch(P[idx_u], delta_P[to_update])
            ok = ~singular
            P[idx_u[ok]] = P_new[ok]
            if singular.any():
                active[idx_u[singular]] = False

    # Extract results
    U = np.column_stack([P[:, 4], P[:, 5]])
    F = P[:, :4].copy()
    return U, F, conv_iter, pre["mark_hole"]


# ---------------------------------------------------------------------------
# Batch 2-DOF IC-GN solver (ADMM subpb1)
# ---------------------------------------------------------------------------

def icgn_2dof_batch(
    coords: NDArray[np.float64],
    U_old_2d: NDArray[np.float64],
    F_old_2d: NDArray[np.float64],
    udual_2d: NDArray[np.float64],
    img_ref: NDArray[np.float64],
    img_def: NDArray[np.float64],
    df_dx: NDArray[np.float64],
    df_dy: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    winsize_x_arr: NDArray[np.int64],
    winsize_y_arr: NDArray[np.int64],
    mu: float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray, NDArray]:
    """Batch IC-GN 2-DOF solver for ADMM subproblem 1.

    Returns:
        (U, conv_iter) where U is (N, 2) and conv_iter is (N,).
    """
    pre = precompute_subsets_2dof(
        coords, img_ref, df_dx, df_dy, img_ref_mask,
        winsize_x_arr, winsize_y_arr,
    )
    return _iterate_2dof_batch(
        coords, U_old_2d, F_old_2d, udual_2d, img_def, pre, mu, tol, max_iter,
    )


def _iterate_2dof_batch(
    coords: NDArray,
    U_old_2d: NDArray,
    F_old_2d: NDArray,
    udual_2d: NDArray,
    img_def: NDArray,
    pre: dict,
    mu: float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray, NDArray]:
    """Run batched Gauss-Newton iteration for 2-DOF IC-GN."""
    N = coords.shape[0]
    h, w = pre["img_h"], pre["img_w"]
    Sy, Sx = pre["Sy"], pre["Sx"]

    ref_all = pre["ref_all"]
    gx_all = pre["gx_all"]
    gy_all = pre["gy_all"]
    mask_all = pre["mask_all"]
    XX_all = pre["XX_all"]
    YY_all = pre["YY_all"]
    H2_img_all = pre["H2_img_all"]
    meanf_all = pre["meanf_all"]
    bottomf_all = pre["bottomf_all"]
    valid = pre["valid"]

    # Initialize P from subpb2 results
    P = np.zeros((N, 6), dtype=np.float64)
    P[:, 0] = F_old_2d[:, 0]  # F11-1
    P[:, 1] = F_old_2d[:, 1]  # F21
    P[:, 2] = F_old_2d[:, 2]  # F12
    P[:, 3] = F_old_2d[:, 3]  # F22-1
    P[:, 4] = U_old_2d[:, 0]  # u
    P[:, 5] = U_old_2d[:, 1]  # v

    active = valid.copy()
    conv_iter = np.full(N, max_iter + 2, dtype=np.int64)
    conv_iter[valid] = max_iter + 1
    norm_init = np.full(N, -1.0, dtype=np.float64)

    # Combined Hessian: H = 2*H_img/bottomf^2 + mu*I
    bf2 = bottomf_all ** 2
    bf2[bf2 < 1e-30] = 1e-30
    H2_all = np.zeros((N, 2, 2), dtype=np.float64)
    for i in range(N):
        if valid[i]:
            H2_all[i] = H2_img_all[i] * 2.0 / bf2[i] + mu * np.eye(2)

    delta_lm = 1e-3

    x0s = np.round(coords[:, 0])
    y0s = np.round(coords[:, 1])

    for step in range(1, max_iter + 1):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break

        Na = len(idx)
        Pa = P[idx]

        # --- Batch warp coordinates ---
        u22 = ((1.0 + Pa[:, 0, None, None]) * XX_all[idx]
               + Pa[:, 2, None, None] * YY_all[idx]
               + x0s[idx, None, None] + Pa[:, 4, None, None])
        v22 = (Pa[:, 1, None, None] * XX_all[idx]
               + (1.0 + Pa[:, 3, None, None]) * YY_all[idx]
               + y0s[idx, None, None] + Pa[:, 5, None, None])

        # Boundary check
        margin = 2.5
        oob = ((u22.reshape(Na, -1).min(axis=1) < margin) |
               (u22.reshape(Na, -1).max(axis=1) > w - 1 - margin) |
               (v22.reshape(Na, -1).min(axis=1) < margin) |
               (v22.reshape(Na, -1).max(axis=1) > h - 1 - margin))
        if oob.any():
            active[idx[oob]] = False
            still = ~oob
            idx, u22, v22 = idx[still], u22[still], v22[still]
            Na = len(idx)
            if Na == 0:
                continue

        # --- Single batch map_coordinates ---
        all_g = map_coordinates(
            img_def, [v22.ravel(), u22.ravel()],
            order=3, mode="constant", cval=0.0,
        )
        tempg = all_g.reshape(Na, Sy, Sx)

        # Masking
        g_valid = np.abs(tempg) > 1e-10
        combined = (mask_all[idx] > 0.5) & g_valid
        valid_counts = combined.sum(axis=(1, 2))

        too_few = valid_counts < 4
        if too_few.any():
            active[idx[too_few]] = False
            keep = ~too_few
            idx, tempg, combined, valid_counts = (
                idx[keep], tempg[keep], combined[keep], valid_counts[keep],
            )
            Na = len(idx)
            if Na == 0:
                continue

        tempf_m = ref_all[idx] * combined
        gx_m = gx_all[idx] * combined
        gy_m = gy_all[idx] * combined
        tempg_m = tempg * combined

        # --- Batch ZNSSD ---
        g_sum = np.sum(tempg_m, axis=(1, 2))
        meang = g_sum / valid_counts
        tempg_c = tempg_m - meang[:, None, None] * combined
        varg = np.sum(tempg_c ** 2, axis=(1, 2)) / valid_counts
        bottomg = np.sqrt(np.maximum((valid_counts - 1) * varg, 1e-30))

        mf = meanf_all[idx]
        bf = bottomf_all[idx]
        residual = ((tempf_m - mf[:, None, None]) / bf[:, None, None]
                    - (tempg_m - meang[:, None, None]) / bottomg[:, None, None])

        # 2-DOF gradient (image term)
        b_img = np.empty((Na, 2), dtype=np.float64)
        b_img[:, 0] = np.sum(gx_m * residual, axis=(1, 2))
        b_img[:, 1] = np.sum(gy_m * residual, axis=(1, 2))
        b_img *= bf[:, None]

        # Combined gradient: image + ADMM penalty
        Pa_now = P[idx]
        tempb = b_img * 2.0 / (bf[:, None] ** 2) + mu * np.column_stack([
            Pa_now[:, 4] - U_old_2d[idx, 0] - udual_2d[idx, 0],
            Pa_now[:, 5] - U_old_2d[idx, 1] - udual_2d[idx, 1],
        ])

        # Convergence
        norm_abs = np.linalg.norm(tempb, axis=1)
        first = norm_init[idx] < 0
        norm_init[idx[first]] = norm_abs[first]
        ni = norm_init[idx]
        norm_rel = np.where(ni > tol, norm_abs / ni, 0.0)

        converged = (norm_rel < tol) | (norm_abs < tol)
        if converged.any():
            active[idx[converged]] = False
            conv_iter[idx[converged]] = step

        still = ~converged
        idx = idx[still]
        tempb = tempb[still]
        Na = len(idx)
        if Na == 0:
            continue

        # LM-damped 2x2 solve
        H2a = H2_all[idx].copy()
        for k in range(Na):
            H2a[k] += delta_lm * np.max(np.diag(H2a[k])) * np.eye(2)

        try:
            delta_uv = np.linalg.solve(
                H2a, (-tempb)[:, :, np.newaxis],
            )[:, :, 0]
        except np.linalg.LinAlgError:
            delta_uv = np.zeros_like(tempb)
            for k in range(Na):
                try:
                    delta_uv[k] = np.linalg.solve(H2a[k], -tempb[k])
                except np.linalg.LinAlgError:
                    active[idx[k]] = False

            still2 = np.array([active[i] for i in idx])
            idx, delta_uv = idx[still2], delta_uv[still2]
            Na = len(idx)
            if Na == 0:
                continue

        # Small update convergence
        dp_norm = np.linalg.norm(delta_uv, axis=1)
        small = dp_norm < tol
        if small.any():
            active[idx[small]] = False
            conv_iter[idx[small]] = step

        # Compose warp (only translation update)
        to_update = ~small
        if to_update.any():
            idx_u = idx[to_update]
            delta_P_full = np.zeros((to_update.sum(), 6), dtype=np.float64)
            delta_P_full[:, 4] = delta_uv[to_update, 0]
            delta_P_full[:, 5] = delta_uv[to_update, 1]

            P_new, singular = _compose_warp_batch(P[idx_u], delta_P_full)
            ok = ~singular
            P[idx_u[ok]] = P_new[ok]
            if singular.any():
                active[idx_u[singular]] = False

    U = P[:, 4:6].copy()
    return U, conv_iter


# ---------------------------------------------------------------------------
# Vectorized helpers
# ---------------------------------------------------------------------------

def _compose_warp_batch(
    P_batch: NDArray[np.float64],
    delta_P_batch: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Vectorized inverse compositional warp composition for (N, 6) arrays.

    Returns (P_new, singular) where singular is a boolean mask.
    """
    dp = delta_P_batch
    det = (1.0 + dp[:, 0]) * (1.0 + dp[:, 3]) - dp[:, 1] * dp[:, 2]

    singular = np.abs(det) < 1e-15
    det_safe = np.where(singular, 1.0, det)

    cross = dp[:, 0] * dp[:, 3] - dp[:, 1] * dp[:, 2]
    ip0 = (-dp[:, 0] - cross) / det_safe
    ip1 = -dp[:, 1] / det_safe
    ip2 = -dp[:, 2] / det_safe
    ip3 = (-dp[:, 3] - cross) / det_safe
    ip4 = (-dp[:, 4] - dp[:, 3] * dp[:, 4] + dp[:, 2] * dp[:, 5]) / det_safe
    ip5 = (-dp[:, 5] - dp[:, 0] * dp[:, 5] + dp[:, 1] * dp[:, 4]) / det_safe

    a00 = 1.0 + P_batch[:, 0]
    a01 = P_batch[:, 2]
    a02 = P_batch[:, 4]
    a10 = P_batch[:, 1]
    a11 = 1.0 + P_batch[:, 3]
    a12 = P_batch[:, 5]

    b00 = 1.0 + ip0
    b01 = ip2
    b02 = ip4
    b10 = ip1
    b11 = 1.0 + ip3
    b12 = ip5

    result = np.empty_like(P_batch)
    result[:, 0] = a00 * b00 + a01 * b10 - 1.0
    result[:, 1] = a10 * b00 + a11 * b10
    result[:, 2] = a00 * b01 + a01 * b11
    result[:, 3] = a10 * b01 + a11 * b11 - 1.0
    result[:, 4] = a00 * b02 + a01 * b12 + a02
    result[:, 5] = a10 * b02 + a11 * b12 + a12

    return result, singular


def _build_hessian_6dof(XX, YY, grad_x, grad_y):
    """Build 6x6 symmetric Hessian for a single node."""
    gx2 = grad_x ** 2
    gy2 = grad_y ** 2
    gxgy = grad_x * grad_y
    XX2 = XX ** 2
    YY2 = YY ** 2
    XXYY = XX * YY

    H = np.zeros((6, 6), dtype=np.float64)
    H[0, 0] = np.sum(XX2 * gx2)
    H[0, 1] = np.sum(XX2 * gxgy)
    H[0, 2] = np.sum(XXYY * gx2)
    H[0, 3] = np.sum(XXYY * gxgy)
    H[0, 4] = np.sum(XX * gx2)
    H[0, 5] = np.sum(XX * gxgy)
    H[1, 1] = np.sum(XX2 * gy2)
    H[1, 2] = H[0, 3]
    H[1, 3] = np.sum(XXYY * gy2)
    H[1, 4] = H[0, 5]
    H[1, 5] = np.sum(XX * gy2)
    H[2, 2] = np.sum(YY2 * gx2)
    H[2, 3] = np.sum(YY2 * gxgy)
    H[2, 4] = np.sum(YY * gx2)
    H[2, 5] = np.sum(YY * gxgy)
    H[3, 3] = np.sum(YY2 * gy2)
    H[3, 4] = H[2, 5]
    H[3, 5] = np.sum(YY * gy2)
    H[4, 4] = np.sum(gx2)
    H[4, 5] = np.sum(gxgy)
    H[5, 5] = np.sum(gy2)
    H = H + H.T - np.diag(np.diag(H))
    return H


def _connected_center_mask(binary_mask):
    """Connected component containing the center pixel."""
    ny, nx = binary_mask.shape
    cy, cx = ny // 2, nx // 2
    labeled, _ = label(binary_mask)
    cl = labeled[cy, cx]
    if cl == 0:
        return np.zeros_like(binary_mask, dtype=np.float64)
    return (labeled == cl).astype(np.float64)
