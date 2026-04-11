"""Numba-compiled IC-GN kernels for maximum performance.

Provides @njit compiled bicubic interpolation and IC-GN solvers that
run without the GIL, enabling true multi-core parallelism via prange.

Falls back gracefully if Numba is not installed.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Stub decorators so the module can still be imported
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)


# ---------------------------------------------------------------------------
# Bicubic interpolation (Catmull-Rom, a = -0.5)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cubic_weight(t):
    """Catmull-Rom cubic interpolation weight."""
    at = abs(t)
    if at <= 1.0:
        return (1.5 * at - 2.5) * at * at + 1.0
    elif at <= 2.0:
        return ((-0.5 * at + 2.5) * at - 4.0) * at + 2.0
    return 0.0


@njit(cache=True)
def _bicubic_interp(img, y, x, h, w):
    """Bicubic interpolation at a single point (y, x).

    Uses Catmull-Rom kernel (a=-0.5). Returns 0.0 for out-of-bounds.
    """
    if x < 1.0 or x > w - 2.0 or y < 1.0 or y > h - 2.0:
        return 0.0

    iy = int(np.floor(y))
    ix = int(np.floor(x))
    fy = y - iy
    fx = x - ix

    val = 0.0
    for jj in range(-1, 3):
        wy = _cubic_weight(fy - jj)
        yy = iy + jj
        if yy < 0 or yy >= h:
            continue
        for ii in range(-1, 3):
            wx = _cubic_weight(fx - ii)
            xx = ix + ii
            if xx < 0 or xx >= w:
                continue
            val += wy * wx * img[yy, xx]

    return val


# ---------------------------------------------------------------------------
# Per-node IC-GN solver (6-DOF) — fully compiled
# ---------------------------------------------------------------------------

@njit(cache=True)
def _icgn_6dof_single(
    x0, y0, u0, v0,
    ref_subset, gx_subset, gy_subset, bw_mask,
    XX, YY, H, meanf, bottomf,
    img_def, tol, max_iter, h, w, Sy, Sx,
):
    """Full 6-DOF IC-GN iteration for a single node, compiled to native code.

    Returns (P, step) where P is the 6-element parameter vector and
    step is the convergence iteration (>max_iter = failure).
    """
    P = np.zeros(6)
    P[4] = u0
    P[5] = v0

    norm_init = -1.0

    tempg = np.zeros((Sy, Sx))

    for step in range(1, max_iter + 1):
        # --- Warp coordinates and interpolate ---
        # Only iterate masked-in pixels — masked-out pixels (incl. those
        # outside the image for partial-edge subsets) are skipped so they
        # don't trigger out_of_bounds rejection.
        out_of_bounds = False
        margin = 2.5

        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] < 0.5:
                    tempg[i, j] = 0.0
                    continue

                u22 = (1.0 + P[0]) * XX[i, j] + P[2] * YY[i, j] + x0 + P[4]
                v22 = P[1] * XX[i, j] + (1.0 + P[3]) * YY[i, j] + y0 + P[5]

                if u22 < margin or u22 > w - 1 - margin or v22 < margin or v22 > h - 1 - margin:
                    out_of_bounds = True
                    break

                tempg[i, j] = _bicubic_interp(img_def, v22, u22, h, w)

            if out_of_bounds:
                break

        if out_of_bounds:
            return P, max_iter + 1

        # --- Masking ---
        n_valid = 0
        g_sum = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] > 0.5 and abs(tempg[i, j]) > 1e-10:
                    n_valid += 1
                    g_sum += tempg[i, j]
                else:
                    tempg[i, j] = 0.0

        if n_valid < 4:
            return P, max_iter + 1

        # --- ZNSSD for deformed ---
        meang = g_sum / n_valid
        var_sum = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] > 0.5 and abs(tempg[i, j]) > 1e-10:
                    d = tempg[i, j] - meang
                    var_sum += d * d

        varg = var_sum / n_valid
        bottomg = np.sqrt(max((n_valid - 1) * varg, 1e-30))

        # --- Residual and gradient assembly ---
        b = np.zeros(6)
        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] < 0.5:
                    continue

                res = ((ref_subset[i, j] - meanf) / bottomf
                       - (tempg[i, j] - meang) / bottomg)

                gx = gx_subset[i, j]
                gy = gy_subset[i, j]
                xx = XX[i, j]
                yy = YY[i, j]

                b[0] += xx * gx * res
                b[1] += xx * gy * res
                b[2] += yy * gx * res
                b[3] += yy * gy * res
                b[4] += gx * res
                b[5] += gy * res

        for k in range(6):
            b[k] *= bottomf

        # --- Convergence check ---
        norm_abs = 0.0
        for k in range(6):
            norm_abs += b[k] * b[k]
        norm_abs = np.sqrt(norm_abs)

        if norm_init < 0:
            norm_init = norm_abs

        norm_rel = norm_abs / norm_init if norm_init > tol else 0.0

        if norm_rel < tol or norm_abs < tol:
            return P, step

        # --- Solve 6x6 system ---
        # Copy H to avoid modifying the original
        A = H.copy()
        rhs = np.empty(6)
        for k in range(6):
            rhs[k] = -b[k]

        # Gaussian elimination with partial pivoting
        piv = np.arange(6)
        for col in range(6):
            # Find pivot
            max_val = abs(A[col, col])
            max_row = col
            for row in range(col + 1, 6):
                if abs(A[row, col]) > max_val:
                    max_val = abs(A[row, col])
                    max_row = row
            if max_val < 1e-15:
                return P, max_iter + 1  # Singular

            if max_row != col:
                for k in range(6):
                    A[col, k], A[max_row, k] = A[max_row, k], A[col, k]
                rhs[col], rhs[max_row] = rhs[max_row], rhs[col]

            for row in range(col + 1, 6):
                factor = A[row, col] / A[col, col]
                for k in range(col + 1, 6):
                    A[row, k] -= factor * A[col, k]
                rhs[row] -= factor * rhs[col]
                A[row, col] = 0.0

        # Back substitution
        delta_P = np.empty(6)
        for row in range(5, -1, -1):
            s = rhs[row]
            for k in range(row + 1, 6):
                s -= A[row, k] * delta_P[k]
            delta_P[row] = s / A[row, row]

        # Check delta_P norm
        dp_norm = 0.0
        for k in range(6):
            dp_norm += delta_P[k] * delta_P[k]
        dp_norm = np.sqrt(dp_norm)
        if dp_norm < tol:
            return P, step

        # --- Compose warp ---
        det_dp = ((1.0 + delta_P[0]) * (1.0 + delta_P[3])
                  - delta_P[1] * delta_P[2])
        if abs(det_dp) < 1e-15:
            return P, max_iter + 1

        cross = delta_P[0] * delta_P[3] - delta_P[1] * delta_P[2]
        ip0 = (-delta_P[0] - cross) / det_dp
        ip1 = -delta_P[1] / det_dp
        ip2 = -delta_P[2] / det_dp
        ip3 = (-delta_P[3] - cross) / det_dp
        ip4 = (-delta_P[4] - delta_P[3] * delta_P[4] + delta_P[2] * delta_P[5]) / det_dp
        ip5 = (-delta_P[5] - delta_P[0] * delta_P[5] + delta_P[1] * delta_P[4]) / det_dp

        a00 = 1.0 + P[0]
        a01 = P[2]
        a02 = P[4]
        a10 = P[1]
        a11 = 1.0 + P[3]
        a12 = P[5]
        b00 = 1.0 + ip0
        b01 = ip2
        b02 = ip4
        b10 = ip1
        b11 = 1.0 + ip3
        b12 = ip5

        P[0] = a00 * b00 + a01 * b10 - 1.0
        P[1] = a10 * b00 + a11 * b10
        P[2] = a00 * b01 + a01 * b11
        P[3] = a10 * b01 + a11 * b11 - 1.0
        P[4] = a00 * b02 + a01 * b12 + a02
        P[5] = a10 * b02 + a11 * b12 + a12

    return P, max_iter + 1


# ---------------------------------------------------------------------------
# Per-node IC-GN solver (2-DOF for ADMM subpb1) — fully compiled
# ---------------------------------------------------------------------------

@njit(cache=True)
def _icgn_2dof_single(
    x0, y0, u_old0, u_old1, f0, f1, f2, f3,
    udual0, udual1,
    ref_subset, gx_subset, gy_subset, bw_mask,
    XX, YY, H2_img, meanf, bottomf,
    img_def, mu, tol, max_iter, h, w, Sy, Sx,
):
    """2-DOF IC-GN for ADMM subpb1 at a single node."""
    P = np.zeros(6)
    P[0] = f0
    P[1] = f1
    P[2] = f2
    P[3] = f3
    P[4] = u_old0
    P[5] = u_old1

    bf2 = bottomf * bottomf
    if bf2 < 1e-30:
        bf2 = 1e-30

    # Combined Hessian
    H2 = np.empty((2, 2))
    H2[0, 0] = H2_img[0, 0] * 2.0 / bf2 + mu
    H2[0, 1] = H2_img[0, 1] * 2.0 / bf2
    H2[1, 0] = H2_img[1, 0] * 2.0 / bf2
    H2[1, 1] = H2_img[1, 1] * 2.0 / bf2 + mu

    delta_lm = 1e-3
    norm_init = -1.0

    tempg = np.zeros((Sy, Sx))

    for step in range(1, max_iter + 1):
        # Warp + interpolate
        # Only iterate masked-in pixels — masked-out pixels (incl. those
        # outside the image for partial-edge subsets) are skipped so they
        # don't trigger out_of_bounds rejection.
        out_of_bounds = False
        margin = 2.5

        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] < 0.5:
                    tempg[i, j] = 0.0
                    continue
                u22 = (1.0 + P[0]) * XX[i, j] + P[2] * YY[i, j] + x0 + P[4]
                v22 = P[1] * XX[i, j] + (1.0 + P[3]) * YY[i, j] + y0 + P[5]
                if u22 < margin or u22 > w - 1 - margin or v22 < margin or v22 > h - 1 - margin:
                    out_of_bounds = True
                    break
                tempg[i, j] = _bicubic_interp(img_def, v22, u22, h, w)
            if out_of_bounds:
                break

        if out_of_bounds:
            return P[4], P[5], max_iter + 1

        # Masking + ZNSSD
        n_valid = 0
        g_sum = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] > 0.5 and abs(tempg[i, j]) > 1e-10:
                    n_valid += 1
                    g_sum += tempg[i, j]
                else:
                    tempg[i, j] = 0.0

        if n_valid < 4:
            return P[4], P[5], max_iter + 1

        meang = g_sum / n_valid
        var_sum = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] > 0.5 and abs(tempg[i, j]) > 1e-10:
                    d = tempg[i, j] - meang
                    var_sum += d * d
        varg = var_sum / n_valid
        bottomg = np.sqrt(max((n_valid - 1) * varg, 1e-30))

        # Image gradient
        b0 = 0.0
        b1 = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw_mask[i, j] < 0.5:
                    continue
                res = ((ref_subset[i, j] - meanf) / bottomf
                       - (tempg[i, j] - meang) / bottomg)
                b0 += gx_subset[i, j] * res
                b1 += gy_subset[i, j] * res
        b0 *= bottomf
        b1 *= bottomf

        # Combined gradient: image + ADMM penalty
        tb0 = b0 * 2.0 / bf2 + mu * (P[4] - u_old0 - udual0)
        tb1 = b1 * 2.0 / bf2 + mu * (P[5] - u_old1 - udual1)

        norm_abs = np.sqrt(tb0 * tb0 + tb1 * tb1)
        if norm_init < 0:
            norm_init = norm_abs
        norm_rel = norm_abs / norm_init if norm_init > tol else 0.0

        if norm_rel < tol or norm_abs < tol:
            return P[4], P[5], step

        # LM-damped 2x2 solve
        max_diag = max(abs(H2[0, 0]), abs(H2[1, 1]))
        Hd00 = H2[0, 0] + delta_lm * max_diag
        Hd01 = H2[0, 1]
        Hd10 = H2[1, 0]
        Hd11 = H2[1, 1] + delta_lm * max_diag

        det = Hd00 * Hd11 - Hd01 * Hd10
        if abs(det) < 1e-15:
            return P[4], P[5], max_iter + 1

        du = -(Hd11 * tb0 - Hd01 * tb1) / det
        dv = -(-Hd10 * tb0 + Hd00 * tb1) / det

        if np.sqrt(du * du + dv * dv) < tol:
            return P[4], P[5], step

        # Compose warp (translation-only delta)
        det_dp = 1.0  # delta_P has zero deformation gradient components
        ip4 = -du
        ip5 = -dv

        a00 = 1.0 + P[0]
        a01 = P[2]
        a02 = P[4]
        a10 = P[1]
        a11 = 1.0 + P[3]
        a12 = P[5]

        P[4] = a00 * ip4 + a01 * ip5 + a02
        P[5] = a10 * ip4 + a11 * ip5 + a12
        # P[0..3] unchanged since delta_P[0..3] = 0

    return P[4], P[5], max_iter + 1


# ---------------------------------------------------------------------------
# Parallel dispatch via prange
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def icgn_6dof_parallel(
    coords, u0s, v0s,
    ref_subsets, gx_subsets, gy_subsets, bw_masks,
    XX_all, YY_all, H_all, meanf_all, bottomf_all,
    valid, img_def, tol, max_iter,
):
    """Parallel IC-GN 6-DOF dispatch across all nodes using Numba prange.

    Args:
        coords: (N, 2) node positions [x0, y0] (rounded).
        u0s, v0s: (N,) initial displacements.
        ref_subsets: (N, Sy, Sx) pre-extracted reference subsets.
        gx_subsets, gy_subsets: (N, Sy, Sx) gradient subsets.
        bw_masks: (N, Sy, Sx) connected center masks.
        XX_all, YY_all: (N, Sy, Sx) local coordinate grids.
        H_all: (N, 6, 6) pre-built Hessians.
        meanf_all, bottomf_all: (N,) reference ZNSSD stats.
        valid: (N,) boolean mask for valid nodes.
        img_def: (H, W) deformed image.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        (P_out, conv_iter): (N, 6) parameters and (N,) iteration counts.
    """
    N = coords.shape[0]
    h, w = img_def.shape
    Sy = ref_subsets.shape[1]
    Sx = ref_subsets.shape[2]

    P_out = np.zeros((N, 6))
    conv_iter = np.full(N, max_iter + 2, dtype=np.int64)

    for i in prange(N):
        if not valid[i]:
            P_out[i, 4] = u0s[i]
            P_out[i, 5] = v0s[i]
            continue

        P, step = _icgn_6dof_single(
            coords[i, 0], coords[i, 1], u0s[i], v0s[i],
            ref_subsets[i], gx_subsets[i], gy_subsets[i], bw_masks[i],
            XX_all[i], YY_all[i], H_all[i],
            meanf_all[i], bottomf_all[i],
            img_def, tol, max_iter, h, w, Sy, Sx,
        )

        P_out[i] = P
        conv_iter[i] = step

    return P_out, conv_iter


@njit(parallel=True, cache=True)
def icgn_2dof_parallel(
    coords, U_old_2d, F_old_2d, udual_2d,
    ref_subsets, gx_subsets, gy_subsets, bw_masks,
    XX_all, YY_all, H2_img_all, meanf_all, bottomf_all,
    valid, img_def, mu, tol, max_iter,
):
    """Parallel IC-GN 2-DOF dispatch for ADMM subpb1.

    Returns:
        (U_out, conv_iter): (N, 2) displacements and (N,) iteration counts.
    """
    N = coords.shape[0]
    h, w = img_def.shape
    Sy = ref_subsets.shape[1]
    Sx = ref_subsets.shape[2]

    U_out = np.zeros((N, 2))
    conv_iter = np.full(N, max_iter + 2, dtype=np.int64)

    for i in prange(N):
        if not valid[i]:
            U_out[i, 0] = U_old_2d[i, 0]
            U_out[i, 1] = U_old_2d[i, 1]
            continue

        u, v, step = _icgn_2dof_single(
            coords[i, 0], coords[i, 1],
            U_old_2d[i, 0], U_old_2d[i, 1],
            F_old_2d[i, 0], F_old_2d[i, 1],
            F_old_2d[i, 2], F_old_2d[i, 3],
            udual_2d[i, 0], udual_2d[i, 1],
            ref_subsets[i], gx_subsets[i], gy_subsets[i], bw_masks[i],
            XX_all[i], YY_all[i], H2_img_all[i],
            meanf_all[i], bottomf_all[i],
            img_def, mu, tol, max_iter, h, w, Sy, Sx,
        )

        U_out[i, 0] = u
        U_out[i, 1] = v
        conv_iter[i] = step

    return U_out, conv_iter


# ---------------------------------------------------------------------------
# Numba-compiled precompute kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _flood_fill_center(binary_mask, Sy, Sx):
    """BFS flood-fill from center pixel on binary_mask (Sy, Sx).

    Replaces scipy.ndimage.label + center-component extraction.
    Returns a float64 mask (1.0 = connected to center, 0.0 = not).
    """
    result = np.zeros((Sy, Sx), dtype=np.float64)
    cy = Sy // 2
    cx = Sx // 2

    if binary_mask[cy, cx] < 0.5:
        return result

    # BFS with a flat stack (max size = Sy * Sx)
    stack_y = np.empty(Sy * Sx, dtype=np.int32)
    stack_x = np.empty(Sy * Sx, dtype=np.int32)
    visited = np.zeros((Sy, Sx), dtype=np.int8)

    stack_y[0] = cy
    stack_x[0] = cx
    visited[cy, cx] = 1
    head = 0
    tail = 1

    while head < tail:
        py = stack_y[head]
        px = stack_x[head]
        head += 1
        result[py, px] = 1.0

        # 4-connected neighbors
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny_ = py + dy
            nx_ = px + dx
            if 0 <= ny_ < Sy and 0 <= nx_ < Sx:
                if visited[ny_, nx_] == 0 and binary_mask[ny_, nx_] >= 0.5:
                    visited[ny_, nx_] = 1
                    stack_y[tail] = ny_
                    stack_x[tail] = nx_
                    tail += 1

    return result


@njit(cache=True)
def _precompute_one_6dof(
    x0, y0, half_w, Sy, Sx,
    img_ref, df_dx, df_dy, img_ref_mask,
    h, w,
):
    """Pre-compute subset data for one node (6-DOF).

    Returns:
        (ref_sub, gx_sub, gy_sub, bw, XX, YY, H, meanf, bottomf,
         is_valid, is_hole)
    """
    ref_sub = np.zeros((Sy, Sx), dtype=np.float64)
    gx_sub = np.zeros((Sy, Sx), dtype=np.float64)
    gy_sub = np.zeros((Sy, Sx), dtype=np.float64)
    bw = np.zeros((Sy, Sx), dtype=np.float64)
    XX = np.zeros((Sy, Sx), dtype=np.float64)
    YY = np.zeros((Sy, Sx), dtype=np.float64)
    H = np.zeros((6, 6), dtype=np.float64)

    x0r = round(x0)
    y0r = round(y0)
    x_lo = int(x0r) - half_w
    y_lo = int(y0r) - half_w

    # Center pixel must be inside image (sanity check on node position)
    if x0r < 0 or x0r >= w or y0r < 0 or y0r >= h:
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H, 0.0, 1.0, False, True

    # Extract mask patch — out-of-image pixels treated as mask=0 so the
    # flood-fill naturally clips the subset to the in-image region.
    mask_patch = np.empty((Sy, Sx), dtype=np.float64)
    for iy in range(Sy):
        gy_pix = y_lo + iy
        for ix in range(Sx):
            gx_pix = x_lo + ix
            if gy_pix < 0 or gy_pix >= h or gx_pix < 0 or gx_pix >= w:
                mask_patch[iy, ix] = 0.0
            else:
                mask_patch[iy, ix] = img_ref_mask[gy_pix, gx_pix]

    # Flood-fill connected center mask
    bw = _flood_fill_center(mask_patch, Sy, Sx)

    # Count connected-component pixels
    n_connected = 0
    for iy in range(Sy):
        for ix in range(Sx):
            if bw[iy, ix] > 0.5:
                n_connected += 1

    if n_connected < int(0.5 * Sy * Sx):
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H, 0.0, 1.0, False, True

    # Extract subsets with mask — img_ref is raw (unmasked).
    # Since flood-fill already zeroed out-of-image pixels, we can safely
    # read img_ref/df_dx/df_dy only where bw>0.5.
    n_valid = 0
    sum_f = 0.0
    for iy in range(Sy):
        for ix in range(Sx):
            b = bw[iy, ix]
            XX[iy, ix] = float(x_lo + ix) - x0r
            YY[iy, ix] = float(y_lo + iy) - y0r
            if b > 0.5:
                gy_pix = y_lo + iy
                gx_pix = x_lo + ix
                r = img_ref[gy_pix, gx_pix]
                ref_sub[iy, ix] = r
                gx_sub[iy, ix] = df_dx[gy_pix, gx_pix]
                gy_sub[iy, ix] = df_dy[gy_pix, gx_pix]
                n_valid += 1
                sum_f += r

    if n_valid < 4:
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H, 0.0, 1.0, False, False

    # Mean and variance of mask-valid reference pixels
    meanf = sum_f / n_valid
    sum_sq = 0.0
    for iy in range(Sy):
        for ix in range(Sx):
            if bw[iy, ix] > 0.5:
                d = ref_sub[iy, ix] - meanf
                sum_sq += d * d
    varf = sum_sq / n_valid
    bottomf = (max((n_valid - 1) * varf, 1e-30)) ** 0.5

    # Build 6x6 Hessian
    for iy in range(Sy):
        for ix in range(Sx):
            gx = gx_sub[iy, ix]
            gy = gy_sub[iy, ix]
            xx = XX[iy, ix]
            yy = YY[iy, ix]
            gx2 = gx * gx
            gy2 = gy * gy
            gxgy = gx * gy
            xx2 = xx * xx
            yy2 = yy * yy
            xxyy = xx * yy

            H[0, 0] += xx2 * gx2
            H[0, 1] += xx2 * gxgy
            H[0, 2] += xxyy * gx2
            H[0, 3] += xxyy * gxgy
            H[0, 4] += xx * gx2
            H[0, 5] += xx * gxgy
            H[1, 1] += xx2 * gy2
            H[1, 3] += xxyy * gy2
            H[1, 5] += xx * gy2
            H[2, 2] += yy2 * gx2
            H[2, 3] += yy2 * gxgy
            H[2, 4] += yy * gx2
            H[2, 5] += yy * gxgy
            H[3, 3] += yy2 * gy2
            H[3, 5] += yy * gy2
            H[4, 4] += gx2
            H[4, 5] += gxgy
            H[5, 5] += gy2

    # Fill symmetric entries
    H[1, 2] = H[0, 3]
    H[1, 4] = H[0, 5]
    H[2, 5] = H[3, 4] = H[2, 5]  # H[3,4] = H[2,5] from symmetry
    H[3, 4] = H[2, 5]
    # Copy upper to lower
    for r in range(6):
        for c in range(r):
            H[r, c] = H[c, r]

    return ref_sub, gx_sub, gy_sub, bw, XX, YY, H, meanf, bottomf, True, False


@njit(parallel=True, cache=True)
def precompute_subsets_6dof_numba(
    coords, img_ref, df_dx, df_dy, img_ref_mask, half_w, Sy, Sx,
):
    """Parallel pre-compute subsets for 6-DOF IC-GN.

    Args:
        coords: (N, 2) node positions [x, y].
        img_ref: (H, W) reference image.
        df_dx, df_dy: (H, W) image gradients.
        img_ref_mask: (H, W) binary mask.
        half_w: Half window size.
        Sy, Sx: Subset size (winsize + 1).

    Returns:
        Tuple of arrays: (ref_all, gx_all, gy_all, mask_all,
            XX_all, YY_all, H_all, meanf_all, bottomf_all,
            valid, mark_hole)
    """
    N = coords.shape[0]
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

    for i in prange(N):
        (ref_sub, gx_sub, gy_sub, bw, XX, YY, H,
         meanf, bottomf, is_valid, is_hole) = _precompute_one_6dof(
            coords[i, 0], coords[i, 1], half_w, Sy, Sx,
            img_ref, df_dx, df_dy, img_ref_mask, h, w,
        )

        ref_all[i] = ref_sub
        gx_all[i] = gx_sub
        gy_all[i] = gy_sub
        mask_all[i] = bw
        XX_all[i] = XX
        YY_all[i] = YY
        H_all[i] = H
        meanf_all[i] = meanf
        bottomf_all[i] = bottomf
        valid[i] = is_valid
        mark_hole[i] = is_hole

    return (ref_all, gx_all, gy_all, mask_all,
            XX_all, YY_all, H_all, meanf_all, bottomf_all,
            valid, mark_hole)


@njit(cache=True)
def _precompute_one_2dof(
    x0, y0, half_wx, half_wy, sy, sx,
    Sy_max, Sx_max,
    img_ref, df_dx, df_dy, img_ref_mask,
    h, w,
):
    """Pre-compute subset data for one node (2-DOF).

    Returns:
        (ref_sub, gx_sub, gy_sub, bw, XX, YY, H2,
         meanf, bottomf, is_valid)
    """
    ref_sub = np.zeros((Sy_max, Sx_max), dtype=np.float64)
    gx_sub = np.zeros((Sy_max, Sx_max), dtype=np.float64)
    gy_sub = np.zeros((Sy_max, Sx_max), dtype=np.float64)
    bw = np.zeros((Sy_max, Sx_max), dtype=np.float64)
    XX = np.zeros((Sy_max, Sx_max), dtype=np.float64)
    YY = np.zeros((Sy_max, Sx_max), dtype=np.float64)
    H2 = np.zeros((2, 2), dtype=np.float64)

    x0r = round(x0)
    y0r = round(y0)
    x_lo = int(x0r) - half_wx
    y_lo = int(y0r) - half_wy

    # Center pixel must be inside image (sanity check on node position)
    if x0r < 0 or x0r >= w or y0r < 0 or y0r >= h:
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H2, 0.0, 1.0, False

    # Extract mask patch — out-of-image pixels treated as mask=0 so the
    # flood-fill naturally clips the subset to the in-image region.
    mask_patch = np.empty((sy, sx), dtype=np.float64)
    for iy in range(sy):
        gy_pix = y_lo + iy
        for ix in range(sx):
            gx_pix = x_lo + ix
            if gy_pix < 0 or gy_pix >= h or gx_pix < 0 or gx_pix >= w:
                mask_patch[iy, ix] = 0.0
            else:
                mask_patch[iy, ix] = img_ref_mask[gy_pix, gx_pix]

    # Flood-fill connected center mask
    bw_local = _flood_fill_center(mask_patch, sy, sx)

    # Count connected-component pixels
    n_connected = 0
    for iy in range(sy):
        for ix in range(sx):
            if bw_local[iy, ix] > 0.5:
                n_connected += 1

    if n_connected < int(0.5 * sy * sx):
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H2, 0.0, 1.0, False

    # Extract subsets and compute stats — img_ref is raw (unmasked).
    # Read pixels only where bw>0.5 (flood-fill already excluded out-of-image).
    n_valid = 0
    sum_f = 0.0
    sum_gx2 = 0.0
    sum_gy2 = 0.0
    sum_gxgy = 0.0

    for iy in range(sy):
        for ix in range(sx):
            b = bw_local[iy, ix]
            bw[iy, ix] = b
            XX[iy, ix] = float(x_lo + ix) - x0r
            YY[iy, ix] = float(y_lo + iy) - y0r
            if b > 0.5:
                gy_pix = y_lo + iy
                gx_pix = x_lo + ix
                r = img_ref[gy_pix, gx_pix]
                gx = df_dx[gy_pix, gx_pix]
                gy = df_dy[gy_pix, gx_pix]
                ref_sub[iy, ix] = r
                gx_sub[iy, ix] = gx
                gy_sub[iy, ix] = gy
                sum_gx2 += gx * gx
                sum_gy2 += gy * gy
                sum_gxgy += gx * gy
                n_valid += 1
                sum_f += r

    if n_valid < 4:
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H2, 0.0, 1.0, False

    H2[0, 0] = sum_gx2
    H2[0, 1] = sum_gxgy
    H2[1, 0] = sum_gxgy
    H2[1, 1] = sum_gy2

    # Hessian conditioning check
    det_h2 = H2[0, 0] * H2[1, 1] - H2[0, 1] * H2[1, 0]
    if det_h2 < 1e-12:
        return ref_sub, gx_sub, gy_sub, bw, XX, YY, H2, 0.0, 1.0, False

    # Mean and variance of mask-valid reference pixels
    meanf = sum_f / n_valid
    sum_sq = 0.0
    for iy in range(sy):
        for ix in range(sx):
            if bw_local[iy, ix] > 0.5:
                d = ref_sub[iy, ix] - meanf
                sum_sq += d * d
    varf = sum_sq / n_valid
    bottomf = (max((n_valid - 1) * varf, 1e-30)) ** 0.5

    return ref_sub, gx_sub, gy_sub, bw, XX, YY, H2, meanf, bottomf, True


@njit(parallel=True, cache=True)
def precompute_subsets_2dof_numba(
    coords, img_ref, df_dx, df_dy, img_ref_mask,
    winsize_x_arr, winsize_y_arr, Sy_max, Sx_max,
):
    """Parallel pre-compute subsets for 2-DOF IC-GN.

    Args:
        coords: (N, 2) node positions.
        img_ref: (H, W) reference image.
        df_dx, df_dy: (H, W) gradients.
        img_ref_mask: (H, W) mask.
        winsize_x_arr, winsize_y_arr: (N,) per-node window sizes.
        Sy_max, Sx_max: Max subset size (for output array allocation).

    Returns:
        Tuple of arrays.
    """
    N = coords.shape[0]
    h, w = img_ref.shape

    ref_all = np.zeros((N, Sy_max, Sx_max), dtype=np.float64)
    gx_all = np.zeros((N, Sy_max, Sx_max), dtype=np.float64)
    gy_all = np.zeros((N, Sy_max, Sx_max), dtype=np.float64)
    mask_all = np.zeros((N, Sy_max, Sx_max), dtype=np.float64)
    XX_all = np.zeros((N, Sy_max, Sx_max), dtype=np.float64)
    YY_all = np.zeros((N, Sy_max, Sx_max), dtype=np.float64)
    H2_img_all = np.zeros((N, 2, 2), dtype=np.float64)
    meanf_all = np.zeros(N, dtype=np.float64)
    bottomf_all = np.ones(N, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)

    for i in prange(N):
        half_wx = winsize_x_arr[i] // 2
        half_wy = winsize_y_arr[i] // 2
        sx = winsize_x_arr[i] + 1
        sy = winsize_y_arr[i] + 1

        (ref_sub, gx_sub, gy_sub, bw, XX, YY, H2,
         meanf, bottomf, is_valid) = _precompute_one_2dof(
            coords[i, 0], coords[i, 1], half_wx, half_wy, sy, sx,
            Sy_max, Sx_max,
            img_ref, df_dx, df_dy, img_ref_mask, h, w,
        )

        ref_all[i] = ref_sub
        gx_all[i] = gx_sub
        gy_all[i] = gy_sub
        mask_all[i] = bw
        XX_all[i] = XX
        YY_all[i] = YY
        H2_img_all[i] = H2
        meanf_all[i] = meanf
        bottomf_all[i] = bottomf
        valid[i] = is_valid

    return (ref_all, gx_all, gy_all, mask_all,
            XX_all, YY_all, H2_img_all,
            meanf_all, bottomf_all, valid)
