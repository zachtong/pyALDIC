"""Second-order (12-parameter) IC-GN refinement kernels.

The second-order stage starts from a converged first-order result and
refines it with a quadratic subset shape function:

    x' = x0 + dx + u + ux*dx + uy*dy + uxx*dx^2/2 + uxy*dx*dy + uyy*dy^2/2
    y' = y0 + dy + v + vx*dx + vy*dy + vxx*dx^2/2 + vxy*dx*dy + vyy*dy^2/2

Parameter layout used by these kernels:

    p = [u, ux, uy, uxx, uxy, uyy,  v, vx, vy, vxx, vxy, vyy]

The reference-side data (subset, gradients, connected-centre mask, local
coordinates, ZNSSD statistics) are the arrays the first-order precompute
already built, so both orders correlate exactly the same pixels: masked
pixels, pixels cut off by a hole or crack and pixels outside the image
never enter either solve.

The inverse-compositional update composes warps through the 6x6 matrix
that acts on ``[dx^2, dx*dy, dy^2, dx, dy, 1]``, with products truncated
at second order (Gao et al., Opt. Lasers Eng. 65 (2015) 73-80).

A node the kernel cannot improve is reported through a status code and
its parameters are returned unchanged; the kernel never raises and never
writes a non-finite value.
"""

from __future__ import annotations

import numpy as np

from .._numba_compat import JIT_CACHE, njit, prange
from .numba_kernels import _bicubic_interp

N_PAR = 12

STATUS_NOT_RUN = -1
STATUS_OK = 0
STATUS_NOT_CONVERGED = 1
STATUS_OUT_OF_IMAGE = 2
STATUS_ILL_CONDITIONED = 3
STATUS_TOO_FEW_PIXELS = 4
STATUS_NON_FINITE = 5

# Image margin (px) a warped pixel must keep, as in the first-order kernel.
_MARGIN = 2.5
# Deformed pixels at or below this magnitude count as masked, as in the
# first-order kernel.
_ZERO_PIXEL = 1e-10
# Absolute floor on the number of connected pixels (2x the unknowns).
_MIN_PIXELS = 2 * N_PAR


@njit(cache=JIT_CACHE)
def _warp_matrix(p, w):
    """Fill ``w`` (6, 6) with the quadratic warp on [x^2, xy, y^2, x, y, 1]."""
    a0, a1, a2 = p[0], 1.0 + p[1], p[2]
    q1, q2, q3 = 0.5 * p[3], p[4], 0.5 * p[5]
    b0, b1, b2 = p[6], p[7], 1.0 + p[8]
    r1, r2, r3 = 0.5 * p[9], p[10], 0.5 * p[11]

    # x'^2
    w[0, 0] = a1 * a1 + 2.0 * a0 * q1
    w[0, 1] = 2.0 * a1 * a2 + 2.0 * a0 * q2
    w[0, 2] = a2 * a2 + 2.0 * a0 * q3
    w[0, 3] = 2.0 * a0 * a1
    w[0, 4] = 2.0 * a0 * a2
    w[0, 5] = a0 * a0
    # x'y'
    w[1, 0] = a1 * b1 + a0 * r1 + b0 * q1
    w[1, 1] = a1 * b2 + a2 * b1 + a0 * r2 + b0 * q2
    w[1, 2] = a2 * b2 + a0 * r3 + b0 * q3
    w[1, 3] = a0 * b1 + b0 * a1
    w[1, 4] = a0 * b2 + b0 * a2
    w[1, 5] = a0 * b0
    # y'^2
    w[2, 0] = b1 * b1 + 2.0 * b0 * r1
    w[2, 1] = 2.0 * b1 * b2 + 2.0 * b0 * r2
    w[2, 2] = b2 * b2 + 2.0 * b0 * r3
    w[2, 3] = 2.0 * b0 * b1
    w[2, 4] = 2.0 * b0 * b2
    w[2, 5] = b0 * b0
    # x', y', 1
    w[3, 0] = q1
    w[3, 1] = q2
    w[3, 2] = q3
    w[3, 3] = a1
    w[3, 4] = a2
    w[3, 5] = a0
    w[4, 0] = r1
    w[4, 1] = r2
    w[4, 2] = r3
    w[4, 3] = b1
    w[4, 4] = b2
    w[4, 5] = b0
    for k in range(5):
        w[5, k] = 0.0
    w[5, 5] = 1.0


@njit(cache=JIT_CACHE)
def _invert6(a, out):
    """Gauss-Jordan inverse of a 6x6 matrix into ``out``. False if singular.

    Written out instead of ``np.linalg.inv`` so a singular update is a
    return value, not an exception inside a parallel loop.
    """
    n = 6
    work = a.copy()
    for i in range(n):
        for j in range(n):
            out[i, j] = 1.0 if i == j else 0.0
    for col in range(n):
        pivot_row = col
        pivot_val = abs(work[col, col])
        for row in range(col + 1, n):
            if abs(work[row, col]) > pivot_val:
                pivot_val = abs(work[row, col])
                pivot_row = row
        if not (pivot_val > 1e-12):
            return False
        if pivot_row != col:
            for k in range(n):
                work[col, k], work[pivot_row, k] = work[pivot_row, k], work[col, k]
                out[col, k], out[pivot_row, k] = out[pivot_row, k], out[col, k]
        inv_pivot = 1.0 / work[col, col]
        for k in range(n):
            work[col, k] *= inv_pivot
            out[col, k] *= inv_pivot
        for row in range(n):
            if row != col:
                factor = work[row, col]
                if factor != 0.0:
                    for k in range(n):
                        work[row, k] -= factor * work[col, k]
                        out[row, k] -= factor * out[col, k]
    return True


@njit(cache=JIT_CACHE)
def _scaled_cholesky(hess, scale, chol, pivot_min):
    """Cholesky factor of the Jacobi-scaled Hessian. False if ill-conditioned.

    ``scale[k] = sqrt(hess[k, k])``; ``chol`` receives the lower factor of
    ``hess[i, j] / (scale[i] * scale[j])``, whose diagonal is 1. A squared
    pivot below ``pivot_min`` means some parameter is (nearly) a linear
    combination of the others on this subset.
    """
    n = N_PAR
    for k in range(n):
        if not (hess[k, k] > 0.0):
            return False
        scale[k] = np.sqrt(hess[k, k])
    for i in range(n):
        for j in range(i + 1):
            s = hess[i, j] / (scale[i] * scale[j])
            for k in range(j):
                s -= chol[i, k] * chol[j, k]
            if i == j:
                if not (s > pivot_min):
                    return False
                chol[i, i] = np.sqrt(s)
            else:
                chol[i, j] = s / chol[j, j]
    return True


@njit(cache=JIT_CACHE)
def _refine_node(
    x0, y0, p,
    ref_sub, gx_sub, gy_sub, bw, XX, YY, meanf, bottomf,
    img_def, tol_px, max_iter, min_fill, pivot_min,
):
    """Second-order IC-GN at one node, starting from ``p`` (12,).

    ``p`` is updated in place on success and left unchanged otherwise.

    Returns:
        (iterations, status, znssd). ``znssd`` is the ZNSSD of the last
        evaluated residual (NaN when the node was refused before any).
    """
    h, w = img_def.shape
    Sy, Sx = ref_sub.shape

    # --- Footprint: connected pixels and their reach from the centre ---
    n_in = 0
    reach = 0.0
    for i in range(Sy):
        for j in range(Sx):
            if bw[i, j] > 0.5:
                n_in += 1
                r = max(abs(XX[i, j]), abs(YY[i, j]))
                if r > reach:
                    reach = r
    if n_in < _MIN_PIXELS or n_in < min_fill * Sy * Sx:
        return 0, STATUS_TOO_FEW_PIXELS, np.nan

    # --- 12x12 Hessian over the connected pixels ---
    hess = np.zeros((N_PAR, N_PAR))
    sd = np.empty(N_PAR)
    for i in range(Sy):
        for j in range(Sx):
            if bw[i, j] < 0.5:
                continue
            gx = gx_sub[i, j]
            gy = gy_sub[i, j]
            xx = XX[i, j]
            yy = YY[i, j]
            sd[0] = gx
            sd[1] = gx * xx
            sd[2] = gx * yy
            sd[3] = gx * 0.5 * xx * xx
            sd[4] = gx * xx * yy
            sd[5] = gx * 0.5 * yy * yy
            sd[6] = gy
            sd[7] = gy * xx
            sd[8] = gy * yy
            sd[9] = gy * 0.5 * xx * xx
            sd[10] = gy * xx * yy
            sd[11] = gy * 0.5 * yy * yy
            for a in range(N_PAR):
                for b in range(a + 1):
                    hess[a, b] += sd[a] * sd[b]
    for a in range(N_PAR):
        for b in range(a):
            hess[b, a] = hess[a, b]

    scale = np.empty(N_PAR)
    chol = np.zeros((N_PAR, N_PAR))
    if not _scaled_cholesky(hess, scale, chol, pivot_min):
        return 0, STATUS_ILL_CONDITIONED, np.nan

    start = p.copy()
    g = np.zeros((Sy, Sx))
    rhs = np.empty(N_PAR)
    dp = np.empty(N_PAR)
    w_p = np.empty((6, 6))
    w_dp = np.empty((6, 6))
    w_inv = np.empty((6, 6))
    znssd = np.nan

    for it in range(1, max_iter + 1):
        # --- Sample the deformed image through the current warp ---
        for i in range(Sy):
            for j in range(Sx):
                if bw[i, j] < 0.5:
                    g[i, j] = 0.0
                    continue
                xx = XX[i, j]
                yy = YY[i, j]
                xw = (x0 + xx + p[0] + p[1] * xx + p[2] * yy
                      + 0.5 * p[3] * xx * xx + p[4] * xx * yy + 0.5 * p[5] * yy * yy)
                yw = (y0 + yy + p[6] + p[7] * xx + p[8] * yy
                      + 0.5 * p[9] * xx * xx + p[10] * xx * yy + 0.5 * p[11] * yy * yy)
                if (not (xw >= _MARGIN)) or xw > w - 1 - _MARGIN \
                        or (not (yw >= _MARGIN)) or yw > h - 1 - _MARGIN:
                    p[:] = start
                    return it, STATUS_OUT_OF_IMAGE, np.nan
                g[i, j] = _bicubic_interp(img_def, yw, xw, h, w)

        # --- ZNSSD statistics of the deformed subset ---
        n_valid = 0
        g_sum = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw[i, j] > 0.5 and abs(g[i, j]) > _ZERO_PIXEL:
                    n_valid += 1
                    g_sum += g[i, j]
                else:
                    g[i, j] = 0.0
        if n_valid < _MIN_PIXELS:
            p[:] = start
            return it, STATUS_TOO_FEW_PIXELS, np.nan
        meang = g_sum / n_valid
        var_sum = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw[i, j] > 0.5 and abs(g[i, j]) > _ZERO_PIXEL:
                    d = g[i, j] - meang
                    var_sum += d * d
        bottomg = np.sqrt(max((n_valid - 1) * (var_sum / n_valid), 1e-30))

        # --- Residual and right-hand side ---
        for a in range(N_PAR):
            rhs[a] = 0.0
        znssd = 0.0
        for i in range(Sy):
            for j in range(Sx):
                if bw[i, j] < 0.5:
                    continue
                res = ((ref_sub[i, j] - meanf) / bottomf
                       - (g[i, j] - meang) / bottomg)
                znssd += res * res
                gx = gx_sub[i, j]
                gy = gy_sub[i, j]
                xx = XX[i, j]
                yy = YY[i, j]
                rhs[0] += gx * res
                rhs[1] += gx * xx * res
                rhs[2] += gx * yy * res
                rhs[3] += gx * 0.5 * xx * xx * res
                rhs[4] += gx * xx * yy * res
                rhs[5] += gx * 0.5 * yy * yy * res
                rhs[6] += gy * res
                rhs[7] += gy * xx * res
                rhs[8] += gy * yy * res
                rhs[9] += gy * 0.5 * xx * xx * res
                rhs[10] += gy * xx * yy * res
                rhs[11] += gy * 0.5 * yy * yy * res

        # --- dp = -bottomf * H^-1 rhs through the scaled Cholesky factor ---
        for a in range(N_PAR):
            s = rhs[a] * bottomf / scale[a]
            for k in range(a):
                s -= chol[a, k] * dp[k]
            dp[a] = s / chol[a, a]
        for a in range(N_PAR - 1, -1, -1):
            s = dp[a]
            for k in range(a + 1, N_PAR):
                s -= chol[k, a] * dp[k]
            dp[a] = s / chol[a, a]
        for a in range(N_PAR):
            dp[a] = -dp[a] / scale[a]

        # Largest pixel motion this update causes anywhere in the subset.
        step = (np.sqrt(dp[0] * dp[0] + dp[6] * dp[6])
                + reach * np.sqrt(dp[1] * dp[1] + dp[2] * dp[2]
                                  + dp[7] * dp[7] + dp[8] * dp[8])
                + 0.5 * reach * reach * np.sqrt(
                    dp[3] * dp[3] + dp[4] * dp[4] + dp[5] * dp[5]
                    + dp[9] * dp[9] + dp[10] * dp[10] + dp[11] * dp[11]))
        if not (step < 0.5 * (Sx + Sy)):
            # Non-finite, or an update larger than the subset itself.
            p[:] = start
            return it, STATUS_NOT_CONVERGED, np.nan

        # --- Compose: W(p) <- W(p) W(dp)^-1 ---
        _warp_matrix(p, w_p)
        _warp_matrix(dp, w_dp)
        if not _invert6(w_dp, w_inv):
            p[:] = start
            return it, STATUS_NOT_CONVERGED, np.nan
        for row in (3, 4):
            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            c3 = 0.0
            c4 = 0.0
            c5 = 0.0
            for k in range(6):
                wk = w_p[row, k]
                c0 += wk * w_inv[k, 0]
                c1 += wk * w_inv[k, 1]
                c2 += wk * w_inv[k, 2]
                c3 += wk * w_inv[k, 3]
                c4 += wk * w_inv[k, 4]
                c5 += wk * w_inv[k, 5]
            base = 0 if row == 3 else 6
            p[base + 0] = c5
            p[base + 3] = 2.0 * c0
            p[base + 4] = c1
            p[base + 5] = 2.0 * c2
            if row == 3:
                p[1] = c3 - 1.0
                p[2] = c4
            else:
                p[7] = c3
                p[8] = c4 - 1.0

        for a in range(N_PAR):
            if not np.isfinite(p[a]):
                p[:] = start
                return it, STATUS_NON_FINITE, np.nan

        if step < tol_px:
            return it, STATUS_OK, znssd

    p[:] = start
    return max_iter, STATUS_NOT_CONVERGED, np.nan


@njit(parallel=True, cache=JIT_CACHE)
def icgn2_refine_parallel(
    coords, p_all, active,
    ref_all, gx_all, gy_all, mask_all, XX_all, YY_all, meanf_all, bottomf_all,
    img_def, tol_px, max_iter, min_fill, pivot_min,
):
    """Second-order refinement of every active node, in parallel.

    Args:
        coords: (N, 2) rounded node positions ``[x0, y0]``.
        p_all: (N, 12) start parameters (see the module docstring). Not
            modified.
        active: (N,) bool; inactive nodes are skipped.
        ref_all, gx_all, gy_all, mask_all, XX_all, YY_all: (N, Sy, Sx)
            arrays of the first-order precompute.
        meanf_all, bottomf_all: (N,) reference ZNSSD statistics.
        img_def: (H, W) deformed image.
        tol_px: stop when an update moves no subset pixel by more than this.
        max_iter: iteration cap.
        min_fill: minimum connected fraction of the window.
        pivot_min: minimum squared pivot of the scaled Cholesky factor.

    Returns:
        (p_out, iters, status, znssd): refined parameters (equal to
        ``p_all`` wherever ``status != STATUS_OK``), iteration counts,
        status codes (``STATUS_NOT_RUN`` for inactive nodes) and ZNSSD.
    """
    n = coords.shape[0]
    p_out = p_all.copy()
    iters = np.zeros(n, dtype=np.int64)
    status = np.full(n, STATUS_NOT_RUN, dtype=np.int8)
    znssd = np.full(n, np.nan)

    for i in prange(n):
        if not active[i]:
            continue
        it, st, zn = _refine_node(
            coords[i, 0], coords[i, 1], p_out[i],
            ref_all[i], gx_all[i], gy_all[i], mask_all[i], XX_all[i], YY_all[i],
            meanf_all[i], bottomf_all[i],
            img_def, tol_px, max_iter, min_fill, pivot_min,
        )
        iters[i] = it
        status[i] = st
        znssd[i] = zn

    return p_out, iters, status, znssd
