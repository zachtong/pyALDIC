"""6-DOF IC-GN subset solver for a single node.

Port of MATLAB solver/icgn_solver.m (Jin Yang, Caltech).

Performs Gauss-Newton Inverse Compositional (IC-GN) iteration to find the
6 affine warp parameters [F11-1, F21, F12, F22-1, Ux, Uy] that best align
a reference subset to a deformed subset.

This is the innermost computational kernel — called once per node per frame
during the local ICGN step.

MATLAB/Python differences:
    - MATLAB uses transposed images: ImgRef(x, y) where x = column.
      Python uses standard: img_ref[y, x] where y = row.
    - MATLAB ``ba_interp2_spline(ImgDef, v, u)`` → Python
      ``scipy.ndimage.map_coordinates(img_def, [y_coords, x_coords], order=3)``.
    - MATLAB ``bwselect`` → ``scipy.ndimage.label`` for connected regions.
    - MATLAB ``ndgrid(xrange, yrange)`` → Python ``np.meshgrid(xrange, yrange, indexing='ij')``.
    - This function is performance-critical and a candidate for ``@numba.njit``.

References:
    B Pan, K Li, W Tong. Fast, robust and accurate DIC computation.
    Opt. Lasers Eng., 56:1406-1414, 2013.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, label

from .icgn_warp import compose_warp


def icgn_solver(
    U0: NDArray[np.float64],
    x0: float,
    y0: float,
    df_dx: NDArray[np.float64],
    df_dy: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    img_ref: NDArray[np.float64],
    img_def: NDArray[np.float64],
    winsize: int,
    tol: float,
    max_iter: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Solve for displacement and deformation gradient at a single node.

    Args:
        U0: Initial displacement guess [Ux, Uy], shape (2,).
        x0: Node x-coordinate (column in image).
        y0: Node y-coordinate (row in image).
        df_dx: Reference image x-gradient (H, W).
        df_dy: Reference image y-gradient (H, W).
        img_ref_mask: Binary mask (H, W), 1.0 = valid.
        img_ref: Normalized reference image (H, W).
        img_def: Normalized deformed image (H, W).
        winsize: Subset size in pixels (full side length = winsize + 1).
        tol: Convergence tolerance.
        max_iter: Maximum Gauss-Newton iterations.

    Returns:
        (U, F, step) where:
            - U: Displacement [Ux, Uy], shape (2,).
            - F: Deformation gradient [F11-1, F21, F12, F22-1], shape (4,).
            - step: Iterations taken. Values > max_iter indicate failure:
              max_iter+1 = not converged, max_iter+2 = masked out,
              max_iter+3 = empty subset.
    """
    h, w = img_ref.shape
    half_w = winsize // 2

    # Warp parameter vector: [F11-1, F21, F12, F22-1, Ux, Uy]
    P = np.array([0.0, 0.0, 0.0, 0.0, U0[0], U0[1]], dtype=np.float64)

    # --- Extract reference subset ---
    x_lo, x_hi = int(x0 - half_w), int(x0 + half_w)
    y_lo, y_hi = int(y0 - half_w), int(y0 + half_w)

    # Bounds check
    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        return np.array([U0[0], U0[1]]), np.zeros(4), max_iter + 2

    # Reference subset and mask
    tempf_mask = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
    tempf = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1] * tempf_mask
    grad_x = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1]
    grad_y = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1]

    # Check valid pixel fraction (>60% must be valid)
    n_masked = np.sum(np.abs(tempf) < 1e-10)
    mask_frac = n_masked / tempf.size

    if mask_frac >= 0.4:
        return np.array([U0[0], U0[1]]), np.zeros(4), max_iter + 2

    # Connected region from center
    bw_mask = _connected_center_mask(tempf_mask > 0.5)
    tempf = tempf * bw_mask
    grad_x = grad_x * bw_mask
    grad_y = grad_y * bw_mask

    # Coordinate grids relative to node center
    ny, nx = tempf.shape
    xx = np.arange(x_lo, x_hi + 1, dtype=np.float64)
    yy = np.arange(y_lo, y_hi + 1, dtype=np.float64)
    # XX[i,j] = xx[j] - x0 (relative x), YY[i,j] = yy[i] - y0 (relative y)
    XX_rel = (xx - x0)[np.newaxis, :]  # (1, nx), broadcast across rows
    YY_rel = (yy - y0)[:, np.newaxis]  # (ny, 1), broadcast across cols
    XX = np.broadcast_to(XX_rel, (ny, nx)).copy()
    YY = np.broadcast_to(YY_rel, (ny, nx)).copy()

    # --- Build Hessian (6x6, symmetric) ---
    H = _build_hessian_6dof(XX, YY, grad_x, grad_y)

    # --- ZNSSD normalization for reference ---
    valid = np.abs(tempf) > 1e-10
    n_valid = valid.sum()
    if n_valid < 4:
        return np.array([P[4], P[5]]), np.array([P[0], P[1], P[2], P[3]]), max_iter + 3

    meanf = np.mean(tempf[valid])
    varf = np.var(tempf[valid])
    bottomf = np.sqrt(max((n_valid - 1) * varf, 1e-30))

    # --- Gauss-Newton iteration ---
    norm_new = 1.0
    norm_abs = 1.0
    norm_init = None
    step = 0

    while step <= max_iter and norm_new > tol and norm_abs > tol:
        step += 1

        # Warp coordinates for deformed image sampling
        u22 = (1.0 + P[0]) * XX + P[2] * YY + x0 + P[4]  # warped x
        v22 = P[1] * XX + (1.0 + P[3]) * YY + y0 + P[5]   # warped y

        # Bounds check on warped coordinates
        margin = 2.5
        if (np.any(u22 < margin) or np.any(u22 > w - 1 - margin) or
                np.any(v22 < margin) or np.any(v22 > h - 1 - margin)):
            norm_new = 1e6
            break

        # Interpolate deformed image: map_coordinates([row, col]) = [y, x]
        tempg = map_coordinates(img_def, [v22.ravel(), u22.ravel()],
                                order=3, mode='constant', cval=0.0)
        tempg = tempg.reshape(ny, nx)

        # Mask handling for deformed image
        g_valid = np.abs(tempg) > 1e-10
        combined_mask = bw_mask.astype(bool) & g_valid

        if combined_mask.sum() < 4:
            norm_new = 1e6
            break

        tempf_iter = tempf * combined_mask
        grad_x_iter = grad_x * combined_mask
        grad_y_iter = grad_y * combined_mask
        tempg = tempg * combined_mask

        # Recompute Hessian if mask changed significantly
        if not np.array_equal(combined_mask, bw_mask.astype(bool)):
            H = _build_hessian_6dof(XX, YY, grad_x_iter, grad_y_iter)
            f_nz = np.abs(tempf_iter) > 1e-10
            n_nz = f_nz.sum()
            if n_nz < 4:
                break
            meanf = np.mean(tempf_iter[f_nz])
            varf = np.var(tempf_iter[f_nz])
            bottomf = np.sqrt(max((n_nz - 1) * varf, 1e-30))

        # ZNSSD normalization for deformed
        g_nz = np.abs(tempg) > 1e-10
        if g_nz.sum() < 4:
            norm_new = 1e6
            break
        meang = np.mean(tempg[g_nz])
        varg = np.var(tempg[g_nz])
        bottomg = np.sqrt(max((g_nz.sum() - 1) * varg, 1e-30))

        # ZNSSD residual
        residual = (tempf_iter - meanf) / bottomf - (tempg - meang) / bottomg

        # Assemble b vector (6x1)
        b = np.zeros(6, dtype=np.float64)
        b[0] = np.sum(XX * grad_x_iter * residual)
        b[1] = np.sum(XX * grad_y_iter * residual)
        b[2] = np.sum(YY * grad_x_iter * residual)
        b[3] = np.sum(YY * grad_y_iter * residual)
        b[4] = np.sum(grad_x_iter * residual)
        b[5] = np.sum(grad_y_iter * residual)
        b *= bottomf

        # Convergence check
        norm_abs = np.linalg.norm(b)
        if norm_init is None:
            norm_init = norm_abs
        norm_new = norm_abs / norm_init if norm_init > tol else 0.0

        if norm_new < tol or norm_abs < tol:
            break

        # Solve for DeltaP
        try:
            delta_P = -np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break

        # Supplementary convergence: parameter update negligible means the
        # solver is at the optimum regardless of the gradient norm (which is
        # amplified by bottomf and may remain above tol due to interpolation noise).
        if np.linalg.norm(delta_P) < tol:
            norm_new = 0.0
            norm_abs = 0.0
            break

        # Compose warp
        result = compose_warp(P, delta_P)
        if result is None:
            break
        P = result

    # Determine convergence status
    if not (norm_new < tol or norm_abs < tol):
        step = max_iter + 1
    if np.isnan(norm_new):
        step = max_iter + 1

    U = np.array([P[4], P[5]], dtype=np.float64)
    F = np.array([P[0], P[1], P[2], P[3]], dtype=np.float64)

    return U, F, step


def _build_hessian_6dof(
    XX: NDArray[np.float64],
    YY: NDArray[np.float64],
    grad_x: NDArray[np.float64],
    grad_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the 6x6 symmetric Hessian for IC-GN.

    The Steepest Descent Images are:
        SD[0] = XX * DfDx,  SD[1] = XX * DfDy
        SD[2] = YY * DfDx,  SD[3] = YY * DfDy
        SD[4] = DfDx,       SD[5] = DfDy
    """
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

    # Symmetrize
    H = H + H.T - np.diag(np.diag(H))

    return H


def _connected_center_mask(binary_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Get the connected component containing the center pixel.

    Equivalent to MATLAB's ``bwselect(mask, cx, cy, 4)``.
    """
    ny, nx = binary_mask.shape
    cy, cx = ny // 2, nx // 2

    labeled, n_labels = label(binary_mask)
    center_label = labeled[cy, cx]

    if center_label == 0:
        return np.zeros_like(binary_mask, dtype=np.float64)

    return (labeled == center_label).astype(np.float64)
