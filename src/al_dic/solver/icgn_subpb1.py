"""2-DOF IC-GN solver for ADMM subproblem 1 at a single node.

Port of MATLAB solver/icgn_subpb1.m (Jin Yang, Caltech).

Performs IC-GN iteration with only 2 DOF (Ux, Uy) while the deformation
gradient is fixed from the previous ADMM iteration.  The cost function
includes both the image matching term and the ADMM augmented Lagrangian
penalty terms (mu, dual variables).

Unlike the 6-DOF icgn_solver, here F11, F21, F12, F22 are known constants
from subproblem 2, and only translations Ux, Uy are updated.

MATLAB/Python differences:
    - Same image coordinate conventions as icgn_solver.py.
    - The 2x2 Hessian: H = 2*H_image/bottomf^2 + mu*I
    - The gradient includes ADMM penalty: b += mu*(U - UOld - vdual)
    - MATLAB ``ba_interp2_spline`` → ``scipy.ndimage.map_coordinates``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, label

from .icgn_warp import compose_warp


def icgn_subpb1(
    U_old: NDArray[np.float64],
    F_old: NDArray[np.float64],
    x0: float,
    y0: float,
    df_dx: NDArray[np.float64],
    df_dy: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    img_ref: NDArray[np.float64],
    img_def: NDArray[np.float64],
    winsize_x: int,
    winsize_y: int,
    mu: float,
    udual: NDArray[np.float64],
    tol: float,
    max_iter: int = 100,
) -> tuple[NDArray[np.float64], int]:
    """Solve ADMM subproblem 1 for displacement at a single node.

    Minimizes the augmented Lagrangian objective w.r.t. (Ux, Uy):
        image_cost + mu/2 * ||U - UOld - vdual||^2

    Args:
        U_old: Displacement from subproblem 2, [Ux, Uy] shape (2,).
        F_old: Deformation gradient from subproblem 2,
               [F11-1, F21, F12, F22-1] shape (4,).
        x0: Node x-coordinate (column).
        y0: Node y-coordinate (row).
        df_dx: Reference image x-gradient (H, W).
        df_dy: Reference image y-gradient (H, W).
        img_ref_mask: Binary mask (H, W), 1.0 = valid.
        img_ref: Normalized reference image (H, W).
        img_def: Normalized deformed image (H, W).
        winsize_x: Subset half-width in x.
        winsize_y: Subset half-width in y.
        mu: ADMM penalty parameter.
        udual: Displacement dual variable [du, dv], shape (2,).
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        (U, step) where U = [Ux, Uy] and step = iteration count.
        step > max_iter indicates failure.
    """
    h, w = img_ref.shape
    half_wx = winsize_x // 2
    half_wy = winsize_y // 2

    # Initialize P from subpb2 results
    P = np.array([F_old[0], F_old[1], F_old[2], F_old[3],
                  U_old[0], U_old[1]], dtype=np.float64)

    # Subset bounds
    x_lo, x_hi = int(x0 - half_wx), int(x0 + half_wx)
    y_lo, y_hi = int(y0 - half_wy), int(y0 + half_wy)

    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        return np.array([U_old[0], U_old[1]]), max_iter + 2

    # Reference subset
    tempf_mask = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
    tempf = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1] * tempf_mask
    grad_x = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1]
    grad_y = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1]

    # Check valid pixel fraction
    n_masked = np.sum(np.abs(tempf) < 1e-10)
    if n_masked >= 0.4 * tempf.size:
        return np.array([U_old[0], U_old[1]]), max_iter + 2

    # Connected region from center
    bw_mask = _connected_center_mask(tempf_mask > 0.5)
    tempf = tempf * bw_mask
    grad_x = grad_x * bw_mask
    grad_y = grad_y * bw_mask

    ny, nx = tempf.shape

    # Coordinate grids
    xx = np.arange(x_lo, x_hi + 1, dtype=np.float64)
    yy = np.arange(y_lo, y_hi + 1, dtype=np.float64)
    XX = (xx - x0)[np.newaxis, :] * np.ones((ny, 1))
    YY = (yy - y0)[:, np.newaxis] * np.ones((1, nx))

    # --- 2x2 image Hessian ---
    gx2 = grad_x ** 2
    gy2 = grad_y ** 2
    gxgy = grad_x * grad_y
    H_img = np.zeros((2, 2), dtype=np.float64)
    H_img[0, 0] = np.sum(gx2)
    H_img[0, 1] = np.sum(gxgy)
    H_img[1, 1] = np.sum(gy2)
    H_img = H_img + H_img.T - np.diag(np.diag(H_img))

    # ZNSSD normalization for reference
    valid = np.abs(tempf) > 1e-10
    if valid.sum() < 4:
        return np.array([P[4], P[5]]), max_iter + 3

    meanf = np.mean(tempf[valid])
    varf = np.var(tempf[valid])
    bottomf = np.sqrt(max((valid.sum() - 1) * varf, 1e-30))

    # Combined Hessian: H = 2*H_img/bottomf^2 + mu*I
    H2 = H_img * 2.0 / (bottomf ** 2) + mu * np.eye(2)

    # --- Gauss-Newton iteration (2-DOF) ---
    norm_new = 1.0
    norm_abs = 1.0
    norm_init = None
    step = 0
    delta_lm = 1e-3  # Levenberg-Marquardt damping

    while step <= max_iter and norm_new > tol and norm_abs > tol:
        step += 1

        # Warp coordinates
        u22 = (1.0 + P[0]) * XX + P[2] * YY + x0 + P[4]
        v22 = P[1] * XX + (1.0 + P[3]) * YY + y0 + P[5]

        # Bounds check
        margin = 2.5
        if (np.any(u22 < margin) or np.any(u22 > w - 1 - margin) or
                np.any(v22 < margin) or np.any(v22 > h - 1 - margin)):
            norm_new = 1e6
            break

        # Interpolate deformed image
        tempg = map_coordinates(img_def, [v22.ravel(), u22.ravel()],
                                order=3, mode='constant', cval=0.0)
        tempg = tempg.reshape(ny, nx)

        # Mask handling
        g_valid = np.abs(tempg) > 1e-10
        combined_mask = bw_mask.astype(bool) & g_valid

        if combined_mask.sum() < 4:
            norm_new = 1e6
            break

        tempf_iter = tempf * combined_mask
        grad_x_iter = grad_x * combined_mask
        grad_y_iter = grad_y * combined_mask
        tempg = tempg * combined_mask

        # Recompute Hessian if mask changed
        if not np.array_equal(combined_mask, bw_mask.astype(bool)):
            H_img = np.zeros((2, 2), dtype=np.float64)
            H_img[0, 0] = np.sum(grad_x_iter ** 2)
            H_img[0, 1] = np.sum(grad_x_iter * grad_y_iter)
            H_img[1, 1] = np.sum(grad_y_iter ** 2)
            H_img = H_img + H_img.T - np.diag(np.diag(H_img))

            f_nz = np.abs(tempf_iter) > 1e-10
            if f_nz.sum() < 4:
                break
            meanf = np.mean(tempf_iter[f_nz])
            varf = np.var(tempf_iter[f_nz])
            bottomf = np.sqrt(max((f_nz.sum() - 1) * varf, 1e-30))
            H2 = H_img * 2.0 / (bottomf ** 2) + mu * np.eye(2)

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

        # Image gradient (2x1)
        b_img = np.zeros(2, dtype=np.float64)
        b_img[0] = np.sum(grad_x_iter * residual)
        b_img[1] = np.sum(grad_y_iter * residual)
        b_img *= bottomf

        # Combined gradient: image term + ADMM penalty
        tempb = b_img * 2.0 / (bottomf ** 2) + mu * np.array([
            P[4] - U_old[0] - udual[0],
            P[5] - U_old[1] - udual[1],
        ])

        # Convergence check
        norm_abs = np.linalg.norm(tempb)
        if norm_init is None:
            norm_init = norm_abs
        norm_new = norm_abs / norm_init if norm_init > tol else 0.0

        if norm_new < tol or norm_abs < tol:
            break

        # Solve 2x2 system with LM damping
        H_damped = H2 + delta_lm * np.max(np.diag(H2)) * np.eye(2)
        try:
            delta_uv = -np.linalg.solve(H_damped, tempb)
        except np.linalg.LinAlgError:
            break

        # Supplementary convergence: displacement update negligible
        if np.linalg.norm(delta_uv) < tol:
            norm_new = 0.0
            norm_abs = 0.0
            break

        # Build 6-DOF delta_P (only translation components)
        delta_P = np.array([0.0, 0.0, 0.0, 0.0, delta_uv[0], delta_uv[1]])

        # Compose warp
        result = compose_warp(P, delta_P)
        if result is None:
            break
        P = result

    # Convergence status
    if not (norm_new < tol or norm_abs < tol):
        step = max_iter + 1
    if np.isnan(norm_new):
        step = max_iter + 1

    return np.array([P[4], P[5]], dtype=np.float64), step


def _connected_center_mask(binary_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Get the connected component containing the center pixel."""
    ny, nx = binary_mask.shape
    cy, cx = ny // 2, nx // 2
    labeled, _ = label(binary_mask)
    center_label = labeled[cy, cx]
    if center_label == 0:
        return np.zeros_like(binary_mask, dtype=np.float64)
    return (labeled == center_label).astype(np.float64)
