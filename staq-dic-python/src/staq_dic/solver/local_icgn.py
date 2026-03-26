"""Parallel dispatcher for per-node IC-GN subset solving.

Port of MATLAB solver/local_icgn.m (Jin Yang, Caltech).

Distributes the 6-DOF IC-GN solve across all mesh nodes. After solving,
detects bad points and fills NaN values via scattered interpolation.

MATLAB/Python differences:
    - MATLAB uses parfor for parallel execution; Python uses sequential
      loop for now (Numba/multiprocessing for future optimization).
    - MATLAB's markCoordHoleStrain tracks nodes with >40% masked pixels.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara, ImageGradients
from .icgn_solver import icgn_solver
from .outlier_detection import detect_bad_points, fill_nan_rbf


def local_icgn(
    U0: NDArray[np.float64],
    coordinates_fem: NDArray[np.float64],
    Df: ImageGradients,
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    para: DICPara,
    tol: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.int64],
    int,
    NDArray[np.int64],
]:
    """Dispatch IC-GN solver to all mesh nodes (local ICGN step).

    Args:
        U0: Initial displacement (2*n_nodes,), interleaved [u0,v0,...].
        coordinates_fem: Node coordinates (n_nodes, 2), [x, y].
        Df: Reference image gradients.
        f_img: Reference image (H, W).
        g_img: Deformed image (H, W).
        para: DIC parameters.
        tol: Convergence tolerance.

    Returns:
        (U, F, local_time, conv_iter, bad_pt_num, mark_hole_strain)
    """
    n_nodes = coordinates_fem.shape[0]
    winsize = para.winsize
    max_iter = para.icgn_max_iter

    u_arr = np.zeros(n_nodes, dtype=np.float64)
    v_arr = np.zeros(n_nodes, dtype=np.float64)
    f11 = np.zeros(n_nodes, dtype=np.float64)
    f21 = np.zeros(n_nodes, dtype=np.float64)
    f12 = np.zeros(n_nodes, dtype=np.float64)
    f22 = np.zeros(n_nodes, dtype=np.float64)
    conv_iter = np.zeros(n_nodes, dtype=np.int64)
    mark_hole = np.zeros(n_nodes, dtype=np.bool_)

    t0 = time.perf_counter()

    for j in range(n_nodes):
        x0 = round(coordinates_fem[j, 0])
        y0 = round(coordinates_fem[j, 1])

        # Check mask coverage
        half_w = winsize // 2
        h, w = f_img.shape
        xl, xr = int(x0 - half_w), int(x0 + half_w)
        yl, yr = int(y0 - half_w), int(y0 + half_w)

        if 0 <= xl and xr < w and 0 <= yl and yr < h:
            patch = f_img[yl:yr + 1, xl:xr + 1] * Df.img_ref_mask[yl:yr + 1, xl:xr + 1]
            n_masked = np.sum(np.abs(patch) < 1e-10)
            if n_masked > 0.4 * (winsize + 1) ** 2:
                mark_hole[j] = True
        else:
            mark_hole[j] = True

        try:
            U_j, F_j, step = icgn_solver(
                U0[2 * j: 2 * j + 2],
                float(x0), float(y0),
                Df.df_dx, Df.df_dy, Df.img_ref_mask,
                f_img, g_img,
                winsize, tol, max_iter,
            )
            u_arr[j] = U_j[0]
            v_arr[j] = U_j[1]
            f11[j] = F_j[0]
            f21[j] = F_j[1]
            f12[j] = F_j[2]
            f22[j] = F_j[3]
            conv_iter[j] = step
        except Exception:
            conv_iter[j] = -1
            u_arr[j] = np.nan
            v_arr[j] = np.nan
            f11[j] = np.nan
            f21[j] = np.nan
            f12[j] = np.nan
            f22[j] = np.nan

    local_time = time.perf_counter() - t0

    # Assemble interleaved vectors
    U = np.empty(2 * n_nodes, dtype=np.float64)
    U[0::2] = u_arr
    U[1::2] = v_arr

    F = np.empty(4 * n_nodes, dtype=np.float64)
    F[0::4] = f11
    F[1::4] = f21
    F[2::4] = f12
    F[3::4] = f22

    mark_hole_strain = np.where(mark_hole)[0].astype(np.int64)

    # Detect bad points and fill NaN
    bad_pts, bad_pt_num = detect_bad_points(
        conv_iter, max_iter, coordinates_fem,
        sigma_factor=1.0, min_threshold=6,
    )

    # Set bad points to NaN
    U[2 * bad_pts] = np.nan
    U[2 * bad_pts + 1] = np.nan
    F[4 * bad_pts] = np.nan
    F[4 * bad_pts + 1] = np.nan
    F[4 * bad_pts + 2] = np.nan
    F[4 * bad_pts + 3] = np.nan

    # Fill NaN via interpolation
    U = fill_nan_rbf(U, coordinates_fem, n_components=2)
    F = fill_nan_rbf(F, coordinates_fem, n_components=4)

    return U, F, local_time, conv_iter, bad_pt_num, mark_hole_strain
