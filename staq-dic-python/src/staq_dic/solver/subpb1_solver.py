"""ADMM subproblem 1 dispatcher: per-node 2-DOF IC-GN.

Port of MATLAB solver/subpb1_solver.m (Jin Yang, Caltech).

Dispatches the ADMM subproblem 1 solve across all mesh nodes.  For each
node, calls ``icgn_subpb1`` to update the displacement while holding the
deformation gradient fixed from subproblem 2.

MATLAB/Python differences:
    - Same parallelism differences as ``local_icgn.py``.
    - Per-node subset sizes from winsize_list.
    - Bad point detection and NaN filling same structure as local_icgn.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara, ImageGradients
from .icgn_subpb1 import icgn_subpb1
from .outlier_detection import detect_bad_points, fill_nan_rbf


def subpb1_solver(
    USubpb2: NDArray[np.float64],
    FSubpb2: NDArray[np.float64],
    udual: NDArray[np.float64],
    vdual: NDArray[np.float64],
    coordinates_fem: NDArray[np.float64],
    Df: ImageGradients,
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    mu: float,
    beta: float,
    para: DICPara,
    tol: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.int64], int]:
    """Dispatch ADMM subproblem 1 to all mesh nodes.

    Args:
        USubpb2: Displacement from subproblem 2 (2*n_nodes,).
        FSubpb2: Deformation gradient from subproblem 2 (4*n_nodes,).
        udual: Displacement dual variables (2*n_nodes,).
        vdual: Deformation gradient dual variables (4*n_nodes,).
        coordinates_fem: Node coordinates (n_nodes, 2).
        Df: Reference image gradients.
        f_img: Reference image (H, W).
        g_img: Deformed image (H, W).
        mu: ADMM image-matching weight.
        beta: ADMM penalty parameter.
        para: DIC parameters.
        tol: Convergence tolerance.

    Returns:
        (U, solve_time, conv_iter, bad_pt_num)
    """
    n_nodes = coordinates_fem.shape[0]
    winsize = para.winsize
    max_iter = para.icgn_max_iter

    # Per-node subset sizes (use uniform if winsize_list not available)
    if para.winsize_list is not None and len(para.winsize_list) == n_nodes:
        winsize_x_arr = para.winsize_list[:, 0].astype(int)
        winsize_y_arr = para.winsize_list[:, 1].astype(int)
    else:
        winsize_x_arr = np.full(n_nodes, winsize, dtype=int)
        winsize_y_arr = np.full(n_nodes, winsize, dtype=int)

    u_arr = np.zeros(n_nodes, dtype=np.float64)
    v_arr = np.zeros(n_nodes, dtype=np.float64)
    conv_iter = np.zeros(n_nodes, dtype=np.int64)

    t0 = time.perf_counter()

    for j in range(n_nodes):
        x0 = round(coordinates_fem[j, 0])
        y0 = round(coordinates_fem[j, 1])

        U_old_j = USubpb2[2 * j: 2 * j + 2]
        F_old_j = FSubpb2[4 * j: 4 * j + 4]
        udual_j = udual[2 * j: 2 * j + 2]

        try:
            U_j, step = icgn_subpb1(
                U_old_j, F_old_j,
                float(x0), float(y0),
                Df.df_dx, Df.df_dy, Df.img_ref_mask,
                f_img, g_img,
                int(winsize_x_arr[j]), int(winsize_y_arr[j]),
                mu, udual_j, tol, max_iter,
            )
            u_arr[j] = U_j[0]
            v_arr[j] = U_j[1]
            conv_iter[j] = step
        except Exception:
            u_arr[j] = np.nan
            v_arr[j] = np.nan
            conv_iter[j] = -1

    solve_time = time.perf_counter() - t0

    # Assemble interleaved displacement
    U = USubpb2.copy()
    U[0::2] = u_arr
    U[1::2] = v_arr

    # Detect bad points and fill NaN
    sigma_factor = para.outlier_sigma_factor
    min_threshold = para.outlier_min_threshold
    bad_pts, bad_pt_num = detect_bad_points(
        conv_iter, max_iter, coordinates_fem,
        sigma_factor=sigma_factor, min_threshold=min_threshold,
    )

    U[2 * bad_pts] = np.nan
    U[2 * bad_pts + 1] = np.nan

    U = fill_nan_rbf(U, coordinates_fem, n_components=2)

    return U, solve_time, conv_iter, bad_pt_num
