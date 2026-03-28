"""ADMM subproblem 1 dispatcher: per-node 2-DOF IC-GN.

Port of MATLAB solver/subpb1_solver.m (Jin Yang, Caltech).

Dispatches the ADMM subproblem 1 solve across all mesh nodes.  For each
node, updates the displacement while holding the deformation gradient
fixed from subproblem 2.

Backend selection:
1. Numba prange (preferred) — true multi-core, no GIL
2. Batch vectorized NumPy (fallback) — single map_coordinates per iter
3. Sequential per-node (last resort)

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
from .outlier_detection import detect_bad_points, fill_nan_rbf


def precompute_subpb1(
    coordinates_fem: NDArray[np.float64],
    Df: ImageGradients,
    f_img: NDArray[np.float64],
    para: DICPara,
) -> dict:
    """Pre-compute reference subsets for subpb1 (2-DOF IC-GN).

    Call once before the ADMM loop, then pass the result to
    ``subpb1_solver(..., precomputed=pre)`` to avoid redundant work.

    Args:
        coordinates_fem: Node coordinates (n_nodes, 2).
        Df: Reference image gradients.
        f_img: Reference image (H, W).
        para: DIC parameters (uses winsize, winsize_list).

    Returns:
        Pre-computed subset dict for ``subpb1_solver``.
    """
    n_nodes = coordinates_fem.shape[0]
    winsize = para.winsize

    if para.winsize_list is not None and len(para.winsize_list) == n_nodes:
        winsize_x_arr = para.winsize_list[:, 0].astype(int)
        winsize_y_arr = para.winsize_list[:, 1].astype(int)
    else:
        winsize_x_arr = np.full(n_nodes, winsize, dtype=int)
        winsize_y_arr = np.full(n_nodes, winsize, dtype=int)

    from .icgn_batch import precompute_subsets_2dof
    return precompute_subsets_2dof(
        coordinates_fem, f_img, Df.df_dx, Df.df_dy, Df.img_ref_mask,
        winsize_x_arr, winsize_y_arr,
    )


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
    n_workers: int | None = None,
    precomputed: dict | None = None,
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
        n_workers: Unused (kept for API compatibility).
        precomputed: Pre-computed subset data from ``precompute_subpb1``.
            If provided, skips the expensive precomputation step.
            Use this when calling subpb1_solver multiple times with the
            same reference image and mesh (e.g., across ADMM iterations).

    Returns:
        (U, solve_time, conv_iter, bad_pt_num)
    """
    n_nodes = coordinates_fem.shape[0]
    winsize = para.winsize
    max_iter = para.icgn_max_iter

    # Reshape to (N, 2) and (N, 4)
    U_old_2d = USubpb2.reshape(-1, 2)
    F_old_2d = FSubpb2.reshape(-1, 4)
    udual_2d = udual.reshape(-1, 2)

    t0 = time.perf_counter()

    # Use precomputed subsets if available, otherwise compute
    if precomputed is not None:
        pre = precomputed
    else:
        if para.winsize_list is not None and len(para.winsize_list) == n_nodes:
            winsize_x_arr = para.winsize_list[:, 0].astype(int)
            winsize_y_arr = para.winsize_list[:, 1].astype(int)
        else:
            winsize_x_arr = np.full(n_nodes, winsize, dtype=int)
            winsize_y_arr = np.full(n_nodes, winsize, dtype=int)

        from .icgn_batch import precompute_subsets_2dof
        pre = precompute_subsets_2dof(
            coordinates_fem, f_img, Df.df_dx, Df.df_dy, Df.img_ref_mask,
            winsize_x_arr, winsize_y_arr,
        )

    # Dispatch to best available backend
    U_2d, conv_iter = _dispatch_2dof(
        coordinates_fem, U_old_2d, F_old_2d, udual_2d,
        g_img, pre, mu, tol, max_iter,
    )

    solve_time = time.perf_counter() - t0

    # Assemble interleaved displacement
    U = USubpb2.copy()
    U[0::2] = U_2d[:, 0]
    U[1::2] = U_2d[:, 1]

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


def _dispatch_2dof(coords, U_old_2d, F_old_2d, udual_2d, img_def, pre,
                   mu, tol, max_iter):
    """Try Numba prange, then batch vectorized, then sequential."""
    # --- Try Numba (skip for very small N to avoid import/JIT overhead) ---
    N = coords.shape[0]
    if N >= 50:
        try:
            from .numba_kernels import icgn_2dof_parallel, HAS_NUMBA
            if HAS_NUMBA:
                rounded_coords = np.round(coords).astype(np.float64)
                U_out, conv_iter = icgn_2dof_parallel(
                    rounded_coords, U_old_2d.copy(), F_old_2d.copy(), udual_2d.copy(),
                    pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
                    pre["XX_all"], pre["YY_all"], pre["H2_img_all"],
                    pre["meanf_all"], pre["bottomf_all"],
                    pre["valid"], img_def, mu, tol, max_iter,
                )
                return U_out, conv_iter
        except Exception:
            pass

    # --- Fallback: batch vectorized ---
    try:
        from .icgn_batch import _iterate_2dof_batch
        U_out, conv_iter = _iterate_2dof_batch(
            coords, U_old_2d, F_old_2d, udual_2d,
            img_def, pre, mu, tol, max_iter,
        )
        return U_out, conv_iter
    except Exception:
        pass

    # --- Last resort: sequential ---
    return _sequential_2dof(
        coords, U_old_2d, F_old_2d, udual_2d,
        img_def, pre, mu, tol, max_iter,
    )


def _sequential_2dof(coords, U_old_2d, F_old_2d, udual_2d, img_def, pre,
                     mu, tol, max_iter):
    """Sequential per-node fallback using original icgn_subpb1."""
    from .icgn_subpb1 import icgn_subpb1

    N = coords.shape[0]
    Sx = pre["Sx"]
    winsize_default = Sx - 1

    U_out = U_old_2d.copy()
    conv_iter = np.full(N, max_iter + 2, dtype=np.int64)

    img_ref_mask = pre["mask_all"]

    for j in range(N):
        if not pre["valid"][j]:
            continue

        x0 = float(round(coords[j, 0]))
        y0 = float(round(coords[j, 1]))

        try:
            U_j, step = icgn_subpb1(
                U_old_2d[j], F_old_2d[j],
                x0, y0,
                pre["gx_all"][j], pre["gy_all"][j], pre["mask_all"][j],
                pre["ref_all"][j], img_def,
                winsize_default, winsize_default,
                mu, udual_2d[j], tol, max_iter,
            )
            U_out[j] = U_j
            conv_iter[j] = step
        except Exception:
            U_out[j] = np.nan
            conv_iter[j] = -1

    return U_out, conv_iter
