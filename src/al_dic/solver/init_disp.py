"""Initialize displacement from FFT cross-correlation results.

Port of MATLAB solver/init_disp.m (Jin Yang, Caltech).

Processes the raw FFT-based displacement guess: removes outliers via
neighbor-based median filtering, inpaints NaN gaps, then assembles
into the interleaved displacement vector format.

MATLAB/Python differences:
    - MATLAB ``inpaint_nans(u, 4)`` (spring model) -> iterative mean
      fill via ``scipy.ndimage.generic_filter``.
    - MATLAB 8-neighbor weighted average -> vectorized NumPy.
    - MATLAB column-major assembly -> Python row-major with explicit
      interleaving.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation


def init_disp(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    cc_max: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    method: int = 1,
) -> NDArray[np.float64]:
    """Clean up FFT displacement grids and assemble U0 vector.

    Processing steps:
        1. Inpaint NaN values (iterative mean fill).
        2. Three passes of outlier removal (matching MATLAB init_disp.m):
           - Pass 1: 8-neighbor weighted avg, 3*std threshold.
           - Pass 2: 4-neighbor mean, 2*std threshold.
           - Pass 3: 8-neighbor weighted avg, 2*std threshold.
        3. Final inpaint.
        4. Interleave into [u0, v0, u1, v1, ...] format.

    Args:
        u: Raw x-displacement grid (N, M), may contain NaN.
        v: Raw y-displacement grid (N, M), may contain NaN.
        cc_max: Peak NCC values (N, M). Currently unused but kept
            for API compatibility.
        x0: 1-D grid x-coordinates (M,).
        y0: 1-D grid y-coordinates (N,).
        method: 1 = full outlier removal (default), 0 = inpaint only.

    Returns:
        Interleaved displacement vector (2 * N * M,), with nodes
        ordered row-major: [u(0,0), v(0,0), u(0,1), v(0,1), ...].
    """
    u_clean = u.copy()
    v_clean = v.copy()

    # Step 1: Initial NaN inpainting
    u_clean = _inpaint_nans(u_clean)
    v_clean = _inpaint_nans(v_clean)

    if method == 1:
        # Pass 1: 8-neighbor weighted, 3*std threshold
        u_clean, v_clean = _outlier_pass(
            u_clean, v_clean,
            use_8_neighbors=True, n_sigma=3.0,
        )
        u_clean = _inpaint_nans(u_clean)
        v_clean = _inpaint_nans(v_clean)

        # Pass 2: 4-neighbor, 2*std threshold
        u_clean, v_clean = _outlier_pass(
            u_clean, v_clean,
            use_8_neighbors=False, n_sigma=2.0,
        )
        u_clean = _inpaint_nans(u_clean)
        v_clean = _inpaint_nans(v_clean)

        # Pass 3: 8-neighbor weighted, 2*std threshold
        u_clean, v_clean = _outlier_pass(
            u_clean, v_clean,
            use_8_neighbors=True, n_sigma=2.0,
        )
        u_clean = _inpaint_nans(u_clean)
        v_clean = _inpaint_nans(v_clean)

    # Assemble interleaved vector.
    # u_clean is (ny, nx).  mesh_setup uses meshgrid(..., indexing="ij")
    # producing node index = ix * ny + iy.  Transpose + C-ravel matches
    # that ordering (same as MATLAB's uInit = uInit' before vectorizing).
    n_nodes = u_clean.size
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)
    U0[0::2] = u_clean.T.ravel()
    U0[1::2] = v_clean.T.ravel()

    return U0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _inpaint_nans(
    field: NDArray[np.float64],
    max_iter: int = 100,
) -> NDArray[np.float64]:
    """Fill NaN values by iterative mean of non-NaN neighbors.

    Simple diffusion-based inpainting: repeatedly replace each NaN with
    the mean of its non-NaN 4-connected neighbors until convergence.

    This approximates MATLAB's ``inpaint_nans(u, 4)`` spring model.
    """
    result = field.copy()
    for _ in range(max_iter):
        nan_mask = np.isnan(result)
        if not np.any(nan_mask):
            break

        # Compute mean of neighbors for all pixels
        filled = _nan_neighbor_mean(result)
        result[nan_mask] = filled[nan_mask]

    return result


def _nan_neighbor_mean(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute mean of non-NaN 4-connected neighbors for each pixel."""
    ny, nx = field.shape
    result = np.full_like(field, np.nan)

    # Padded array for boundary handling
    padded = np.full((ny + 2, nx + 2), np.nan, dtype=np.float64)
    padded[1:-1, 1:-1] = field

    # 4-connected neighbors
    neighbors = np.stack([
        padded[0:-2, 1:-1],  # up
        padded[2:, 1:-1],    # down
        padded[1:-1, 0:-2],  # left
        padded[1:-1, 2:],    # right
    ], axis=0)

    with np.errstate(all="ignore"):
        result = np.nanmean(neighbors, axis=0)

    return result


def _outlier_pass(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    use_8_neighbors: bool,
    n_sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One pass of outlier detection and removal (vectorized).

    For each interior pixel, computes a weighted neighbor average and
    standard deviation.  If ``|value - avg| > n_sigma * std``, the
    pixel and its 3x3 neighborhood are set to NaN.

    Matches MATLAB init_disp.m logic.
    """
    ny, nx = u.shape
    if ny < 3 or nx < 3:
        return u.copy(), v.copy()

    # Interior slices — all neighbor arrays share the (ny-2, nx-2) interior grid
    ui = u[1:-1, 1:-1]
    vi = v[1:-1, 1:-1]

    # Cardinal neighbors (N, S, W, E)
    u_N = u[0:-2, 1:-1]
    u_S = u[2:,   1:-1]
    u_W = u[1:-1, 0:-2]
    u_E = u[1:-1, 2:]
    v_N = v[0:-2, 1:-1]
    v_S = v[2:,   1:-1]
    v_W = v[1:-1, 0:-2]
    v_E = v[1:-1, 2:]

    if use_8_neighbors:
        # Diagonal neighbors (NW, NE, SW, SE)
        u_NW = u[0:-2, 0:-2]
        u_NE = u[0:-2, 2:]
        u_SW = u[2:,   0:-2]
        u_SE = u[2:,   2:]
        v_NW = v[0:-2, 0:-2]
        v_NE = v[0:-2, 2:]
        v_SW = v[2:,   0:-2]
        v_SE = v[2:,   2:]

        # Weighted average: (cardinal_sum/8 + diagonal_sum/16) / 0.75
        u_card_sum = u_N + u_S + u_W + u_E
        u_diag_sum = u_NW + u_NE + u_SW + u_SE
        u_avg = (u_card_sum / 8.0 + u_diag_sum / 16.0) / 0.75

        v_card_sum = v_N + v_S + v_W + v_E
        v_diag_sum = v_NW + v_NE + v_SW + v_SE
        v_avg = (v_card_sum / 8.0 + v_diag_sum / 16.0) / 0.75

        # Std of all 8 neighbors (population std, ddof=0)
        u_all = np.stack([u_N, u_S, u_W, u_E, u_NW, u_NE, u_SW, u_SE], axis=0)
        v_all = np.stack([v_N, v_S, v_W, v_E, v_NW, v_NE, v_SW, v_SE], axis=0)
        u_std = np.std(u_all, axis=0)
        v_std = np.std(v_all, axis=0)
    else:
        # 4-neighbor mean and std
        u_nb = np.stack([u_N, u_S, u_W, u_E], axis=0)
        v_nb = np.stack([v_N, v_S, v_W, v_E], axis=0)
        u_avg = np.mean(u_nb, axis=0)
        v_avg = np.mean(v_nb, axis=0)
        u_std = np.std(u_nb, axis=0)
        v_std = np.std(v_nb, axis=0)

    # Outlier detection on interior pixels
    u_err = np.abs(ui - u_avg) - n_sigma * u_std
    v_err = np.abs(vi - v_avg) - n_sigma * v_std
    outlier_interior = (u_err > 0) | (v_err > 0)

    # Map interior outlier mask to full grid, then dilate by 3x3
    outlier_full = np.zeros((ny, nx), dtype=bool)
    outlier_full[1:-1, 1:-1] = outlier_interior
    dilated = binary_dilation(outlier_full, structure=np.ones((3, 3), dtype=bool))

    u_out = u.copy()
    v_out = v.copy()
    u_out[dilated] = np.nan
    v_out[dilated] = np.nan

    # NaN the border (MATLAB does this)
    u_out[0, :] = np.nan
    u_out[-1, :] = np.nan
    u_out[:, 0] = np.nan
    u_out[:, -1] = np.nan
    v_out[0, :] = np.nan
    v_out[-1, :] = np.nan
    v_out[:, 0] = np.nan
    v_out[:, -1] = np.nan

    return u_out, v_out
