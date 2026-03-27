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
from scipy.ndimage import generic_filter


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
    """One pass of outlier detection and removal.

    For each interior pixel, computes a weighted neighbor average and
    standard deviation.  If ``|value - avg| > n_sigma * std``, the
    pixel and its neighbors are set to NaN.

    Matches MATLAB init_disp.m logic.
    """
    u_out = u.copy()
    v_out = v.copy()
    ny, nx = u.shape

    if ny < 3 or nx < 3:
        return u_out, v_out

    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            if use_8_neighbors:
                # 8-neighbor weighted average: cardinal=1/8, diagonal=1/16
                u_card = np.array([
                    u[iy - 1, ix], u[iy + 1, ix],
                    u[iy, ix - 1], u[iy, ix + 1],
                ])
                u_diag = np.array([
                    u[iy - 1, ix - 1], u[iy + 1, ix - 1],
                    u[iy + 1, ix + 1], u[iy - 1, ix + 1],
                ])
                u_all = np.concatenate([u_card, u_diag])
                v_card = np.array([
                    v[iy - 1, ix], v[iy + 1, ix],
                    v[iy, ix - 1], v[iy, ix + 1],
                ])
                v_diag = np.array([
                    v[iy - 1, ix - 1], v[iy + 1, ix - 1],
                    v[iy + 1, ix + 1], v[iy - 1, ix + 1],
                ])
                v_all = np.concatenate([v_card, v_diag])

                # Weighted average: (1/8 * sum(cardinal) + 1/16 * sum(diag)) / (3/4)
                u_avg = (np.nansum(u_card) / 8 + np.nansum(u_diag) / 16) / 0.75
                v_avg = (np.nansum(v_card) / 8 + np.nansum(v_diag) / 16) / 0.75
                u_std = np.nanstd(u_all)
                v_std = np.nanstd(v_all)
            else:
                # 4-neighbor mean
                u_nb = np.array([
                    u[iy - 1, ix], u[iy + 1, ix],
                    u[iy, ix - 1], u[iy, ix + 1],
                ])
                v_nb = np.array([
                    v[iy - 1, ix], v[iy + 1, ix],
                    v[iy, ix - 1], v[iy, ix + 1],
                ])
                u_avg = np.nanmean(u_nb)
                v_avg = np.nanmean(v_nb)
                u_std = np.nanstd(u_nb)
                v_std = np.nanstd(v_nb)

            u_err = abs(u[iy, ix] - u_avg) - n_sigma * u_std
            v_err = abs(v[iy, ix] - v_avg) - n_sigma * v_std

            if u_err > 0 or v_err > 0:
                # NaN the center pixel and all its neighbors
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny2 = iy + dy
                        nx2 = ix + dx
                        if 0 <= ny2 < ny and 0 <= nx2 < nx:
                            u_out[ny2, nx2] = np.nan
                            v_out[ny2, nx2] = np.nan

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
