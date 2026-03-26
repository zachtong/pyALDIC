"""Outlier detection and NaN filling for IC-GN results.

Port of MATLAB solver/detect_bad_points.m + solver/fill_nan_rbf.m
(Jin Yang, Caltech).

After IC-GN solving, some nodes may have failed to converge or converged
anomalously slowly.  This module identifies those "bad" nodes and fills
their displacement/gradient values by scattered interpolation from
neighboring good nodes.

MATLAB/Python differences:
    - MATLAB ``scatteredInterpolant('natural', 'nearest')`` →
      ``scipy.interpolate.LinearNDInterpolator`` with nearest-neighbor
      fallback via ``NearestNDInterpolator``.
    - MATLAB ``setdiff`` → ``np.setdiff1d``.
    - MATLAB 1-based indices → Python 0-based.
    - The statistical outlier criterion uses the same formula:
      ``conv_iter > mean + sigma_factor * std``.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def detect_bad_points(
    conv_iter: NDArray[np.int64],
    max_iter_num: int,
    coordinates_fem: NDArray[np.float64],
    sigma_factor: float = 0.25,
    min_threshold: int = 10,
) -> tuple[NDArray[np.int64], int]:
    """Identify IC-GN subsets that failed or converged abnormally.

    Flags nodes that:
        - Have negative convergence count (exception/failure).
        - Exceeded ``max_iter_num - 1`` iterations without converging.
        - Converged significantly slower than the population mean
          (statistical outlier: ``iter > mean + sigma_factor * std``
          and ``iter > min_threshold``).

    Args:
        conv_iter: Per-node convergence iteration counts (n_nodes,).
            Negative values indicate failure.
        max_iter_num: Maximum iteration limit from DICPara.
        coordinates_fem: Node coordinates (n_nodes, 2).
        sigma_factor: Multiplier for standard-deviation-based outlier
            detection.  Default 0.25 (strict) for subpb1, use 1.0 for
            local_icgn.
        min_threshold: Minimum iteration count to be considered an
            outlier.  Default 10 for subpb1, 6 for local_icgn.

    Returns:
        (bad_pts, bad_pt_num) where bad_pts are 0-based indices and
        bad_pt_num excludes mask-only failures (conv_iter == max_iter_num + 2).
    """
    n_nodes = coordinates_fem.shape[0]
    ci = conv_iter.ravel()

    # Nodes with negative convergence (exception failures)
    row1 = np.where(ci < 0)[0]
    # Nodes exceeding max iterations
    row2 = np.where(ci > max_iter_num - 1)[0]
    # Mask-only failures (subset entirely in masked region)
    row3 = np.where(ci == max_iter_num + 2)[0]

    bad_pts = np.unique(np.concatenate([row1, row2]))
    bad_pt_num = len(bad_pts) - len(row3)

    # Statistical outlier detection on "good" points
    good_pts = np.setdiff1d(np.arange(n_nodes), bad_pts)
    if len(good_pts) > 0:
        good_iters = ci[good_pts].astype(np.float64)
        mu_val = np.mean(good_iters)
        sigma_val = np.std(good_iters, ddof=1) if len(good_iters) > 1 else 0.0

        threshold = max(mu_val + sigma_factor * sigma_val, min_threshold)
        row4 = np.where(ci > threshold)[0]
        bad_pts = np.unique(np.concatenate([bad_pts, row4]))

    return bad_pts, max(bad_pt_num, 0)


def fill_nan_rbf(
    V: NDArray[np.float64],
    coordinates_fem: NDArray[np.float64],
    n_components: int = 2,
) -> NDArray[np.float64]:
    """Fill NaN values in an interleaved vector via scattered interpolation.

    Uses ``LinearNDInterpolator`` (Delaunay-based, O(N log N)) with
    nearest-neighbor extrapolation for NaN nodes.

    Args:
        V: Interleaved vector, shape (n_components * n_nodes,).
            For displacement: n_components=2, layout [u0,v0,...].
            For deformation gradient: n_components=4.
        coordinates_fem: Node coordinates (n_nodes, 2), columns [x, y].
        n_components: Number of interleaved components (2 or 4).

    Returns:
        New vector with NaN values replaced by interpolated values.
    """
    V_out = V.copy()
    n_nodes = coordinates_fem.shape[0]
    nc = n_components

    # Identify NaN nodes (check first component)
    nan_idx = np.where(np.isnan(V_out[0::nc]))[0]
    if len(nan_idx) == 0:
        return V_out

    not_nan_idx = np.setdiff1d(np.arange(n_nodes), nan_idx)
    if len(not_nan_idx) == 0:
        warnings.warn(
            "All nodes are NaN, cannot interpolate. Returning zeros.",
            stacklevel=2,
        )
        return np.zeros_like(V_out)

    src_xy = coordinates_fem[not_nan_idx]
    dst_xy = coordinates_fem  # Interpolate to all nodes (overwrites NaN)

    for c in range(nc):
        # Extract values for this component from non-NaN nodes
        src_vals = V_out[nc * not_nan_idx + c]

        # Build interpolator (LinearNDInterpolator may fail for collinear points)
        try:
            interp_lin = LinearNDInterpolator(src_xy, src_vals)
            vals = interp_lin(dst_xy)
        except Exception:
            # Fall back to nearest-neighbor for degenerate geometries
            vals = np.full(len(dst_xy), np.nan)

        # Fill remaining NaN with nearest-neighbor
        interp_nn = NearestNDInterpolator(src_xy, src_vals)
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            vals[nan_mask] = interp_nn(dst_xy[nan_mask])

        V_out[c::nc] = vals

    return V_out
