"""Sparse Gaussian smoothing for nodal fields.

Port of MATLAB strain/smooth_field_sparse.m (Jin Yang / Zach Tong).

Applies per-connected-region Gaussian-weighted averaging using
``scipy.spatial.KDTree.query_ball_point`` (equivalent to MATLAB's
``rangesearch``).  O(N log N) complexity, replacing the O(N^3) RBF
smoothing from earlier versions.

MATLAB/Python differences:
    - MATLAB ``rangesearch`` -> ``scipy.spatial.KDTree.query_ball_point``.
    - MATLAB ``bwconncomp`` for region detection -> pre-computed
      ``NodeRegionMap`` from ``utils.region_analysis``.
    - MATLAB handles smoothness=0 as a no-op; Python does the same.
    - The sigma formula is: ``sigma = h * max(0.3, 500 * smoothness)``
      where ``h = winstepsize``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import KDTree

from ..utils.region_analysis import NodeRegionMap


def smooth_field_sparse(
    values: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    sigma: float,
    region_map: NodeRegionMap,
    n_components: int = 2,
) -> NDArray[np.float64]:
    """Smooth a nodal field using sparse Gaussian kernel within regions.

    For each connected region independently, builds a sparse Gaussian
    weight matrix and applies weighted averaging.  This ensures smoothing
    does not bleed across disconnected material regions (e.g., across
    holes).

    Args:
        values: Interleaved nodal values (n_components * n_nodes,).
        coordinates: Node coordinates (n_nodes, 2), columns [x, y].
        sigma: Gaussian kernel standard deviation in pixels.
        region_map: Pre-computed node-to-region mapping.
        n_components: Number of interleaved components (2 or 4).

    Returns:
        Smoothed values, same shape and layout as input ``values``.
    """
    if sigma < 1e-8:
        return values.copy()

    result = values.copy()
    n_nodes = coordinates.shape[0]
    radius = 3.0 * sigma
    two_sigma_sq = 2.0 * sigma * sigma

    for region_nodes in region_map.region_node_lists:
        if len(region_nodes) < 2:
            continue

        # Build KDTree for this region's nodes
        region_coords = coordinates[region_nodes]
        tree = KDTree(region_coords)

        # Find neighbors within radius for all region nodes
        neighbor_lists = tree.query_ball_point(region_coords, radius)

        # Build sparse weight matrix (local indices within region)
        rows = []
        cols = []
        wts = []
        for i, neighbors in enumerate(neighbor_lists):
            if len(neighbors) == 0:
                continue
            neighbors = np.array(neighbors, dtype=np.int64)
            dists_sq = np.sum(
                (region_coords[neighbors] - region_coords[i]) ** 2,
                axis=1,
            )
            w = np.exp(-dists_sq / two_sigma_sq)
            rows.extend([i] * len(neighbors))
            cols.extend(neighbors.tolist())
            wts.extend(w.tolist())

        if len(rows) == 0:
            continue

        n_region = len(region_nodes)
        W = sparse.csr_matrix(
            (wts, (rows, cols)), shape=(n_region, n_region),
        )

        # Row-normalize
        row_sums = np.array(W.sum(axis=1)).ravel()
        row_sums[row_sums < 1e-15] = 1.0
        D_inv = sparse.diags(1.0 / row_sums)
        W = D_inv @ W

        # Apply smoothing to each component
        for c in range(n_components):
            # Extract component values for this region
            global_idx = n_components * region_nodes + c
            vals_c = result[global_idx].copy()

            # Handle NaN: zero out contributions from NaN nodes
            nan_mask = np.isnan(vals_c)
            if nan_mask.all():
                continue

            if nan_mask.any():
                # Zero NaN columns, re-normalize
                vals_c[nan_mask] = 0.0
                # Create modified weight matrix with NaN columns zeroed
                W_mod = W.copy()
                nan_col_mask = np.zeros(n_region, dtype=np.float64)
                nan_col_mask[~nan_mask] = 1.0
                W_mod = W_mod.multiply(sparse.diags(nan_col_mask))
                # Re-normalize rows
                row_sums_mod = np.array(W_mod.sum(axis=1)).ravel()
                row_sums_mod[row_sums_mod < 1e-15] = 1.0
                W_mod = sparse.diags(1.0 / row_sums_mod) @ W_mod
                smoothed = W_mod @ vals_c
                # Restore NaN for originally-NaN nodes
                smoothed[nan_mask] = np.nan
            else:
                smoothed = W @ vals_c

            result[global_idx] = smoothed

    return result
