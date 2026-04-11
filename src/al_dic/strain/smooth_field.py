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
      where ``h = winstepsize`` (uniform) or per-node local spacing
      (non-uniform mesh).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import KDTree

from ..utils.region_analysis import NodeRegionMap


def compute_node_local_spacing(
    coordinates: NDArray[np.float64],
    elements: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute per-node local mesh spacing from element connectivity.

    For each node, averages the edge length of all connected elements.
    Midside (hanging) nodes receive half the parent element edge length,
    reflecting their denser local spacing.

    Args:
        coordinates: (n_nodes, 2) node coordinates.
        elements: (n_elements, 4+) element connectivity (Q4 or Q8).

    Returns:
        (n_nodes,) per-node effective spacing in pixels.
    """
    n_nodes = coordinates.shape[0]
    corners = elements[:, :4]

    # Element edge length ~ diagonal / sqrt(2)
    dx = coordinates[corners[:, 0], 0] - coordinates[corners[:, 2], 0]
    dy = coordinates[corners[:, 0], 1] - coordinates[corners[:, 2], 1]
    elem_h = np.sqrt(dx**2 + dy**2) / np.sqrt(2.0)

    node_h_sum = np.zeros(n_nodes, dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.float64)
    for c in range(4):
        np.add.at(node_h_sum, corners[:, c], elem_h)
        np.add.at(node_count, corners[:, c], 1.0)

    # Midside (hanging) nodes — columns 4-7 when present
    if elements.shape[1] > 4:
        midsides = elements[:, 4:8]
        for c in range(4):
            valid = midsides[:, c] >= 0
            if valid.any():
                np.add.at(node_h_sum, midsides[valid, c], elem_h[valid] * 0.5)
                np.add.at(node_count, midsides[valid, c], 1.0)

    node_count[node_count == 0] = 1.0
    return node_h_sum / node_count


def smooth_field_sparse(
    values: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    sigma: float | NDArray[np.float64],
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
        sigma: Gaussian kernel standard deviation in pixels.  Either a
            scalar (uniform sigma for all nodes) or a (n_nodes,) array
            for per-node adaptive smoothing on non-uniform meshes.
        region_map: Pre-computed node-to-region mapping.
        n_components: Number of interleaved components (2 or 4).

    Returns:
        Smoothed values, same shape and layout as input ``values``.
    """
    n_nodes = coordinates.shape[0]

    # Normalize sigma to per-node array
    if np.isscalar(sigma):
        if sigma < 1e-8:
            return values.copy()
        sigma_arr = np.full(n_nodes, float(sigma), dtype=np.float64)
    else:
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if sigma_arr.max() < 1e-8:
            return values.copy()

    result = values.copy()

    for region_nodes in region_map.region_node_lists:
        if len(region_nodes) < 2:
            continue

        # Build KDTree for this region's nodes
        region_coords = coordinates[region_nodes]
        region_sigma = sigma_arr[region_nodes]
        tree = KDTree(region_coords)

        # Per-node search radius = 3 * sigma_i
        radii = 3.0 * region_sigma
        neighbor_lists = tree.query_ball_point(region_coords, radii)

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
            two_sigma_sq_i = 2.0 * region_sigma[i] * region_sigma[i]
            if two_sigma_sq_i < 1e-15:
                continue
            w = np.exp(-dists_sq / two_sigma_sq_i)
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
