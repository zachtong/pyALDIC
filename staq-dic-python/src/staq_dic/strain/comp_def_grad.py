"""Local weighted deformation gradient via plane fitting.

Port of MATLAB strain/comp_def_grad.m (Jin Yang, Caltech).

Computes the deformation gradient tensor at each node by fitting a plane
(1st-order polynomial) to the displacement field within a local
neighborhood defined by a pixel-unit search radius.  Uses moving least
squares (MLS) with Gaussian weighting.

MATLAB/Python differences:
    - MATLAB ``rangesearch`` -> ``scipy.spatial.KDTree.query_ball_point``.
    - MATLAB per-node backslash solve -> ``np.linalg.lstsq``.
    - Python returns only F (coordinates and U available to caller).
    - MATLAB ``bwselect`` for mask filtering -> ``scipy.ndimage.label``
      connected component check.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


def comp_def_grad(
    U: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    elements: NDArray[np.int64],
    rad: float,
    mask: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute local deformation gradient via weighted plane fitting.

    For each node, finds neighbors within ``rad`` pixels, fits a 1st-order
    polynomial to displacement using Gaussian weighting (MLS), and extracts
    the displacement gradients.

    Args:
        U: Displacement vector (2*n_nodes,), interleaved [u0,v0,...].
        coordinates: Node coordinates (n_nodes, 2), columns [x, y].
        elements: Element connectivity (n_elements, 8), 0-based.
        rad: Search radius in pixels.
        mask: Optional binary mask (H, W).  Not currently used for
            neighbor filtering (reserved for future use).

    Returns:
        Deformation gradient vector (4*n_nodes,), interleaved as
        [F11_0, F21_0, F12_0, F22_0, ...].
    """
    n_nodes = coordinates.shape[0]
    F = np.full(4 * n_nodes, np.nan, dtype=np.float64)

    if n_nodes == 0:
        return F

    # Build KDTree for neighbor search
    tree = KDTree(coordinates)
    neighbor_lists = tree.query_ball_point(coordinates, rad)

    u = U[0::2]  # x-displacements
    v = U[1::2]  # y-displacements

    # Gaussian weight decay scale: Rd = rad (same as MATLAB)
    rd_sq = rad * rad

    for i in range(n_nodes):
        neighbors = np.array(neighbor_lists[i], dtype=np.int64)

        # Need at least 3 neighbors for a plane fit (3 unknowns: a0, a1, a2)
        if len(neighbors) < 3:
            continue

        xi = coordinates[i, 0]
        yi = coordinates[i, 1]

        # Relative coordinates
        dx = coordinates[neighbors, 0] - xi
        dy = coordinates[neighbors, 1] - yi
        dist_sq = dx * dx + dy * dy

        # Gaussian weights: w = exp(-d²/Rd²)
        w = np.exp(-dist_sq / max(rd_sq, 1e-10))

        # Weighted least squares: [1, dx, dy] * [a0; a1; a2] = u
        # a1 = du/dx, a2 = du/dy
        A = np.column_stack([np.ones(len(neighbors)), dx, dy])  # (n, 3)
        W_diag = np.sqrt(w)  # apply weights via sqrt(W) * A, sqrt(W) * b

        Aw = A * W_diag[:, None]
        uw = u[neighbors] * W_diag
        vw = v[neighbors] * W_diag

        try:
            sol_u, _, _, _ = np.linalg.lstsq(Aw, uw, rcond=None)
            sol_v, _, _, _ = np.linalg.lstsq(Aw, vw, rcond=None)

            F[4 * i + 0] = sol_u[1]  # du/dx = F11
            F[4 * i + 1] = sol_v[1]  # dv/dx = F21
            F[4 * i + 2] = sol_u[2]  # du/dy = F12
            F[4 * i + 3] = sol_v[2]  # dv/dy = F22
        except np.linalg.LinAlgError:
            pass  # Leave as NaN

    return F
