"""FEM shape function and Gauss point evaluation for Q4+hanging-node elements.

Port of MATLAB solver/compute_all_elements_gp.m (Jin Yang, Caltech).

Computes shape functions N, their physical-space derivatives DN, and Jacobian
determinants at a single Gauss point for all elements simultaneously.  Supports
Q4 elements with optional hanging-node midpoints (up to 8-node Q8).

The element formulation uses the ``delta`` flags to selectively activate
midside shape functions:
    - ``delta[k] = 1`` if midside node k+4 exists (hanging node present)
    - ``delta[k] = 0`` if midside node k+4 is absent (standard Q4 edge)

MATLAB/Python differences:
    - MATLAB returns (2, 16, n_ele); Python returns (n_ele, 2, 16).
    - MATLAB's ``sign()`` and NumPy's ``np.sign()`` both return 0 for 0.
    - Fully vectorized over all elements (no Python for-loops in hot path).

References:
    S Funken, A Schmidt. Adaptive mesh refinement in 2D.
    Comp. Meth. Appl. Math. 20:459-479, 2020.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Standard 2x2 Gauss quadrature points and weights
_GP_2x2 = 1.0 / np.sqrt(3.0)
GAUSS_PTS_2x2 = np.array([
    [-_GP_2x2, -_GP_2x2],
    [+_GP_2x2, -_GP_2x2],
    [+_GP_2x2, +_GP_2x2],
    [-_GP_2x2, +_GP_2x2],
], dtype=np.float64)
GAUSS_WTS_2x2 = np.ones(4, dtype=np.float64)  # All weights = 1


def gauss_points(order: int = 2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Gauss quadrature points and weights for a quad element.

    Args:
        order: Gauss quadrature order (1, 2, or 3).

    Returns:
        (points, weights) where points has shape (n_gp, 2) with columns
        [ksi, eta], and weights has shape (n_gp,).
    """
    if order == 1:
        return np.array([[0.0, 0.0]]), np.array([4.0])
    elif order == 2:
        return GAUSS_PTS_2x2.copy(), GAUSS_WTS_2x2.copy()
    elif order == 3:
        g = np.sqrt(3.0 / 5.0)
        pts_1d = np.array([-g, 0.0, g])
        wts_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        pts = np.array([[xi, ei] for ei in pts_1d for xi in pts_1d])
        wts = np.array([wi * wj for wj in wts_1d for wi in wts_1d])
        return pts, wts
    else:
        raise ValueError(f"Gauss order {order} not supported (use 1, 2, or 3)")


def compute_all_elements_gp(
    ksi: float,
    eta: float,
    ptx: NDArray[np.float64],
    pty: NDArray[np.float64],
    delta: NDArray[np.float64],
    n_ele: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute shape functions and derivatives at one Gauss point for all elements.

    Evaluates the Q4+hanging-node shape functions and their physical-space
    derivatives at the reference coordinate ``(ksi, eta)`` for every element
    in the mesh simultaneously.

    Args:
        ksi: Gauss point xi-coordinate in reference space, scalar.
        eta: Gauss point eta-coordinate in reference space, scalar.
        ptx: Element x-coordinates (n_ele, 8).  Columns 0-3 are corner
            nodes; columns 4-7 are midside nodes (0.0 if absent).
        pty: Element y-coordinates (n_ele, 8), same layout as ``ptx``.
        delta: Hanging-node flags (n_ele, 4).  ``delta[e, k] = 1.0`` if
            midside node ``k+4`` of element ``e`` exists.
        n_ele: Number of elements.

    Returns:
        A tuple ``(N_all, DN_all, Jdet_all)`` where:
            - ``N_all``: (n_ele, 2, 16) shape function matrix.
            - ``DN_all``: (n_ele, 4, 16) shape function derivative matrix.
            - ``Jdet_all``: (n_ele,) Jacobian determinants.
    """
    # Extract delta flags: (n_ele,) each
    d5 = delta[:, 0]
    d6 = delta[:, 1]
    d7 = delta[:, 2]
    d8 = delta[:, 3]

    # --- Shape functions (n_ele,) each ---
    # Midside shape functions (conditionally active)
    N5 = d5 * (0.5 * (1.0 + ksi) * (1.0 - abs(eta)))
    N6 = d6 * (0.5 * (1.0 + eta) * (1.0 - abs(ksi)))
    N7 = d7 * (0.5 * (1.0 - ksi) * (1.0 - abs(eta)))
    N8 = d8 * (0.5 * (1.0 - eta) * (1.0 - abs(ksi)))

    # Corner shape functions (adjusted for midside contributions)
    N1 = 0.25 * (1.0 - ksi) * (1.0 - eta) - 0.5 * (N7 + N8)
    N2 = 0.25 * (1.0 + ksi) * (1.0 - eta) - 0.5 * (N8 + N5)
    N3 = 0.25 * (1.0 + ksi) * (1.0 + eta) - 0.5 * (N5 + N6)
    N4 = 0.25 * (1.0 - ksi) * (1.0 + eta) - 0.5 * (N6 + N7)

    # --- Shape function derivatives w.r.t. ksi ---
    seta = np.sign(-eta)  # d|eta|/d_eta = sign(eta), but we need -sign(eta)
    sksi = np.sign(-ksi)

    dN5k = d5 * (0.5 * (1.0 - abs(eta)))
    dN6k = d6 * (0.5 * (1.0 + eta) * sksi)
    dN7k = d7 * (-0.5 * (1.0 - abs(eta)))
    dN8k = d8 * (0.5 * (1.0 - eta) * sksi)

    dN1k = -0.25 * (1.0 - eta) - 0.5 * (dN7k + dN8k)
    dN2k = 0.25 * (1.0 - eta) - 0.5 * (dN8k + dN5k)
    dN3k = 0.25 * (1.0 + eta) - 0.5 * (dN5k + dN6k)
    dN4k = -0.25 * (1.0 + eta) - 0.5 * (dN6k + dN7k)

    # --- Shape function derivatives w.r.t. eta ---
    dN5e = d5 * (0.5 * (1.0 + ksi) * seta)
    dN6e = d6 * (0.5 * (1.0 - abs(ksi)))
    dN7e = d7 * (0.5 * (1.0 - ksi) * seta)
    dN8e = d8 * (-0.5 * (1.0 - abs(ksi)))

    dN1e = -0.25 * (1.0 - ksi) - 0.5 * (dN7e + dN8e)
    dN2e = -0.25 * (1.0 + ksi) - 0.5 * (dN8e + dN5e)
    dN3e = 0.25 * (1.0 + ksi) - 0.5 * (dN5e + dN6e)
    dN4e = 0.25 * (1.0 - ksi) - 0.5 * (dN6e + dN7e)

    # Stack: (n_ele, 8) — columns = [N1..N4, N5..N8]
    dNdk = np.column_stack([dN1k, dN2k, dN3k, dN4k, dN5k, dN6k, dN7k, dN8k])
    dNde = np.column_stack([dN1e, dN2e, dN3e, dN4e, dN5e, dN6e, dN7e, dN8e])

    # --- Jacobian: J = [[J11, J12], [J21, J22]] per element ---
    J11 = np.sum(dNdk * ptx, axis=1)
    J12 = np.sum(dNdk * pty, axis=1)
    J21 = np.sum(dNde * ptx, axis=1)
    J22 = np.sum(dNde * pty, axis=1)
    Jdet_all = J11 * J22 - J12 * J21

    # --- Physical derivatives: J^{-1} * [dN/dksi; dN/deta] ---
    inv_det = 1.0 / Jdet_all  # (n_ele,)
    # dN/dx = (J22 * dN/dksi - J12 * dN/deta) / det
    # dN/dy = (-J21 * dN/dksi + J11 * dN/deta) / det
    dNdx = inv_det[:, None] * (J22[:, None] * dNdk - J12[:, None] * dNde)  # (n_ele, 8)
    dNdy = inv_det[:, None] * (-J21[:, None] * dNdk + J11[:, None] * dNde)  # (n_ele, 8)

    # --- Build N_all: (n_ele, 2, 16) ---
    Nvals = np.column_stack([N1, N2, N3, N4, N5, N6, N7, N8])  # (n_ele, 8)
    N_all = np.zeros((n_ele, 2, 16), dtype=np.float64)
    for k in range(8):
        N_all[:, 0, 2 * k] = Nvals[:, k]      # row 0, odd columns (u DOFs)
        N_all[:, 1, 2 * k + 1] = Nvals[:, k]  # row 1, even columns (v DOFs)

    # --- Build DN_all: (n_ele, 4, 16) ---
    DN_all = np.zeros((n_ele, 4, 16), dtype=np.float64)
    for k in range(8):
        DN_all[:, 0, 2 * k] = dNdx[:, k]      # du/dx
        DN_all[:, 1, 2 * k] = dNdy[:, k]      # du/dy
        DN_all[:, 2, 2 * k + 1] = dNdx[:, k]  # dv/dx
        DN_all[:, 3, 2 * k + 1] = dNdy[:, k]  # dv/dy

    return N_all, DN_all, Jdet_all
