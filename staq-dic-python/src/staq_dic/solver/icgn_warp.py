"""Inverse compositional warp composition for IC-GN solvers.

Port of MATLAB solver/icgn_compose_warp.m (Jin Yang, Caltech).

Implements the IC-GN warp update rule:
    W(P) <- W(P) * W(DeltaP)^{-1}

where P = [F11-1, F21, F12, F22-1, Ux, Uy] is a 6-element affine warp
parameter vector.  Used by both the 6-DOF full IC-GN (icgn_solver) and
the 2-DOF ADMM subproblem 1 (icgn_subpb1).

MATLAB/Python differences:
    - MATLAB uses 1-based indexing into the P vector (P(1)..P(6)).
    - Python uses 0-based indexing (P[0]..P[5]).
    - Returns None instead of MATLAB's [] on singular update.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compose_warp(
    P: NDArray[np.float64],
    delta_P: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compose IC-GN warp parameters via inverse compositional update.

    Computes the updated warp ``W(P_new) = W(P) * W(delta_P)^{-1}``
    using 3x3 homogeneous matrix multiplication.

    Args:
        P: Current warp parameters, shape (6,).
            ``[F11-1, F21, F12, F22-1, Ux, Uy]``.
        delta_P: Incremental warp update, shape (6,), same layout.

    Returns:
        Updated warp parameters, shape (6,), or ``None`` if the delta_P
        warp matrix is singular (``det == 0``).

    Notes:
        The warp matrix convention is::

              W(P) = [[1+P[0], P[2], P[4]],
                      [P[1],   1+P[3], P[5]],
                      [0,      0,      1   ]]

        The update inverts W(delta_P) analytically (no np.linalg.inv needed),
        then multiplies: W(P) @ W(delta_P)^{-1}.
    """
    # Determinant of W(delta_P)
    det_dp = (1.0 + delta_P[0]) * (1.0 + delta_P[3]) - delta_P[1] * delta_P[2]

    if det_dp == 0.0:
        return None  # Singular update — signal failure

    # Analytically invert W(delta_P) → get parameters of W(delta_P)^{-1}
    cross = delta_P[0] * delta_P[3] - delta_P[1] * delta_P[2]
    ip0 = (-delta_P[0] - cross) / det_dp       # iF11 - 1
    ip1 = -delta_P[1] / det_dp                  # iF21
    ip2 = -delta_P[2] / det_dp                  # iF12
    ip3 = (-delta_P[3] - cross) / det_dp        # iF22 - 1
    ip4 = (-delta_P[4] - delta_P[3] * delta_P[4] + delta_P[2] * delta_P[5]) / det_dp  # iUx
    ip5 = (-delta_P[5] - delta_P[0] * delta_P[5] + delta_P[1] * delta_P[4]) / det_dp  # iUy

    # Compose: W(P) @ W(delta_P)^{-1} via 3x3 matrix multiply
    # W(P)         = [[1+P[0], P[2],   P[4]],
    #                 [P[1],   1+P[3], P[5]],
    #                 [0,      0,      1   ]]
    # W(inv)       = [[1+ip0,  ip2,    ip4],
    #                 [ip1,    1+ip3,  ip5],
    #                 [0,      0,      1  ]]
    a00 = 1.0 + P[0]
    a01 = P[2]
    a02 = P[4]
    a10 = P[1]
    a11 = 1.0 + P[3]
    a12 = P[5]

    b00 = 1.0 + ip0
    b01 = ip2
    b02 = ip4
    b10 = ip1
    b11 = 1.0 + ip3
    b12 = ip5

    m00 = a00 * b00 + a01 * b10
    m01 = a00 * b01 + a01 * b11
    m02 = a00 * b02 + a01 * b12 + a02
    m10 = a10 * b00 + a11 * b10
    m11 = a10 * b01 + a11 * b11
    m12 = a10 * b02 + a11 * b12 + a12

    return np.array([m00 - 1.0, m10, m01, m11 - 1.0, m02, m12], dtype=np.float64)
