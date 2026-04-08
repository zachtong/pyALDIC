"""Convert deformation gradient to chosen strain measure.

Port of MATLAB strain/apply_strain_type.m (Jin Yang, Caltech).

Converts the infinitesimal displacement gradient tensor to one of several
finite strain measures, then applies a world-coordinate sign convention
flip on the off-diagonal terms (F21, F12).

Supported strain types:
    0 -- Infinitesimal (engineering) strain: identity transform.
    1 -- Eulerian-Almansi strain: E = (I - F^{-T} F^{-1}) / 2.
    2 -- Green-Lagrangian strain: E = (F^T F - I) / 2.

MATLAB/Python differences:
    - The interleaved vector layout [F11, F21, F12, F22, ...] is
      identical in both versions.
    - World coordinate conversion flips signs of F21 and F12 (off-diagonal
      terms) because the y-axis is inverted between image coordinates
      (y-down) and world coordinates (y-up).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara


def apply_strain_type(
    F: NDArray[np.float64],
    para: DICPara,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert displacement gradient to the selected strain measure.

    Args:
        F: Deformation gradient vector (4*n_nodes,), interleaved as
            [F11_0, F21_0, F12_0, F22_0, F11_1, ...] where
            F11 = du/dx, F21 = dv/dx, F12 = du/dy, F22 = dv/dy.
        para: DIC parameters.  Uses ``strain_type``:
            0 = infinitesimal, 1 = Eulerian-Almansi,
            2 = Green-Lagrangian.

    Returns:
        A tuple ``(F_strain, F_strain_world)`` where:
            - ``F_strain``: Strain in image coordinates (4*n_nodes,),
              same interleaved layout as input.
            - ``F_strain_world``: Strain in world coordinates
              (4*n_nodes,).  Identical to ``F_strain`` except
              F21 and F12 terms have flipped signs.
    """
    F_strain = F.copy()
    strain_type = para.strain_type

    if strain_type == 0:
        # Infinitesimal -- no conversion needed
        pass
    elif strain_type == 1:
        # Eulerian-Almansi
        dudx = F[0::4]
        dvdx = F[1::4]
        dudy = F[2::4]
        dvdy = F[3::4]
        F_strain[0::4] = 1.0 / (1.0 - dudx) - 1.0    # exx
        F_strain[3::4] = 1.0 / (1.0 - dvdy) - 1.0    # eyy
        F_strain[2::4] = dudy / (1.0 - dvdy)          # exy
        F_strain[1::4] = dvdx / (1.0 - dudx)          # eyx
    elif strain_type == 2:
        # Green-Lagrangian: E = (F_cm^T F_cm - I) / 2, where F_cm = I + grad(u)
        # Positive quadratic terms because fibers in the reference config stretch.
        dudx = F[0::4]
        dvdx = F[1::4]
        dudy = F[2::4]
        dvdy = F[3::4]
        F_strain[0::4] = dudx + 0.5 * (dudx**2 + dvdx**2)
        F_strain[3::4] = dvdy + 0.5 * (dudy**2 + dvdy**2)
        F_strain[2::4] = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
        F_strain[1::4] = 0.5 * (dvdx + dudy + dudy * dudx + dvdy * dvdx)
    else:
        warnings.warn(
            f"Unknown strain_type {strain_type}, using infinitesimal.",
            stacklevel=2,
        )

    # World coordinate conversion: flip off-diagonal signs (y-axis inversion)
    F_strain_world = F_strain.copy()
    F_strain_world[1::4] = -F_strain_world[1::4]  # F21
    F_strain_world[2::4] = -F_strain_world[2::4]  # F12

    return F_strain, F_strain_world
