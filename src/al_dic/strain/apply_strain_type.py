"""Convert deformation gradient to chosen strain measure.

Port of MATLAB strain/apply_strain_type.m (Jin Yang, Caltech).

Converts the infinitesimal displacement gradient tensor to one of several
finite strain measures, then applies a world-coordinate sign convention
flip on the off-diagonal terms (F21, F12).

Supported strain types:
    0 -- Infinitesimal (engineering) strain: identity transform.
    1 -- Eulerian-Almansi strain: E = (I - F^{-T} F^{-1}) / 2.
    2 -- Green-Lagrangian strain: E = (F^T F - I) / 2.

Here F = I + grad(u), and grad(u) is taken on the reference grid, as DIC
measures it. For types 1 and 2 the result is a symmetric tensor: both
off-diagonal slots of the output hold the shear component.

MATLAB/Python differences:
    - The interleaved vector layout [F11, F21, F12, F22, ...] is
      identical in both versions.
    - World coordinate conversion flips signs of F21 and F12 (off-diagonal
      terms) because the y-axis is inverted between image coordinates
      (y-down) and world coordinates (y-up).
    - Types 1 and 2 do NOT follow the MATLAB formulas, which are not the
      measures they are named after. MATLAB type 2 has the quadratic terms
      negative (the Almansi form, applied to reference-grid gradients); it
      reports -theta^2 for a rigid rotation by theta. MATLAB type 1 is
      1/(1 - dudx) - 1 etc., which turns a gradient taken on the deformed
      grid into one on the reference grid; applied to reference-grid
      gradients it is not a strain measure (10 % stretch gives 0.111,
      Almansi is 0.0868). Both are implemented here from their definitions.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara


def _euler_almansi(
    dudx: NDArray[np.float64],
    dvdx: NDArray[np.float64],
    dudy: NDArray[np.float64],
    dvdy: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Euler-Almansi strain e = (I - F^-T F^-1) / 2 with F = I + grad(u).

    The gradients are the ones DIC measures: displacement differentiated on
    the reference grid. F^-1 is written out for the 2x2 case, so

        F^-1 = [[1 + dvdy, -dudy], [-dvdx, 1 + dudx]] / det F.

    Where det F <= 0 the material would be collapsed or turned inside out;
    that is not a deformation and the strain there is NaN.

    Returns:
        (e_xx, e_xy, e_yy), each of the shape of the inputs.
    """
    fxx = 1.0 + dudx
    fyy = 1.0 + dvdy
    det = fxx * fyy - dudy * dvdx
    valid = det > 0.0                              # False for NaN as well
    inv_det_sq = np.full(det.shape, np.nan)
    np.divide(1.0, det * det, out=inv_det_sq, where=valid)

    exx = 0.5 * (1.0 - (fyy * fyy + dvdx * dvdx) * inv_det_sq)
    eyy = 0.5 * (1.0 - (fxx * fxx + dudy * dudy) * inv_det_sq)
    exy = 0.5 * (dudy * fyy + dvdx * fxx) * inv_det_sq
    return exx, exy, eyy


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
        exx, exy, eyy = _euler_almansi(F[0::4], F[1::4], F[2::4], F[3::4])
        F_strain[0::4] = exx
        F_strain[3::4] = eyy
        F_strain[2::4] = exy
        F_strain[1::4] = exy
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
