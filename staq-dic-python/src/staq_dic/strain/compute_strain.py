"""Strain computation router: FEM strain + smoothing + type conversion.

Port of MATLAB strain/compute_strain.m (Jin Yang, Caltech).

Orchestrates the strain computation pipeline:
    1. Compute raw deformation gradient via FEM (global_nodal_strain_fem)
       or plane fitting (comp_def_grad).
    2. Smooth the deformation gradient field (smooth_field_sparse).
    3. Convert to the selected strain measure (apply_strain_type).
    4. Compute derived quantities (principal strains, max shear, von Mises).

MATLAB/Python differences:
    - MATLAB supports methods 0-3; Python preserves this but recommends
      method 2 (plane fitting) or 3 (FEM nodal strain).
    - Python returns a StrainResult dataclass with all derived quantities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICMesh, DICPara, StrainResult
from ..utils.region_analysis import NodeRegionMap
from .apply_strain_type import apply_strain_type
from .comp_def_grad import comp_def_grad
from .nodal_strain_fem import global_nodal_strain_fem
from .smooth_field import smooth_field_sparse


def _compute_derived_strains(
    exx: NDArray[np.float64],
    exy: NDArray[np.float64],
    eyy: NDArray[np.float64],
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute principal strains, max shear, and von Mises from 2D tensor.

    Args:
        exx, exy, eyy: Strain components (n_nodes,).

    Returns:
        (principal_max, principal_min, maxshear, von_mises) each (n_nodes,).
    """
    # Principal strains from eigenvalues of 2D symmetric tensor
    avg = 0.5 * (exx + eyy)
    diff = 0.5 * (exx - eyy)
    R = np.sqrt(diff**2 + exy**2)

    principal_max = avg + R
    principal_min = avg - R
    maxshear = R  # max shear = (e1 - e2) / 2 = R

    # Von Mises equivalent strain (plane stress)
    von_mises = np.sqrt(
        exx**2 + eyy**2 - exx * eyy + 3.0 * exy**2
    )

    return principal_max, principal_min, maxshear, von_mises


def compute_strain(
    mesh: DICMesh,
    para: DICPara,
    U: NDArray[np.float64],
    node_region_map: NodeRegionMap,
) -> StrainResult:
    """Compute strain from displacement via FEM + smoothing + type conversion.

    Routes through the strain computation pipeline based on
    ``para.method_to_compute_strain``:
        - 0: Use ADMM deformation gradient directly (requires F input).
        - 2: Local weighted plane fitting (comp_def_grad).
        - 3: FEM nodal strain via shape function derivatives (recommended).

    After computing raw gradients, applies:
        1. Sparse Gaussian smoothing within connected regions.
        2. Strain type conversion (infinitesimal/Eulerian/Green-Lagrangian).
        3. Derived strain computation (principal, shear, von Mises).

    Args:
        mesh: DICMesh with coordinates and connectivity.
        para: DIC parameters.
        U: Displacement vector (2*n_nodes,), interleaved.
        node_region_map: Pre-computed node-to-region mapping.

    Returns:
        StrainResult with all strain components populated.
    """
    n_nodes = mesh.coordinates_fem.shape[0]
    method = para.method_to_compute_strain

    # --- Step 1: Compute raw deformation gradient ---
    if method == 0 or method == 1:
        # Method 0: use F directly (caller should provide); fallback to FEM
        # Method 1: legacy central difference; use FEM instead
        F_raw = global_nodal_strain_fem(mesh, para, U)
    elif method == 2:
        # Plane fitting with search radius
        F_raw = comp_def_grad(
            U, mesh.coordinates_fem, mesh.elements_fem,
            rad=para.strain_plane_fit_rad,
            mask=para.img_ref_mask,
        )
        # Fill any NaN from plane fitting
        from ..utils.outlier_detection import fill_nan_idw
        F_raw = fill_nan_idw(F_raw, mesh.coordinates_fem, n_components=4)
    elif method == 3:
        F_raw = global_nodal_strain_fem(mesh, para, U)
    else:
        F_raw = global_nodal_strain_fem(mesh, para, U)

    # --- Step 2: Smooth the strain field ---
    smoothness = para.strain_smoothness
    if smoothness > 0:
        factor = 500.0 * smoothness
        # Adaptive sigma: per-node based on local element spacing
        from .smooth_field import compute_node_local_spacing
        sigma = compute_node_local_spacing(
            mesh.coordinates_fem, mesh.elements_fem,
        ) * factor
        F_smooth = smooth_field_sparse(
            F_raw, mesh.coordinates_fem, sigma, node_region_map,
            n_components=4,
        )
    else:
        F_smooth = F_raw

    # --- Step 2b: Rotation from raw gradients (before strain-type conversion) ---
    # F_smooth layout per node: [F11, F21, F12, F22] = [du/dx, dv/dx, du/dy, dv/dy]
    # ω = (dv/dx - du/dy_image) / 2  (CCW positive in image coordinates, y down)
    rotation = (F_smooth[1::4] - F_smooth[2::4]) / 2.0

    # --- Step 3: Strain type conversion ---
    F_strain, F_strain_world = apply_strain_type(F_smooth, para)

    # --- Step 4: Extract components and compute derived quantities ---
    # World coordinates: F21 and F12 have flipped signs
    dudx = F_strain_world[0::4]
    dvdx = F_strain_world[1::4]
    dudy = F_strain_world[2::4]
    dvdy = F_strain_world[3::4]

    # Strain tensor (symmetric part of displacement gradient)
    exx = dudx
    eyy = dvdy
    exy = 0.5 * (dudy + dvdx)

    # Displacement in world coordinates
    disp_u = U[0::2] * para.um2px
    disp_v = -U[1::2] * para.um2px  # y-flip for world coords

    # Derived strains
    principal_max, principal_min, maxshear, von_mises = _compute_derived_strains(
        exx, exy, eyy,
    )

    return StrainResult(
        disp_u=disp_u,
        disp_v=disp_v,
        dudx=dudx,
        dvdx=dvdx,
        dudy=dudy,
        dvdy=dvdy,
        strain_exx=exx,
        strain_exy=exy,
        strain_eyy=eyy,
        strain_principal_max=principal_max,
        strain_principal_min=principal_min,
        strain_maxshear=maxshear,
        strain_von_mises=von_mises,
        strain_rotation=rotation,
    )
