"""FEM-based nodal strain computation.

Port of MATLAB strain/global_nodal_strain_fem.m (Jin Yang, Caltech).

Computes the deformation gradient (displacement gradient tensor) at each
FEM node by evaluating shape function derivatives at element centers,
then averaging to nodes weighted by element area.  This is the primary
strain computation method used in STAQ-DIC (MethodToComputeStrain = 3).

MATLAB/Python differences:
    - MATLAB uses ``compute_all_elements_gp`` at the element centroid
      (ksi=0, eta=0) for a single Gauss point evaluation.
    - Python reuses the same ``fem_assembly.compute_all_elements_gp``.
    - MATLAB ``rmoutliers('movmedian', ...)`` -> ``scipy.ndimage.median_filter``
      + MAD-based outlier detection.
    - MATLAB ``scatteredInterpolant`` -> ``fill_nan_idw`` from outlier_detection.
    - MATLAB 1-based indices -> Python 0-based.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from ..core.data_structures import DICMesh, DICPara
from ..solver.fem_assembly import compute_all_elements_gp
from ..utils.outlier_detection import fill_nan_idw


def _detect_outliers_movmedian(
    values: NDArray[np.float64],
    window_size: int,
) -> NDArray[np.bool_]:
    """Detect outliers via moving median + scaled MAD.

    Replicates MATLAB's ``rmoutliers('movmedian', window_size)``.

    Args:
        values: 1-D array of values.
        window_size: Window size for the moving median.

    Returns:
        Boolean mask, True for outlier indices.
    """
    if len(values) < 3:
        return np.zeros(len(values), dtype=np.bool_)

    # Moving median
    med = median_filter(values, size=window_size, mode="reflect")

    # Moving MAD (median absolute deviation)
    abs_dev = np.abs(values - med)
    mad = median_filter(abs_dev, size=window_size, mode="reflect")

    # Scaled MAD (consistent estimator for normal distribution)
    # c = -1 / (sqrt(2) * erfcinv(3/2)) ≈ 1.4826
    scaled_mad = 1.4826 * mad

    # Outlier: |x - median| > 3 * scaled_MAD
    threshold = 3.0 * scaled_mad
    # Avoid flagging everything when MAD is zero (constant regions)
    threshold[threshold < 1e-10] = np.inf

    return np.abs(values - med) > threshold


def global_nodal_strain_fem(
    mesh: DICMesh,
    para: DICPara,
    U: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute deformation gradient at each node via FEM shape functions.

    Evaluates displacement gradients (du/dx, du/dy, dv/dx, dv/dy) at each
    element center using the shape function derivative matrix DN, then
    area-weighted averages to nodes.

    Args:
        mesh: DICMesh with ``coordinates_fem`` and ``elements_fem``.
        para: DIC parameters.  Uses ``winstepsize`` for outlier window size.
        U: Displacement vector (2*n_nodes,), interleaved [u0,v0,...].

    Returns:
        Deformation gradient vector (4*n_nodes,), interleaved as
        [F11_0, F21_0, F12_0, F22_0, ...].
    """
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem
    n_nodes = coords.shape[0]
    n_ele = elems.shape[0]

    if n_ele == 0:
        return np.zeros(4 * n_nodes, dtype=np.float64)

    # --- Gather element node coordinates and delta flags ---
    ptx = np.zeros((n_ele, 8), dtype=np.float64)
    pty = np.zeros((n_ele, 8), dtype=np.float64)
    for k in range(8):
        valid = elems[:, k] >= 0
        ptx[valid, k] = coords[elems[valid, k], 0]
        pty[valid, k] = coords[elems[valid, k], 1]

    delta = (elems[:, 4:8] >= 0).astype(np.float64)

    # --- Build DOF index array with dummy node ---
    dummy_node = n_nodes
    all_index_u = np.zeros((n_ele, 16), dtype=np.int64)
    for k in range(8):
        node_ids = elems[:, k].copy()
        node_ids[node_ids < 0] = dummy_node
        all_index_u[:, 2 * k] = 2 * node_ids
        all_index_u[:, 2 * k + 1] = 2 * node_ids + 1

    # Pad U for dummy node
    U_ext = np.concatenate([U, np.zeros(2, dtype=np.float64)])
    U_ele = U_ext[all_index_u]  # (n_ele, 16)

    # --- Evaluate shape function derivatives at element center (ksi=0, eta=0) ---
    _, DN_all, Jdet = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, n_ele)
    # DN_all: (n_ele, 4, 16)
    # Row 0: du/dx (odd cols = u-DOF shape func x-derivatives)
    # Row 1: du/dy
    # Row 2: dv/dx
    # Row 3: dv/dy

    # Extract dN/dx and dN/dy for the 8 shape functions (u-DOF columns: 0,2,4,...,14)
    dNdx = DN_all[:, 0, 0::2]  # (n_ele, 8)
    dNdy = DN_all[:, 1, 0::2]  # (n_ele, 8)

    # Element u and v components
    u_ele = U_ele[:, 0::2]  # (n_ele, 8)
    v_ele = U_ele[:, 1::2]  # (n_ele, 8)

    # Displacement gradients at element center
    dudx_ele = np.sum(dNdx * u_ele, axis=1)  # F11
    dudy_ele = np.sum(dNdy * u_ele, axis=1)  # F12
    dvdx_ele = np.sum(dNdx * v_ele, axis=1)  # F21
    dvdy_ele = np.sum(dNdy * v_ele, axis=1)  # F22

    # --- Area-weighted averaging to nodes ---
    ele_area = np.abs(Jdet)  # element area proxy
    F_nodal = np.zeros((n_nodes, 4), dtype=np.float64)
    node_weight = np.zeros(n_nodes, dtype=np.float64)

    for k in range(8):
        node_ids = elems[:, k]
        valid = node_ids >= 0
        nids = node_ids[valid]
        w = ele_area[valid]

        np.add.at(F_nodal[:, 0], nids, w * dudx_ele[valid])
        np.add.at(F_nodal[:, 1], nids, w * dvdx_ele[valid])
        np.add.at(F_nodal[:, 2], nids, w * dudy_ele[valid])
        np.add.at(F_nodal[:, 3], nids, w * dvdy_ele[valid])
        np.add.at(node_weight, nids, w)

    # Divide by weight; orphan nodes get NaN
    orphan = node_weight < 1e-15
    node_weight[orphan] = 1.0  # avoid /0
    F_nodal /= node_weight[:, None]
    F_nodal[orphan, :] = np.nan

    # --- Interleave to [F11, F21, F12, F22] per node ---
    F_out = np.empty(4 * n_nodes, dtype=np.float64)
    F_out[0::4] = F_nodal[:, 0]
    F_out[1::4] = F_nodal[:, 1]
    F_out[2::4] = F_nodal[:, 2]
    F_out[3::4] = F_nodal[:, 3]

    # --- Outlier removal via moving median ---
    window = 1 + para.winstepsize
    outlier_f11 = _detect_outliers_movmedian(F_out[0::4], window)
    outlier_f21 = _detect_outliers_movmedian(F_out[1::4], window)
    outlier_f12 = _detect_outliers_movmedian(F_out[2::4], window)
    outlier_f22 = _detect_outliers_movmedian(F_out[3::4], window)

    # Set all 4 components to NaN for any flagged node
    outlier_any = outlier_f11 | outlier_f21 | outlier_f12 | outlier_f22
    outlier_idx = np.where(outlier_any)[0]
    for c in range(4):
        F_out[4 * outlier_idx + c] = np.nan

    # --- Fill NaN via scattered interpolation ---
    F_out = fill_nan_idw(F_out, coords, n_components=4)

    return F_out
