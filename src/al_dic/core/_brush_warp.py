"""Warp the user-painted brush refinement mask to a new reference frame.

The brush refinement mask is painted on frame 0 (the first reference) in
pixel coordinates. When the AL-DIC pipeline switches to a later reference
frame K (incremental mode), we want the *same material points* — not the
same pixel locations — to remain inside the mesh-refinement zone.

This module provides ``warp_brush_mask_to_ref``: given the cumulative
node displacement field that maps frame 0 -> frame K (defined on K's mesh),
it returns the brush mask warped into K's image coordinates.

Pipeline:
    1. Densify the (sparse) node displacements to a full pixel grid using
       ``FieldInterpolator`` (Delaunay + linear), filling outside the
       convex hull with the nearest valid value so the warp degrades
       gracefully near image borders.
    2. Call ``warp_mask`` (iterative inverse mapping, already used by the
       deformed-config visualization) to produce the warped binary mask.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from al_dic.utils.interpolation import FieldInterpolator
from al_dic.utils.warp_mask import warp_mask


def warp_brush_mask_to_ref(
    brush_mask: NDArray[np.bool_],
    cumulative_U: NDArray[np.float64],
    coords: NDArray[np.float64],
    img_shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Warp a frame-0 brush mask into another reference frame.

    Parameters
    ----------
    brush_mask : (H, W) bool array
        Brush refinement mask painted in frame-0 pixel coordinates.
    cumulative_U : (2*N,) float64 array
        Accumulated displacement (frame 0 -> frame K) on K's mesh, stored
        as interleaved [u0, v0, u1, v1, ...] over the N nodes of mesh K.
    coords : (N, 2) float64 array
        [x, y] node coordinates of mesh K.
    img_shape : (H, W)
        Image dimensions in pixels.

    Returns
    -------
    warped : (H, W) bool array
        Brush mask in frame-K pixel coordinates.
    """
    h, w = img_shape
    coords = np.asarray(coords, dtype=np.float64)
    cumulative_U = np.asarray(cumulative_U, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coords must have shape (N, 2), got {coords.shape}"
        )
    n_nodes = coords.shape[0]
    if cumulative_U.shape != (2 * n_nodes,):
        raise ValueError(
            f"cumulative_U must have shape ({2 * n_nodes},), "
            f"got {cumulative_U.shape}"
        )

    u_node = cumulative_U[0::2]
    v_node = cumulative_U[1::2]

    # Densify the node displacements to a full pixel grid. Use linear
    # interpolation (cheap, monotone) and fall back to nearest-neighbor
    # outside the node convex hull so warp_mask doesn't see NaN.
    interp = FieldInterpolator(coords, method="linear")
    x_grid, y_grid = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
    )
    u_pix = interp.interpolate(u_node, x_grid, y_grid, fill_outside="nearest")
    v_pix = interp.interpolate(v_node, x_grid, y_grid, fill_outside="nearest")

    # Replace any residual NaNs with zero (e.g., when all nodes are NaN).
    u_pix = np.nan_to_num(u_pix, nan=0.0)
    v_pix = np.nan_to_num(v_pix, nan=0.0)

    warped_f64 = warp_mask(brush_mask.astype(np.float64), u_pix, v_pix)
    return warped_f64 > 0.5
