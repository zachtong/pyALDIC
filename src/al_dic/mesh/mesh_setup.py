"""Initial uniform rectangular Q8 mesh generation.

Port of MATLAB mesh/mesh_setup.m.

Constructs the starting uniform mesh from DIC grid coordinates before
quadtree refinement. Uses Q8 (8-node quadrilateral) elements where
nodes 5-8 are midside nodes set to -1 in the uniform case (no hanging nodes).

MATLAB/Python differences:
    - MATLAB uses 1-based indices; we use 0-based.
    - Both use coordinates_fem columns as [x, y].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICMesh, DICPara


def mesh_setup(
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    para: DICPara,
) -> DICMesh:
    """Create an initial uniform rectangular Q8 mesh from grid coordinates.

    Port of MATLAB mesh_setup.m.

    Node ordering within each element (counter-clockwise):
        3 --- 2
        |     |
        0 --- 1

    Args:
        x0: 1D array of x-coordinates (sorted ascending), shape (nx,).
        y0: 1D array of y-coordinates (sorted ascending), shape (ny,).
        para: DICPara with img_size for world coordinate computation.

    Returns:
        DICMesh with coordinates_fem, elements_fem (0-based), and world coords.

    Raises:
        ValueError: If x0 or y0 have fewer than 2 points.
    """
    nx = len(x0)
    ny = len(y0)
    if nx < 2 or ny < 2:
        raise ValueError(f"Need >= 2 grid points in each direction (got nx={nx}, ny={ny}).")

    # Create 2D grid using 'ij' indexing: xx[i, j] = x0[i], yy[i, j] = y0[j]
    # This matches MATLAB's ndgrid convention used in mesh_setup.m
    xx, yy = np.meshgrid(x0, y0, indexing="ij")  # shape (nx, ny)

    # Flatten to coordinate array: node index = i * ny + j
    # This matches MATLAB's column-major flatten of transposed x, y
    coordinates_fem = np.column_stack([xx.ravel(), yy.ravel()])

    # Build Q4 element connectivity (0-based)
    # For element at grid position (i, j): i in [0, nx-2], j in [0, ny-2]
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()

    node0 = ii * ny + jj            # bottom-left
    node1 = (ii + 1) * ny + jj      # bottom-right
    node2 = (ii + 1) * ny + (jj + 1)  # top-right
    node3 = ii * ny + (jj + 1)      # top-left

    # Q8 format: 4 corners + 4 midside nodes (-1 = no midside node)
    # Convention: midside >= 0 means hanging node exists; -1 means Q4 edge
    n_elem = len(node0)
    elements_fem = np.full((n_elem, 8), -1, dtype=np.int64)
    elements_fem[:, 0] = node0
    elements_fem[:, 1] = node1
    elements_fem[:, 2] = node2
    elements_fem[:, 3] = node3
    # Columns 4-7 remain -1 (no midside/hanging nodes for uniform mesh)

    # World coordinates: flip y-axis
    img_h = para.img_size[0] if para.img_size[0] > 0 else int(np.max(yy)) + 1
    coordinates_fem_world = np.column_stack([
        coordinates_fem[:, 0],
        img_h - coordinates_fem[:, 1],
    ])

    return DICMesh(
        coordinates_fem=coordinates_fem,
        elements_fem=elements_fem,
        coordinates_fem_world=coordinates_fem_world,
        x0=x0.copy(),
        y0=y0.copy(),
        element_min_size=para.winsize_min,
    )
