"""Mark quadrilateral elements on mask edges for refinement.

Port of MATLAB mesh/mark_edge.m.

Identifies elements whose bounding box spans both masked (0) and unmasked (1)
regions — i.e., elements that straddle a material boundary.

MATLAB/Python differences:
    - MATLAB uses transposed mask indexing: mask(x, y) with 1-based coords.
    - Python uses standard mask[y, x] with 0-based coords.
    - Coordinates in both: col 0 = x, col 1 = y.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mark_edge(
    coordinates: NDArray[np.float64],
    elements: NDArray[np.int64],
    mask: NDArray[np.float64],
    min_size: int,
) -> NDArray[np.bool_]:
    """Mark elements that straddle mask boundaries for quadtree refinement.

    An element is marked if its bounding box contains both masked-in and
    masked-out pixels (grayscale range > 0) AND the element size exceeds
    min_size.

    Args:
        coordinates: (n_nodes, 2) node coordinates, columns [x, y].
        elements: (n_elements, 4+) element connectivity (only corners used), 0-based.
        mask: (H, W) binary mask. 1.0 = valid, 0.0 = hole.
        min_size: Minimum element side length below which no refinement occurs.

    Returns:
        (n_elements,) boolean array. True = should be refined.
    """
    n_elem = elements.shape[0]
    if n_elem == 0:
        return np.empty(0, dtype=np.bool_)

    h, w = mask.shape

    # Get corner node coordinates for all elements
    corners = elements[:, :4]  # (n_elem, 4)
    cx = coordinates[corners, 0]  # (n_elem, 4) x-coords
    cy = coordinates[corners, 1]  # (n_elem, 4) y-coords

    x_min = np.floor(cx.min(axis=1)).astype(np.int64)
    x_max = np.ceil(cx.max(axis=1)).astype(np.int64)
    y_min = np.floor(cy.min(axis=1)).astype(np.int64)
    y_max = np.ceil(cy.max(axis=1)).astype(np.int64)

    # Clamp to image bounds
    x_min = np.clip(x_min, 0, w - 1)
    x_max = np.clip(x_max, 0, w - 1)
    y_min = np.clip(y_min, 0, h - 1)
    y_max = np.clip(y_max, 0, h - 1)

    # Element size = min of x-extent and y-extent
    elem_size = np.minimum(x_max - x_min, y_max - y_min)

    # Check grayscale range within each element's bounding box
    grayscale_range = np.zeros(n_elem, dtype=np.float64)
    for i in range(n_elem):
        patch = mask[y_min[i]:y_max[i] + 1, x_min[i]:x_max[i] + 1]
        if patch.size > 0:
            grayscale_range[i] = patch.max() - patch.min()

    return (grayscale_range > 0) & (elem_size > min_size)
