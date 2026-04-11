"""Identify elements inside mask holes.

Port of MATLAB mesh/mark_inside.m.

Classifies elements as "inside a hole" when >50% of their bounding box
pixels are masked out. Returns element indices for inside/outside classification.

MATLAB/Python differences:
    - MATLAB uses transposed mask: mask(x, y). Python: mask[y, x].
    - MATLAB returns 1-based indices. Python returns 0-based.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mark_inside(
    coordinates: NDArray[np.float64],
    elements: NDArray[np.int64],
    mask: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Classify elements as inside or outside mask holes.

    An element is "inside" when >50% of its bounding box pixels have
    mask value 0.

    Args:
        coordinates: (n_nodes, 2) node coordinates, columns [x, y].
        elements: (n_elements, 4+) element connectivity, 0-based.
        mask: (H, W) binary mask. 1.0 = valid, 0.0 = hole.

    Returns:
        (mark_inside_idx, mark_outside_idx): 0-based element indices.
    """
    n_elem = elements.shape[0]
    if n_elem == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    h, w = mask.shape
    is_inside = np.zeros(n_elem, dtype=np.bool_)

    corners = elements[:, :4]
    cx = coordinates[corners, 0]
    cy = coordinates[corners, 1]

    x_min = np.floor(cx.min(axis=1)).astype(np.int64)
    x_max = np.ceil(cx.max(axis=1)).astype(np.int64)
    y_min = np.floor(cy.min(axis=1)).astype(np.int64)
    y_max = np.ceil(cy.max(axis=1)).astype(np.int64)

    x_min = np.clip(x_min, 0, w - 1)
    x_max = np.clip(x_max, 0, w - 1)
    y_min = np.clip(y_min, 0, h - 1)
    y_max = np.clip(y_max, 0, h - 1)

    for i in range(n_elem):
        patch = mask[y_min[i]:y_max[i] + 1, x_min[i]:x_max[i] + 1]
        area = (x_max[i] - x_min[i]) * (y_max[i] - y_min[i])
        if area > 0 and patch.sum() < 0.5 * area:
            is_inside[i] = True

    inside_idx = np.where(is_inside)[0]
    outside_idx = np.where(~is_inside)[0]

    return inside_idx, outside_idx
