"""Node-to-region mapping for smoothing with disconnected domains.

Port of MATLAB mesh/precompute_node_regions.m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label


@dataclass(frozen=True)
class NodeRegionMap:
    """Mapping from FEM nodes to connected mask regions.

    Attributes:
        region_node_lists: List of arrays, each containing 0-based node
            indices belonging to that region.
        n_regions: Number of connected regions with >= 2 nodes.
    """

    region_node_lists: list[NDArray[np.int64]]
    n_regions: int


def precompute_node_regions(
    coordinates_fem: NDArray[np.float64],
    img_ref_mask: NDArray[np.float64],
    img_size: tuple[int, int],
    min_area: int = 20,
) -> NodeRegionMap:
    """Map each FEM node to its enclosing connected mask region.

    Port of MATLAB precompute_node_regions.m.

    Uses scipy.ndimage.label for connected component analysis (8-connectivity),
    then assigns each node to the region containing its pixel.

    Args:
        coordinates_fem: (n_nodes, 2) node coordinates. Col 0 = x, col 1 = y.
        img_ref_mask: Binary mask (H, W) — 1.0 = valid region.
        img_size: (height, width) of the image.
        min_area: Minimum region pixel count to consider.

    Returns:
        NodeRegionMap with per-region node index lists.
    """
    h, w = img_size
    mask_bool = img_ref_mask > 0.5

    # 8-connectivity labeling
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, n_labels = label(mask_bool, structure=structure)

    # Compute node pixel indices
    # coordinates_fem: col 0 = x (col index), col 1 = y (row index)
    node_x = np.clip(np.round(coordinates_fem[:, 0]).astype(np.int64), 0, w - 1)
    node_y = np.clip(np.round(coordinates_fem[:, 1]).astype(np.int64), 0, h - 1)
    node_labels = labeled[node_y, node_x]

    region_node_lists = []
    for lbl in range(1, n_labels + 1):
        # Check region area
        region_area = np.sum(labeled == lbl)
        if region_area <= min_area:
            continue

        nodes_in_region = np.where(node_labels == lbl)[0]
        if len(nodes_in_region) >= 2:
            region_node_lists.append(nodes_in_region)

    return NodeRegionMap(
        region_node_lists=region_node_lists,
        n_regions=len(region_node_lists),
    )
