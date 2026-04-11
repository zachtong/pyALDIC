"""ROI-edge refinement criterion.

Marks elements near the ROI outer boundary for refinement.  Two
conditions are OR'd together:

1. **Window proximity**: the element's bounding box, expanded by
   ``half_win`` pixels, overlaps the ROI outer background.
2. **Mesh periphery**: the element sits on the mesh boundary (has at
   least one unshared edge) AND expanding by the element's own size
   reaches the ROI outer background.

Condition 2 handles the common case where the mesh grid is not
centered within the mask, leaving an asymmetric gap that may exceed
``half_win`` on some sides.

Uses connected-component labeling to distinguish the outer background
(mask=0 regions connected to the image border) from internal holes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

from ..refinement import RefinementContext


def _compute_outer_region(mask: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Identify mask=0 pixels that belong to the ROI outer background.

    Uses connected-component labeling on the mask=0 region.  Components
    touching any image border are classified as "outer" (ROI background);
    the rest are internal holes.

    Args:
        mask: Binary mask (H, W). 1.0 = valid ROI, 0.0 = outside/hole.

    Returns:
        (H, W) boolean array.  True = pixel is in the outer background.
    """
    labels, n_labels = label(mask < 0.5)

    if n_labels == 0:
        return np.zeros(mask.shape, dtype=np.bool_)

    # Labels touching any image border are "outer"
    border_labels: set[int] = set()
    h, w = mask.shape
    for edge in [labels[0, :], labels[h - 1, :], labels[:, 0], labels[:, w - 1]]:
        border_labels.update(int(v) for v in edge if v > 0)

    if not border_labels:
        return np.zeros(mask.shape, dtype=np.bool_)

    return np.isin(labels, list(border_labels))


def _find_peripheral_elements(elems: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Detect elements on the mesh boundary (having at least one unshared edge).

    An edge is a pair of adjacent corner nodes.  Interior edges are shared
    by exactly two elements; boundary edges belong to one element only.

    Args:
        elems: (n_elem, 4+) element connectivity (uses columns 0-3).

    Returns:
        (n_elem,) boolean array.  True = element has a boundary edge.
    """
    n_elem = elems.shape[0]
    if n_elem == 0:
        return np.empty(0, dtype=np.bool_)

    # Count edge occurrences across all elements
    edge_count: Counter[tuple[int, int]] = Counter()
    elem_edge_list: list[list[tuple[int, int]]] = []

    for i in range(n_elem):
        corners = elems[i, :4]
        if np.any(corners < 0):
            elem_edge_list.append([])
            continue
        edges: list[tuple[int, int]] = []
        for j in range(4):
            a, b = int(corners[j]), int(corners[(j + 1) % 4])
            canonical = (min(a, b), max(a, b))
            edges.append(canonical)
            edge_count[canonical] += 1
        elem_edge_list.append(edges)

    # An element is peripheral if any of its edges appears only once
    peripheral = np.zeros(n_elem, dtype=np.bool_)
    for i, edges in enumerate(elem_edge_list):
        for e in edges:
            if edge_count[e] == 1:
                peripheral[i] = True
                break

    return peripheral


@dataclass(frozen=True)
class ROIEdgeCriterion:
    """Refine elements near the ROI outer boundary.

    Combines two detection strategies:

    1. **Window proximity** — expand element bbox by ``half_win`` and
       check overlap with the ROI outer background.
    2. **Mesh periphery** — for elements on the mesh boundary, expand
       by the element's own diagonal length to bridge asymmetric gaps
       between the mesh grid and the mask boundary.

    Both strategies use ``_compute_outer_region`` to ignore internal
    holes, ensuring only the ROI outer edge triggers refinement.

    Attributes:
        half_win: Half the IC-GN window size (pixels).
        min_element_size: Elements smaller than this are never marked.
    """

    half_win: int = 16
    min_element_size: int = 8

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Mark elements near the ROI outer edge.

        Args:
            ctx: Refinement context containing mesh and mask.

        Returns:
            (n_elements,) boolean array. True = refine this element.

        Raises:
            ValueError: If ctx.mask is None.
        """
        if ctx.mask is None:
            raise ValueError("ROIEdgeCriterion requires ctx.mask to be set")

        coords = ctx.mesh.coordinates_fem
        elems = ctx.mesh.elements_fem
        n_elem = elems.shape[0]

        if n_elem == 0:
            return np.empty(0, dtype=np.bool_)

        mask = ctx.mask
        h, w = mask.shape

        outer = _compute_outer_region(mask)

        # Corner coordinates
        corners = elems[:, :4]
        cx = coords[corners, 0]  # (n_elem, 4)
        cy = coords[corners, 1]  # (n_elem, 4)

        # Original bounding boxes
        x_min = np.clip(np.floor(cx.min(axis=1)).astype(np.int64), 0, w - 1)
        x_max = np.clip(np.ceil(cx.max(axis=1)).astype(np.int64), 0, w - 1)
        y_min = np.clip(np.floor(cy.min(axis=1)).astype(np.int64), 0, h - 1)
        y_max = np.clip(np.ceil(cy.max(axis=1)).astype(np.int64), 0, h - 1)

        # Element side lengths (used for peripheral expansion)
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Detect mesh-peripheral elements
        peripheral = _find_peripheral_elements(elems)

        # Per-element expansion: half_win for interior, diagonal for peripheral
        expansion = np.full(n_elem, self.half_win, dtype=np.int64)
        for i in range(n_elem):
            if peripheral[i]:
                diag = int(np.ceil(np.sqrt(
                    float(x_extent[i]) ** 2 + float(y_extent[i]) ** 2
                )))
                expansion[i] = max(self.half_win, diag)

        # Expanded bounding boxes
        ex_min = np.clip(x_min - expansion, 0, w - 1)
        ex_max = np.clip(x_max + expansion, 0, w - 1)
        ey_min = np.clip(y_min - expansion, 0, h - 1)
        ey_max = np.clip(y_max + expansion, 0, h - 1)

        # Check: expanded bbox overlaps outer region
        hits_outer = np.zeros(n_elem, dtype=np.bool_)
        for i in range(n_elem):
            patch = outer[ey_min[i] : ey_max[i] + 1, ex_min[i] : ex_max[i] + 1]
            if patch.size > 0 and patch.any():
                hits_outer[i] = True

        # Size check
        elem_size = np.minimum(x_extent, y_extent)
        return hits_outer & (elem_size > self.min_element_size)
