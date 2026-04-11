"""Mask-boundary refinement criterion.

Marks elements whose bounding box overlaps an **internal** mask hole
(mask=0 region NOT connected to the image border).  The outer ROI
boundary is deliberately excluded — use ``ROIEdgeCriterion`` for that.

Uses ``scipy.ndimage.label`` on the mask=0 regions to separate background
(touching image border) from interior holes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label as ndimage_label

from ..refinement import RefinementContext


def _build_hole_mask(mask: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return a boolean image that is True only at internal holes.

    Internal holes are mask=0 connected components that do NOT touch
    any image border.  The outer background (connected to the border)
    is excluded.
    """
    h, w = mask.shape
    zero_regions = mask < 0.5

    # Label connected components of the zero regions (4-connectivity)
    labeled, n_labels = ndimage_label(zero_regions)

    # Collect labels that touch any border row/column
    border_labels: set[int] = set()
    border_labels.update(labeled[0, :].tolist())      # top row
    border_labels.update(labeled[h - 1, :].tolist())   # bottom row
    border_labels.update(labeled[:, 0].tolist())        # left col
    border_labels.update(labeled[:, w - 1].tolist())    # right col
    border_labels.discard(0)  # 0 = mask=1, not a zero-region

    # Mark only labels that are NOT border-connected → internal holes
    hole_mask = np.zeros((h, w), dtype=np.bool_)
    for lbl in range(1, n_labels + 1):
        if lbl not in border_labels:
            hole_mask[labeled == lbl] = True

    return hole_mask


@dataclass(frozen=True)
class MaskBoundaryCriterion:
    """Refine elements that straddle **internal** mask hole boundaries.

    An element is marked when its bounding box overlaps at least one
    pixel that belongs to an internal hole (mask=0, not connected to
    the image border).

    The outer ROI boundary is deliberately ignored — that is the job
    of ``ROIEdgeCriterion``.

    Attributes:
        min_element_size: Elements smaller than this are never marked.
    """

    min_element_size: int = 8

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Mark elements overlapping internal hole boundaries.

        Returns:
            (n_elements,) boolean array.

        Raises:
            ValueError: If ctx.mask is None.
        """
        if ctx.mask is None:
            raise ValueError(
                "MaskBoundaryCriterion requires ctx.mask to be set"
            )

        coords = ctx.mesh.coordinates_fem
        elems = ctx.mesh.elements_fem
        mask = ctx.mask
        n_elem = elems.shape[0]

        if n_elem == 0:
            return np.empty(0, dtype=np.bool_)

        h, w = mask.shape

        # Build a binary image of internal holes only (excludes outer bg)
        hole_mask = _build_hole_mask(mask)

        # If there are no internal holes at all, nothing to mark
        if not hole_mask.any():
            return np.zeros(n_elem, dtype=np.bool_)

        # Extract corner node coordinates (first 4 columns of Q8 elements)
        corners = elems[:, :4]
        cx = coords[corners, 0]  # (n_elem, 4) x-coords
        cy = coords[corners, 1]  # (n_elem, 4) y-coords

        # Compute bounding boxes, clamped to image bounds
        x_min = np.clip(np.floor(cx.min(axis=1)).astype(np.int64), 0, w - 1)
        x_max = np.clip(np.ceil(cx.max(axis=1)).astype(np.int64), 0, w - 1)
        y_min = np.clip(np.floor(cy.min(axis=1)).astype(np.int64), 0, h - 1)
        y_max = np.clip(np.ceil(cy.max(axis=1)).astype(np.int64), 0, h - 1)

        # An element is marked if its bbox contains any hole pixel
        has_hole = np.zeros(n_elem, dtype=np.bool_)
        for i in range(n_elem):
            patch = hole_mask[y_min[i] : y_max[i] + 1, x_min[i] : x_max[i] + 1]
            if patch.size > 0 and patch.any():
                has_hole[i] = True

        # Element size = min of x-extent and y-extent
        elem_size = np.minimum(x_max - x_min, y_max - y_min)

        return has_hole & (elem_size > self.min_element_size)
