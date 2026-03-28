"""Mask-boundary refinement criterion.

Marks elements whose bounding box straddles a mask boundary (contains
both masked-in and masked-out pixels). Equivalent to the legacy
mark_edge() function wrapped in the RefinementCriterion interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..refinement import RefinementContext


@dataclass(frozen=True)
class MaskBoundaryCriterion:
    """Refine elements that straddle mask boundaries.

    An element is marked when its bounding box contains both mask=1
    and mask=0 pixels (grayscale range > 0).

    Attributes:
        min_element_size: Elements smaller than this are never marked.
    """

    min_element_size: int = 8

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Mark elements straddling mask boundaries.

        Returns a boolean array of shape (n_elements,). True means the
        element's bounding box spans a mask boundary AND the element
        is larger than ``min_element_size``.

        Args:
            ctx: Refinement context containing mesh and mask.

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

        # Extract corner node coordinates (first 4 columns of Q8 elements)
        corners = elems[:, :4]
        cx = coords[corners, 0]  # (n_elem, 4) x-coords
        cy = coords[corners, 1]  # (n_elem, 4) y-coords

        # Compute bounding boxes, clamped to image bounds
        x_min = np.clip(np.floor(cx.min(axis=1)).astype(np.int64), 0, w - 1)
        x_max = np.clip(np.ceil(cx.max(axis=1)).astype(np.int64), 0, w - 1)
        y_min = np.clip(np.floor(cy.min(axis=1)).astype(np.int64), 0, h - 1)
        y_max = np.clip(np.ceil(cy.max(axis=1)).astype(np.int64), 0, h - 1)

        # Check grayscale range within each element's bounding box
        grayscale_range = np.zeros(n_elem, dtype=np.float64)
        for i in range(n_elem):
            patch = mask[y_min[i] : y_max[i] + 1, x_min[i] : x_max[i] + 1]
            if patch.size > 0:
                grayscale_range[i] = patch.max() - patch.min()

        # Element size = min of x-extent and y-extent
        elem_size = np.minimum(x_max - x_min, y_max - y_min)

        # Mark elements that straddle a boundary AND exceed min size.
        # Note: _union_marks in refinement.py also enforces min_element_size,
        # but we apply it here so mark() is self-contained for standalone use.
        return (grayscale_range > 0) & (elem_size > self.min_element_size)
