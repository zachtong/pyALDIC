"""Brush-region refinement criterion.

Marks elements that overlap with a user-painted binary refinement mask.
The refinement mask has the same dimensions as the DIC image; pixels
set to 1.0 indicate regions where the user wants finer mesh resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..refinement import RefinementContext


@dataclass(frozen=True)
class BrushRegionCriterion:
    """Refine elements that overlap with a user-painted region.

    An element is marked when its bounding box contains at least one
    pixel where ``refinement_mask > 0.5``.

    Attributes:
        refinement_mask: Binary image (H, W). 1.0 = refine here.
        min_element_size: Elements smaller than this are never marked.
    """

    refinement_mask: NDArray[np.float64]
    min_element_size: int = 4

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Mark elements overlapping the painted region.

        Args:
            ctx: Refinement context containing mesh.

        Returns:
            (n_elements,) boolean array. True = refine this element.
        """
        coords = ctx.mesh.coordinates_fem
        elems = ctx.mesh.elements_fem
        n_elem = elems.shape[0]

        if n_elem == 0:
            return np.empty(0, dtype=np.bool_)

        h, w = self.refinement_mask.shape

        # Extract corner node coordinates (first 4 columns of Q8 elements)
        corners = elems[:, :4]
        cx = coords[corners, 0]  # (n_elem, 4)
        cy = coords[corners, 1]  # (n_elem, 4)

        # Bounding boxes, clamped to image bounds
        x_min = np.clip(np.floor(cx.min(axis=1)).astype(np.int64), 0, w - 1)
        x_max = np.clip(np.ceil(cx.max(axis=1)).astype(np.int64), 0, w - 1)
        y_min = np.clip(np.floor(cy.min(axis=1)).astype(np.int64), 0, h - 1)
        y_max = np.clip(np.ceil(cy.max(axis=1)).astype(np.int64), 0, h - 1)

        # Check if any pixel in bounding box is painted
        has_painted = np.zeros(n_elem, dtype=np.bool_)
        for i in range(n_elem):
            patch = self.refinement_mask[
                y_min[i] : y_max[i] + 1, x_min[i] : x_max[i] + 1
            ]
            if patch.size > 0 and patch.max() > 0.5:
                has_painted[i] = True

        # Size check: element must exceed min_element_size
        elem_size = np.minimum(x_max - x_min, y_max - y_min)
        return has_painted & (elem_size > self.min_element_size)
