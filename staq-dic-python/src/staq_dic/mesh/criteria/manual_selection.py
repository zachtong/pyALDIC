"""Manual element selection refinement criterion.

Marks user-specified elements for refinement. Element indices can be
provided at construction time or via ctx.user_marks at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..refinement import RefinementContext


@dataclass(frozen=True)
class ManualSelectionCriterion:
    """Refine user-specified elements.

    Element indices can come from:
        1. ``element_indices`` (set at construction), or
        2. ``ctx.user_marks`` (set at runtime via RefinementContext).

    If both are provided, they are combined (union).

    Attributes:
        element_indices: Fixed element indices to mark.
        min_element_size: Elements smaller than this are never marked.
    """

    element_indices: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    min_element_size: int = 1

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Return boolean mask with True for user-specified elements.

        Args:
            ctx: Refinement context containing mesh and optional user_marks.

        Returns:
            (n_elements,) boolean array. True = refine this element.
        """
        n_elem = ctx.mesh.elements_fem.shape[0]
        marks = np.zeros(n_elem, dtype=np.bool_)

        # Combine construction-time and runtime indices
        all_indices: list[NDArray[np.int64]] = []
        if len(self.element_indices) > 0:
            all_indices.append(self.element_indices)
        if ctx.user_marks is not None and len(ctx.user_marks) > 0:
            all_indices.append(ctx.user_marks)

        if not all_indices:
            return marks

        combined = np.unique(np.concatenate(all_indices))
        valid = combined[(combined >= 0) & (combined < n_elem)]
        marks[valid] = True
        return marks
