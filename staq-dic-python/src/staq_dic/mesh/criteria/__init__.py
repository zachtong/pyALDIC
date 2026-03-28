"""Built-in mesh refinement criteria."""
from .mask_boundary import MaskBoundaryCriterion
from .manual_selection import ManualSelectionCriterion
from .posterior_error import PosteriorErrorCriterion

__all__ = [
    "MaskBoundaryCriterion",
    "ManualSelectionCriterion",
    "PosteriorErrorCriterion",
]
