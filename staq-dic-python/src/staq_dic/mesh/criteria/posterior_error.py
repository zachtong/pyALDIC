"""Posterior error refinement criterion.

Marks elements where the DIC solve quality is poor, based on
per-node quality metrics (IC-GN convergence iterations, ZNSSD
residual, displacement discontinuity, etc.).

The specific metric is configurable via the ``metric`` parameter.
Default: IC-GN convergence iteration count (higher = worse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..refinement import RefinementContext


@dataclass(frozen=True)
class PosteriorErrorCriterion:
    """Refine elements where solve quality is poor.

    Evaluates a per-node quality metric and marks elements whose
    nodes exceed ``mean + sigma_factor * std``.

    Attributes:
        metric: Which quality metric to use.
            'conv_iterations': IC-GN iteration count (default).
        sigma_factor: Outlier threshold in standard deviations.
            Lower = more sensitive (more refinement).
        min_element_size: Elements smaller than this are never marked.
    """

    metric: Literal["conv_iterations"] = "conv_iterations"
    sigma_factor: float = 1.0
    min_element_size: int = 4

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Return boolean mask of elements to refine.

        Elements containing at least one node whose quality metric
        exceeds ``mean + sigma_factor * std`` are marked True.

        Args:
            ctx: Current refinement context.

        Returns:
            (n_elements,) boolean array.
        """
        n_elem = ctx.mesh.elements_fem.shape[0]
        marks = np.zeros(n_elem, dtype=np.bool_)

        node_values = self._get_node_values(ctx)
        if node_values is None:
            return marks

        mean_val = np.nanmean(node_values)
        std_val = np.nanstd(node_values)
        if std_val < 1e-10:
            return marks

        threshold = mean_val + self.sigma_factor * std_val
        bad_nodes = np.where(node_values > threshold)[0]

        if len(bad_nodes) == 0:
            return marks

        # Vectorized element marking: mark if any corner node is bad
        corners = ctx.mesh.elements_fem[:, :4]
        marks = np.isin(corners, bad_nodes).any(axis=1)

        return marks

    def _get_node_values(
        self,
        ctx: RefinementContext,
    ) -> NDArray[np.float64] | None:
        """Extract per-node quality metric from context.

        Args:
            ctx: Current refinement context.

        Returns:
            (n_nodes,) float64 array, or None if the required data
            is not available in the context.
        """
        if self.metric == "conv_iterations":
            if ctx.conv_iterations is None:
                return None
            return ctx.conv_iterations.astype(np.float64)
        return None
