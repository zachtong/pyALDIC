"""Adaptive refinement framework for quadtree DIC meshes.

Provides a Protocol-based criterion system for marking elements,
a Policy container for pre/post-solve refinement, and a ``refine_mesh``
driver that iteratively refines marked elements via ``qrefine_r``.

Design:
    - ``RefinementCriterion`` is a ``typing.Protocol`` — any class with
      ``min_element_size: int`` and ``mark(ctx) -> NDArray[bool_]`` qualifies.
    - ``RefinementContext`` bundles the mesh + optional solver state so
      criteria can inspect whatever they need.
    - ``RefinementPolicy`` groups pre-solve and post-solve criteria lists.
    - ``refine_mesh()`` is the main entry point: iteratively refine until
      no elements are marked (or all are below ``min_element_size``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICMesh, ImageGradients
from .generate_mesh import (
    _find_boundary_nodes,
    _inject_hanging_nodes,
    _interpolate_u0,
    _reorder_element_nodes_ccw,
)
from .mark_inside import mark_inside
from .qrefine_r import qrefine_r

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinementContext:
    """Immutable context passed to every ``RefinementCriterion.mark()`` call.

    Only ``mesh`` is required.  Optional fields carry solver state so
    post-solve criteria can inspect displacements, gradients, etc.

    Attributes:
        mesh: Current DICMesh (may be updated between refinement rounds).
        mask: Binary mask (H, W). 1.0 = valid, 0.0 = hole.
        Df: Reference image gradients.
        U: Displacement vector (2*n_nodes,) interleaved [u0,v0,...].
        F: Deformation gradient vector (4*n_nodes,).
        conv_iterations: Per-node IC-GN iteration count at convergence.
        user_marks: User-supplied element indices (int64) to refine.
    """

    mesh: DICMesh
    mask: NDArray[np.float64] | None = None
    Df: ImageGradients | None = None
    U: NDArray[np.float64] | None = None
    F: NDArray[np.float64] | None = None
    conv_iterations: NDArray[np.int64] | None = None
    user_marks: NDArray[np.int64] | None = None


@runtime_checkable
class RefinementCriterion(Protocol):
    """Protocol for mesh refinement criteria.

    Any class with a ``min_element_size`` attribute and a ``mark`` method
    that returns a boolean array over elements satisfies this protocol.
    """

    min_element_size: int

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Return a boolean mask of elements to refine.

        Args:
            ctx: Current refinement context.

        Returns:
            (n_elements,) boolean array. True = refine this element.
        """
        ...


@dataclass(frozen=True)
class RefinementPolicy:
    """Groups refinement criteria into pre-solve and post-solve phases.

    Attributes:
        pre_solve: Criteria applied before the DIC solve (e.g. mask boundary).
        post_solve: Criteria applied after the solve (e.g. posterior error).
        max_post_solve_cycles: Maximum number of post-solve refine-then-re-solve
            iterations.  0 disables post-solve refinement even if criteria exist.
    """

    pre_solve: list[RefinementCriterion] = field(default_factory=list)
    post_solve: list[RefinementCriterion] = field(default_factory=list)
    max_post_solve_cycles: int = 0

    @property
    def has_pre_solve(self) -> bool:
        """True if there are pre-solve criteria."""
        return len(self.pre_solve) > 0

    @property
    def has_post_solve(self) -> bool:
        """True if there are post-solve criteria AND cycles > 0."""
        return len(self.post_solve) > 0 and self.max_post_solve_cycles > 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_element_sizes(
    coords: NDArray[np.float64],
    elements: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute element diagonal sizes from corner nodes.

    Args:
        coords: (n_nodes, 2) node coordinates.
        elements: (n_elements, 4+) element connectivity (uses cols 0-3).

    Returns:
        (n_elements,) diagonal length for each element.
    """
    corners = elements[:, :4]
    dx = coords[corners[:, 0], 0] - coords[corners[:, 2], 0]
    dy = coords[corners[:, 0], 1] - coords[corners[:, 2], 1]
    return np.sqrt(dx**2 + dy**2)


def _union_marks(
    criteria: list[RefinementCriterion],
    ctx: RefinementContext,
) -> NDArray[np.bool_]:
    """Combine marks from multiple criteria by OR, respecting min_element_size.

    Each criterion's marks are AND'd with a size check: only elements
    whose diagonal > min_element_size * sqrt(2) can be marked by that
    criterion.  The final result is the OR of all per-criterion masks.

    Args:
        criteria: List of RefinementCriterion instances.
        ctx: Current refinement context.

    Returns:
        (n_elements,) boolean array.
    """
    n_elem = ctx.mesh.elements_fem.shape[0]
    if n_elem == 0 or len(criteria) == 0:
        return np.zeros(n_elem, dtype=np.bool_)

    # Precompute element sizes once
    sizes = _compute_element_sizes(
        ctx.mesh.coordinates_fem, ctx.mesh.elements_fem
    )

    combined = np.zeros(n_elem, dtype=np.bool_)
    for crit in criteria:
        marks = crit.mark(ctx)
        # AND with size check: element diagonal must exceed min_size * sqrt(2)
        size_ok = sizes > crit.min_element_size * np.sqrt(2)
        combined |= marks & size_ok

    return combined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def refine_mesh(
    mesh: DICMesh,
    criteria: list[RefinementCriterion],
    ctx: RefinementContext,
    U0: NDArray[np.float64],
    mask: NDArray[np.float64] | None = None,
    img_size: tuple[int, int] | None = None,
) -> tuple[DICMesh, NDArray[np.float64]]:
    """Refine a DIC mesh according to the given criteria.

    Iteratively applies ``_union_marks`` → ``qrefine_r`` until no elements
    are marked (or all remaining elements are below their criterion's
    ``min_element_size``).  After refinement, builds Q8 elements, removes
    inside-hole elements (if mask provided), and interpolates U0.

    Args:
        mesh: Starting DICMesh.
        criteria: List of refinement criteria.  Empty list → no refinement.
        ctx: RefinementContext for the criteria to inspect.
        U0: Displacement vector on the starting mesh, (2*n_nodes,).
        mask: Binary mask (H, W) for mark_inside. None = no hole removal.
        img_size: (height, width) needed for U0 interpolation when mask is set.

    Returns:
        (refined_mesh, U0_refined):
            - refined_mesh: DICMesh with Q8 elements.
            - U0_refined: Displacement interpolated onto the refined mesh.
    """
    if len(criteria) == 0:
        return mesh, U0.copy()

    # Work with Q4 corners during refinement
    coords = mesh.coordinates_fem.copy()
    elems_q4 = mesh.elements_fem[:, :4].copy()
    irregular = mesh.irregular.copy() if mesh.irregular.size > 0 else np.empty((0, 3), dtype=np.int64)

    n_nodes_start = coords.shape[0]
    n_elem_start = elems_q4.shape[0]

    # --- Iterative refinement loop ---
    any_refined = False
    while True:
        # Build a temporary mesh for context
        temp_mesh = DICMesh(
            coordinates_fem=coords,
            elements_fem=_to_q8_placeholder(elems_q4),
            irregular=irregular,
            x0=mesh.x0,
            y0=mesh.y0,
            element_min_size=mesh.element_min_size,
        )
        temp_ctx = RefinementContext(
            mesh=temp_mesh,
            mask=ctx.mask,
            Df=ctx.Df,
            U=ctx.U,
            F=ctx.F,
            conv_iterations=ctx.conv_iterations,
            user_marks=ctx.user_marks,
        )

        marks = _union_marks(criteria, temp_ctx)
        marked_idx = np.where(marks)[0]

        if len(marked_idx) == 0:
            break

        any_refined = True
        coords, elems_q4, irregular = qrefine_r(
            coords, elems_q4, irregular, marked_idx
        )

    # Early return if no refinement happened at all
    if not any_refined:
        if img_size is not None:
            # Recompute world coordinates using the canonical h+1-y formula
            # so callers that pass img_size get consistent output.
            h = img_size[0]
            coords_world = np.column_stack([
                mesh.coordinates_fem[:, 0],
                h + 1 - mesh.coordinates_fem[:, 1],
            ])
            mesh_out = DICMesh(
                coordinates_fem=mesh.coordinates_fem,
                elements_fem=mesh.elements_fem,
                irregular=mesh.irregular,
                mark_coord_hole_edge=mesh.mark_coord_hole_edge,
                coordinates_fem_world=coords_world,
                x0=mesh.x0,
                y0=mesh.y0,
                element_min_size=mesh.element_min_size,
            )
            return mesh_out, U0.copy()
        return mesh, U0.copy()

    # --- Post-refinement pipeline ---

    # Step 1: Reorder element nodes to CCW
    _reorder_element_nodes_ccw(coords, elems_q4)

    # Step 2: Build Q8 with hanging nodes
    elems_q8 = _inject_hanging_nodes(elems_q4, irregular)

    # Step 3: Remove elements inside holes (if mask provided)
    if mask is not None:
        _, outside_idx = mark_inside(coords, elems_q8, mask)
        elems_q8 = elems_q8[outside_idx]

    # Step 4: Find boundary nodes
    mark_coord_hole_edge = _find_boundary_nodes(
        coords, elems_q8, mesh.coordinates_fem, mesh.element_min_size * 2
    )

    # Step 5: Interpolate U0 (if mask + img_size provided)
    if mask is not None and img_size is not None:
        U0_refined = _interpolate_u0(
            mesh.coordinates_fem, coords, U0, mask, img_size
        )
    else:
        # No mask → simple zero-fill for new nodes
        n_new = coords.shape[0]
        U0_refined = np.zeros(2 * n_new, dtype=np.float64)
        # Copy existing displacements for nodes that survived
        n_copy = min(mesh.coordinates_fem.shape[0], n_new)
        U0_refined[: 2 * n_copy] = U0[: 2 * n_copy]

    # Step 6: Compute world coordinates
    if img_size is not None:
        # Use the same formula as the original generate_mesh:
        # world_y = img_height + 1 - pixel_y
        h = img_size[0]
        coords_world = np.column_stack([
            coords[:, 0],
            h + 1 - coords[:, 1],
        ])
    elif mesh.coordinates_fem_world is not None and mesh.coordinates_fem.shape[0] > 0:
        # Infer image height from original world coords
        # world_y = h_plus_one - pixel_y → h_plus_one = world_y + pixel_y
        h_plus_one = (
            mesh.coordinates_fem_world[0, 1] + mesh.coordinates_fem[0, 1]
        )
        coords_world = np.column_stack([
            coords[:, 0],
            h_plus_one - coords[:, 1],
        ])
    else:
        coords_world = None

    # Step 7: Assemble output mesh
    refined_mesh = DICMesh(
        coordinates_fem=coords,
        elements_fem=elems_q8,
        irregular=irregular,
        mark_coord_hole_edge=mark_coord_hole_edge,
        coordinates_fem_world=coords_world,
        x0=mesh.x0,
        y0=mesh.y0,
        element_min_size=mesh.element_min_size,
    )

    n_nodes_end = coords.shape[0]
    n_elem_end = elems_q8.shape[0]
    logger.info(
        "refine_mesh: %d→%d nodes, %d→%d elements",
        n_nodes_start,
        n_nodes_end,
        n_elem_start,
        n_elem_end,
    )

    return refined_mesh, U0_refined


def _to_q8_placeholder(elems_q4: NDArray[np.int64]) -> NDArray[np.int64]:
    """Pad Q4 elements to Q8 format with -1 midside columns.

    Args:
        elems_q4: (n_elements, 4) Q4 connectivity.

    Returns:
        (n_elements, 8) with columns 4-7 set to -1.
    """
    n_elem = elems_q4.shape[0]
    elems_q8 = np.full((n_elem, 8), -1, dtype=np.int64)
    elems_q8[:, :4] = elems_q4
    return elems_q8
