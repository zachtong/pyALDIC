# Adaptive Mesh Refinement Framework — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an extensible, strategy-pattern-based mesh refinement framework that integrates into the Python DIC pipeline, supporting independent pre-solve and post-solve refinement criteria with per-frame mesh independence.

**Architecture:** A `RefinementCriterion` protocol defines the marking interface. A `RefinementPolicy` groups pre-solve and post-solve criteria. The pipeline calls `refine_mesh()` at two hook points — after mesh_setup (pre-solve) and after ADMM (post-solve). Each frame independently evaluates its refinement policy. Built-in criteria: `MaskBoundaryCriterion` (port of MATLAB `mark_edge`), `ManualSelectionCriterion` (user-specified elements), `PosteriorErrorCriterion` (solve-quality-based). All criteria are independently toggleable; zero criteria = uniform mesh (backward compatible).

**Tech Stack:** Python 3.10+, NumPy, SciPy, dataclasses, Protocol (typing)

---

## Design Decisions (from user interview)

| Decision | Resolution |
|----------|-----------|
| Strategy independence | Zero, one, or many criteria; each independently toggleable |
| No criteria selected | Pure uniform mesh (backward compatible) |
| Same-stage combination | Union (any criterion marks → refine) |
| Cross-stage execution | Sequential: pre-solve → solve → post-solve |
| min_element_size | Per-criterion (not global) |
| Post-solve stop condition | No new marks OR max_cycles reached |
| Post-solve re-solve | Interpolate U/F to new nodes + full re-solve S4-S6 |
| Multi-frame mesh | Per-frame independent (B) — each frame decides its own refinement |
| Configuration | `run_aldic(..., refinement_policy=...)` parameter |
| Implementation order | Framework first, then one criterion at a time with tests |
| Post-solve loop | Pipeline-internal, automatic |

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/staq_dic/mesh/refinement.py` | **Create** | `RefinementCriterion`, `RefinementContext`, `RefinementPolicy`, `refine_mesh()` |
| `src/staq_dic/mesh/criteria/mask_boundary.py` | **Create** | `MaskBoundaryCriterion` — port of current `mark_edge` logic |
| `src/staq_dic/mesh/criteria/manual_selection.py` | **Create** | `ManualSelectionCriterion` — user-specified element indices |
| `src/staq_dic/mesh/criteria/posterior_error.py` | **Create** | `PosteriorErrorCriterion` — convergence/residual based |
| `src/staq_dic/mesh/criteria/__init__.py` | **Create** | Re-exports |
| `src/staq_dic/core/pipeline.py` | **Modify** | Add pre-solve and post-solve refinement hooks |
| `src/staq_dic/core/data_structures.py` | **Modify** | (No change to DICPara — policy is a run_aldic param) |
| `src/staq_dic/mesh/generate_mesh.py` | **Modify** | Extract `_post_refine_pipeline()` for reuse |
| `tests/test_mesh/test_refinement.py` | **Create** | Framework unit tests |
| `tests/test_mesh/test_criteria_mask.py` | **Create** | MaskBoundaryCriterion tests |
| `tests/test_mesh/test_criteria_manual.py` | **Create** | ManualSelectionCriterion tests |
| `tests/test_mesh/test_criteria_posterior.py` | **Create** | PosteriorErrorCriterion tests |
| `tests/test_integration/test_refinement_pipeline.py` | **Create** | End-to-end pipeline + refinement |

---

## Task 1: Framework core — Protocol, Context, Policy, refine_mesh()

**Files:**
- Create: `src/staq_dic/mesh/refinement.py`
- Test: `tests/test_mesh/test_refinement.py`

### Step 1: Write failing tests

```python
# tests/test_mesh/test_refinement.py
"""Tests for the adaptive refinement framework core."""
import numpy as np
import pytest

from staq_dic.mesh.refinement import (
    RefinementContext,
    RefinementCriterion,
    RefinementPolicy,
    refine_mesh,
)
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICPara, DICMesh


class _AlwaysRefineCriterion:
    """Test criterion that marks ALL elements above min_size."""
    def __init__(self, min_element_size: int = 4):
        self.min_element_size = min_element_size

    def mark(self, ctx: RefinementContext) -> np.ndarray:
        coords = ctx.mesh.coordinates_fem
        elems = ctx.mesh.elements_fem[:, :4]
        cx = coords[elems, 0]
        cy = coords[elems, 1]
        size = np.minimum(cx.max(axis=1) - cx.min(axis=1),
                          cy.max(axis=1) - cy.min(axis=1))
        return size > self.min_element_size


class _NeverRefineCriterion:
    """Test criterion that marks no elements."""
    def __init__(self, min_element_size: int = 1):
        self.min_element_size = min_element_size

    def mark(self, ctx: RefinementContext) -> np.ndarray:
        return np.zeros(ctx.mesh.elements_fem.shape[0], dtype=np.bool_)


class TestRefinementContext:
    def test_create_minimal(self):
        """Context only requires mesh."""
        mesh = DICMesh(
            coordinates_fem=np.array([[0, 0], [16, 0], [16, 16], [0, 16]], dtype=np.float64),
            elements_fem=np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64),
        )
        ctx = RefinementContext(mesh=mesh)
        assert ctx.mask is None
        assert ctx.U is None

    def test_create_with_all_fields(self):
        """Context accepts all optional fields."""
        mesh = DICMesh(
            coordinates_fem=np.zeros((4, 2)),
            elements_fem=np.zeros((1, 8), dtype=np.int64),
        )
        ctx = RefinementContext(
            mesh=mesh,
            mask=np.ones((64, 64)),
            U=np.zeros(8),
            F=np.zeros(16),
            conv_iterations=np.zeros(4),
            user_marks=np.array([0]),
        )
        assert ctx.mask is not None
        assert ctx.U is not None


class TestRefinementPolicy:
    def test_empty_policy(self):
        """Empty policy = no refinement."""
        policy = RefinementPolicy()
        assert len(policy.pre_solve) == 0
        assert len(policy.post_solve) == 0
        assert policy.max_post_solve_cycles == 0

    def test_pre_solve_only(self):
        policy = RefinementPolicy(
            pre_solve=[_NeverRefineCriterion()],
        )
        assert len(policy.pre_solve) == 1
        assert len(policy.post_solve) == 0

    def test_has_pre_solve(self):
        policy = RefinementPolicy(pre_solve=[_NeverRefineCriterion()])
        assert policy.has_pre_solve
        assert not policy.has_post_solve

    def test_has_post_solve(self):
        policy = RefinementPolicy(
            post_solve=[_NeverRefineCriterion()],
            max_post_solve_cycles=2,
        )
        assert not policy.has_pre_solve
        assert policy.has_post_solve

    def test_no_post_solve_if_zero_cycles(self):
        """Even with post_solve criteria, zero cycles means no post-solve."""
        policy = RefinementPolicy(
            post_solve=[_NeverRefineCriterion()],
            max_post_solve_cycles=0,
        )
        assert not policy.has_post_solve


class TestRefineMesh:
    @pytest.fixture
    def uniform_mesh_and_para(self):
        """Create a small 4x4 uniform mesh (9 nodes, 4 elements)."""
        para = DICPara(winstepsize=16, winsize=32, winsize_min=8)
        x0 = np.array([16.0, 32.0, 48.0])
        y0 = np.array([16.0, 32.0, 48.0])
        mesh = mesh_setup(x0, y0, para)
        return mesh, para

    def test_no_criteria_returns_unchanged(self, uniform_mesh_and_para):
        """refine_mesh with empty policy returns original mesh."""
        mesh, para = uniform_mesh_and_para
        policy = RefinementPolicy()
        ctx = RefinementContext(mesh=mesh)
        result_mesh, result_U0 = refine_mesh(mesh, policy.pre_solve, ctx, np.zeros(2 * mesh.coordinates_fem.shape[0]))
        assert result_mesh.coordinates_fem.shape == mesh.coordinates_fem.shape

    def test_never_refine_returns_unchanged(self, uniform_mesh_and_para):
        """NeverRefine criterion produces no changes."""
        mesh, para = uniform_mesh_and_para
        ctx = RefinementContext(mesh=mesh)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])
        result_mesh, result_U0 = refine_mesh(mesh, [_NeverRefineCriterion()], ctx, U0)
        assert result_mesh.coordinates_fem.shape == mesh.coordinates_fem.shape

    def test_always_refine_increases_nodes(self, uniform_mesh_and_para):
        """AlwaysRefine should produce more nodes than original."""
        mesh, para = uniform_mesh_and_para
        ctx = RefinementContext(mesh=mesh)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])
        result_mesh, result_U0 = refine_mesh(
            mesh, [_AlwaysRefineCriterion(min_element_size=4)], ctx, U0,
        )
        assert result_mesh.coordinates_fem.shape[0] > mesh.coordinates_fem.shape[0]
        assert len(result_U0) == 2 * result_mesh.coordinates_fem.shape[0]

    def test_union_of_two_criteria(self, uniform_mesh_and_para):
        """Two criteria: union should refine what either marks."""
        mesh, para = uniform_mesh_and_para
        ctx = RefinementContext(mesh=mesh)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])
        # NeverRefine + AlwaysRefine → same as AlwaysRefine alone
        result_both, _ = refine_mesh(
            mesh,
            [_NeverRefineCriterion(), _AlwaysRefineCriterion(min_element_size=4)],
            ctx, U0,
        )
        result_one, _ = refine_mesh(
            mesh, [_AlwaysRefineCriterion(min_element_size=4)], ctx, U0,
        )
        assert result_both.coordinates_fem.shape == result_one.coordinates_fem.shape

    def test_min_size_respected(self, uniform_mesh_and_para):
        """AlwaysRefine with large min_size should not refine."""
        mesh, para = uniform_mesh_and_para
        ctx = RefinementContext(mesh=mesh)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])
        # min_element_size=999 → no element is larger → no refinement
        result_mesh, _ = refine_mesh(
            mesh, [_AlwaysRefineCriterion(min_element_size=999)], ctx, U0,
        )
        assert result_mesh.coordinates_fem.shape == mesh.coordinates_fem.shape
```

### Step 2: Run tests to verify they fail

```bash
cd staq-dic-python && python -m pytest tests/test_mesh/test_refinement.py -v
```
Expected: FAIL — `ImportError: cannot import name 'RefinementContext' from 'staq_dic.mesh.refinement'`

### Step 3: Implement the framework

```python
# src/staq_dic/mesh/refinement.py
"""Extensible adaptive mesh refinement framework.

Provides a strategy-pattern-based system for mesh refinement in DIC:
    - RefinementCriterion: Protocol for marking elements to refine.
    - RefinementContext: Data container for all information a criterion might use.
    - RefinementPolicy: Groups pre-solve and post-solve criteria.
    - refine_mesh(): Executes criteria and applies quadtree refinement.

Each criterion is independently toggleable. Multiple criteria in the same
stage are combined by union (any marks → refine). Zero criteria = no
refinement (uniform mesh preserved).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICMesh, ImageGradients

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefinementContext:
    """All data a refinement criterion might need.

    Only ``mesh`` is required. All other fields are optional and
    populated by the pipeline at the appropriate stage.

    Attributes:
        mesh: Current DICMesh to evaluate.
        mask: (H, W) binary mask for the reference image.
        Df: Image gradients for the reference image.
        U: Displacement vector (2*n_nodes,) from the latest solve.
        F: Deformation gradient vector (4*n_nodes,) from the latest solve.
        conv_iterations: Per-node IC-GN iteration counts (n_nodes,).
        user_marks: User-specified element indices to refine.
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

    Each criterion independently decides which elements to refine.
    Must have a ``min_element_size`` attribute for the stopping condition.
    """

    min_element_size: int

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        """Return boolean array (n_elements,) indicating elements to refine.

        Args:
            ctx: Refinement context with mesh and optional solve data.

        Returns:
            Boolean mask over elements. True = should be refined.
        """
        ...


@dataclass(frozen=True)
class RefinementPolicy:
    """Controls when and how mesh refinement happens in the pipeline.

    Attributes:
        pre_solve: Criteria evaluated before IC-GN/ADMM (geometry-based).
        post_solve: Criteria evaluated after IC-GN/ADMM (error-based).
        max_post_solve_cycles: Maximum re-solve iterations for post-solve
            refinement. 0 = no post-solve refinement even if criteria exist.
    """

    pre_solve: list[RefinementCriterion] = field(default_factory=list)
    post_solve: list[RefinementCriterion] = field(default_factory=list)
    max_post_solve_cycles: int = 0

    @property
    def has_pre_solve(self) -> bool:
        """True if there are any pre-solve criteria."""
        return len(self.pre_solve) > 0

    @property
    def has_post_solve(self) -> bool:
        """True if there are post-solve criteria AND cycles > 0."""
        return len(self.post_solve) > 0 and self.max_post_solve_cycles > 0


def _union_marks(
    criteria: list[RefinementCriterion],
    ctx: RefinementContext,
) -> NDArray[np.bool_]:
    """Combine multiple criteria by union (OR).

    Returns a boolean mask where True means at least one criterion
    marked the element AND the element is larger than that criterion's
    min_element_size.
    """
    n_elem = ctx.mesh.elements_fem.shape[0]
    if n_elem == 0:
        return np.empty(0, dtype=np.bool_)

    combined = np.zeros(n_elem, dtype=np.bool_)

    # Precompute element sizes once
    coords = ctx.mesh.coordinates_fem
    corners = ctx.mesh.elements_fem[:, :4]
    cx = coords[corners, 0]
    cy = coords[corners, 1]
    elem_size = np.minimum(
        cx.max(axis=1) - cx.min(axis=1),
        cy.max(axis=1) - cy.min(axis=1),
    )

    for criterion in criteria:
        marks = criterion.mark(ctx)
        # Apply per-criterion min_element_size
        size_ok = elem_size > criterion.min_element_size
        combined |= (marks & size_ok)

    return combined


def refine_mesh(
    mesh: DICMesh,
    criteria: list[RefinementCriterion],
    ctx: RefinementContext,
    U0: NDArray[np.float64],
    mask: NDArray[np.float64] | None = None,
    img_size: tuple[int, int] | None = None,
) -> tuple[DICMesh, NDArray[np.float64]]:
    """Apply refinement criteria and execute quadtree refinement.

    Iteratively: (1) evaluate all criteria via union, (2) refine marked
    elements via qrefine_r, (3) repeat until no elements are marked.
    Then post-processes: CCW reorder, hanging node injection, hole removal,
    boundary node detection, U0 interpolation.

    Args:
        mesh: Current DICMesh (uniform or already partially refined).
        criteria: List of RefinementCriterion to evaluate (union).
        ctx: RefinementContext (updated each iteration with new mesh).
        U0: Displacement vector (2*n_nodes,) to interpolate to new mesh.
        mask: (H, W) binary mask for hole removal and U0 interpolation.
            If None, no hole removal is performed.
        img_size: (height, width) for coordinate transforms. Required if
            mask is provided.

    Returns:
        (refined_mesh, U0_refined): Refined DICMesh and interpolated U0.
        If no elements are marked, returns the original mesh and U0.
    """
    from .qrefine_r import qrefine_r
    from .mark_inside import mark_inside
    from .generate_mesh import (
        _reorder_element_nodes_ccw,
        _inject_hanging_nodes,
        _find_boundary_nodes,
        _interpolate_u0,
    )

    if len(criteria) == 0:
        return mesh, U0.copy()

    coords = mesh.coordinates_fem.copy()
    elems = mesh.elements_fem[:, :4].copy()
    irregular = mesh.irregular.copy() if mesh.irregular.size > 0 else np.empty((0, 3), dtype=np.int64)

    # --- Iterative refinement loop ---
    any_refined = False
    while True:
        iter_ctx = RefinementContext(
            mesh=DICMesh(
                coordinates_fem=coords,
                elements_fem=np.hstack([elems, np.full((elems.shape[0], 4), -1, dtype=np.int64)]),
                irregular=irregular,
                element_min_size=mesh.element_min_size,
            ),
            mask=ctx.mask,
            Df=ctx.Df,
            U=ctx.U,
            F=ctx.F,
            conv_iterations=ctx.conv_iterations,
            user_marks=ctx.user_marks,
        )

        combined_marks = _union_marks(criteria, iter_ctx)
        marked_idx = np.where(combined_marks)[0]

        if len(marked_idx) == 0:
            break

        any_refined = True
        coords, elems, irregular = qrefine_r(coords, elems, irregular, marked_idx)

    if not any_refined:
        return mesh, U0.copy()

    # --- Post-refinement pipeline (same as generate_mesh.py) ---

    # Reorder to CCW
    _reorder_element_nodes_ccw(coords, elems)

    # Build Q8 with hanging nodes
    elems_q8 = _inject_hanging_nodes(elems, irregular)

    # Remove elements inside holes
    if mask is not None:
        _, outside_idx = mark_inside(coords, elems_q8, mask)
        elems_q8 = elems_q8[outside_idx]

    # Find boundary-adjacent nodes
    mark_coord_hole_edge = _find_boundary_nodes(
        coords, elems_q8, mesh.coordinates_fem, mesh.element_min_size * 2,
    )

    # Interpolate U0
    if mask is not None and img_size is not None:
        U0_new = _interpolate_u0(mesh.coordinates_fem, coords, U0, mask, img_size)
    else:
        # No mask: zero-initialize new nodes
        U0_new = np.zeros(2 * coords.shape[0], dtype=np.float64)
        # Copy values for nodes that exist in both meshes
        n_old = mesh.coordinates_fem.shape[0]
        n_copy = min(n_old, coords.shape[0])
        U0_new[:2 * n_copy] = U0[:2 * n_copy]

    # Build world coordinates
    if img_size is not None:
        h = img_size[0]
        coords_world = np.column_stack([coords[:, 0], h + 1 - coords[:, 1]])
    else:
        coords_world = coords.copy()

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

    logger.info(
        "Refinement: %d -> %d nodes, %d -> %d elements",
        mesh.coordinates_fem.shape[0], coords.shape[0],
        mesh.elements_fem.shape[0], elems_q8.shape[0],
    )

    return refined_mesh, U0_new
```

### Step 4: Run tests to verify they pass

```bash
cd staq-dic-python && python -m pytest tests/test_mesh/test_refinement.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add src/staq_dic/mesh/refinement.py tests/test_mesh/test_refinement.py
git commit -m "feat: adaptive refinement framework — Protocol, Context, Policy, refine_mesh()"
```

---

## Task 2: Extract reusable helpers from generate_mesh.py

**Files:**
- Modify: `src/staq_dic/mesh/generate_mesh.py`

The functions `_reorder_element_nodes_ccw`, `_inject_hanging_nodes`, `_find_boundary_nodes`, `_interpolate_u0` are currently private in `generate_mesh.py`. Task 1's `refine_mesh()` imports them. We need to make them importable (but still underscore-prefixed to signal internal use).

### Step 1: No code change needed

Python allows importing underscore-prefixed names explicitly. The `from .generate_mesh import _reorder_element_nodes_ccw` in Task 1 already works. Verify:

```bash
cd staq-dic-python && python -c "from staq_dic.mesh.generate_mesh import _reorder_element_nodes_ccw; print('OK')"
```
Expected: `OK`

### Step 2: Update generate_mesh.py to use refine_mesh internally

Refactor `generate_mesh()` to delegate to `refine_mesh()` with a `MaskBoundaryCriterion` — but this depends on Task 3 (MaskBoundaryCriterion). **Defer to Task 4.**

### Step 3: Commit (if any changes)

No commit needed for this task — it's a verification step.

---

## Task 3: MaskBoundaryCriterion

**Files:**
- Create: `src/staq_dic/mesh/criteria/__init__.py`
- Create: `src/staq_dic/mesh/criteria/mask_boundary.py`
- Test: `tests/test_mesh/test_criteria_mask.py`

### Step 1: Write failing tests

```python
# tests/test_mesh/test_criteria_mask.py
"""Tests for MaskBoundaryCriterion."""
import numpy as np
import pytest

from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from staq_dic.mesh.refinement import RefinementContext, RefinementCriterion
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICPara, DICMesh


@pytest.fixture
def mesh_64x64():
    """4x4 uniform mesh on 64x64 image."""
    para = DICPara(winstepsize=16, winsize=32, winsize_min=4)
    x0 = np.arange(16, 64, 16, dtype=np.float64)
    y0 = np.arange(16, 64, 16, dtype=np.float64)
    return mesh_setup(x0, y0, para), para


class TestMaskBoundaryCriterion:
    def test_implements_protocol(self):
        """MaskBoundaryCriterion satisfies RefinementCriterion protocol."""
        criterion = MaskBoundaryCriterion(min_element_size=4)
        assert isinstance(criterion, RefinementCriterion)

    def test_solid_mask_no_marks(self, mesh_64x64):
        """Solid mask (all ones) → no elements marked."""
        mesh, para = mesh_64x64
        mask = np.ones((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_empty_mask_no_marks(self, mesh_64x64):
        """All-zero mask → no elements marked (all uniformly zero)."""
        mesh, para = mesh_64x64
        mask = np.zeros((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_center_hole_marks_boundary_elements(self, mesh_64x64):
        """Circular hole in center → marks elements straddling boundary."""
        mesh, para = mesh_64x64
        mask = np.ones((64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        mask[(xx - 32)**2 + (yy - 32)**2 < 15**2] = 0.0
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks = criterion.mark(ctx)
        assert marks.any(), "Should mark at least one element near hole"

    def test_min_size_prevents_refinement(self, mesh_64x64):
        """Large min_element_size prevents any marking."""
        mesh, para = mesh_64x64
        mask = np.ones((64, 64), dtype=np.float64)
        mask[20:40, 20:40] = 0.0
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=999)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_raises_without_mask(self, mesh_64x64):
        """Should raise ValueError if ctx.mask is None."""
        mesh, para = mesh_64x64
        ctx = RefinementContext(mesh=mesh, mask=None)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        with pytest.raises(ValueError, match="mask"):
            criterion.mark(ctx)

    def test_equivalent_to_mark_edge(self, mesh_64x64):
        """Output should match legacy mark_edge function."""
        mesh, para = mesh_64x64
        mask = np.ones((64, 64), dtype=np.float64)
        mask[10:30, 25:50] = 0.0  # rectangular hole
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks_new = criterion.mark(ctx)

        from staq_dic.mesh.mark_edge import mark_edge
        marks_old = mark_edge(
            mesh.coordinates_fem, mesh.elements_fem, mask, 4,
        )
        np.testing.assert_array_equal(marks_new, marks_old)
```

### Step 2: Run tests to verify they fail

```bash
cd staq-dic-python && python -m pytest tests/test_mesh/test_criteria_mask.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'staq_dic.mesh.criteria'`

### Step 3: Implement

```python
# src/staq_dic/mesh/criteria/__init__.py
"""Built-in mesh refinement criteria."""
from .mask_boundary import MaskBoundaryCriterion

__all__ = ["MaskBoundaryCriterion"]
```

```python
# src/staq_dic/mesh/criteria/mask_boundary.py
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
        corners = elems[:, :4]
        cx = coords[corners, 0]
        cy = coords[corners, 1]

        x_min = np.clip(np.floor(cx.min(axis=1)).astype(np.int64), 0, w - 1)
        x_max = np.clip(np.ceil(cx.max(axis=1)).astype(np.int64), 0, w - 1)
        y_min = np.clip(np.floor(cy.min(axis=1)).astype(np.int64), 0, h - 1)
        y_max = np.clip(np.ceil(cy.max(axis=1)).astype(np.int64), 0, h - 1)

        grayscale_range = np.zeros(n_elem, dtype=np.float64)
        for i in range(n_elem):
            patch = mask[y_min[i]:y_max[i] + 1, x_min[i]:x_max[i] + 1]
            if patch.size > 0:
                grayscale_range[i] = patch.max() - patch.min()

        # Note: min_element_size filtering is handled by refine_mesh._union_marks,
        # but we also apply it here so mark() is self-contained when used standalone.
        elem_size = np.minimum(x_max - x_min, y_max - y_min)
        return (grayscale_range > 0) & (elem_size > self.min_element_size)
```

### Step 4: Run tests

```bash
cd staq-dic-python && python -m pytest tests/test_mesh/test_criteria_mask.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add src/staq_dic/mesh/criteria/ tests/test_mesh/test_criteria_mask.py
git commit -m "feat: MaskBoundaryCriterion — mask-boundary refinement strategy"
```

---

## Task 4: Refactor generate_mesh to use refine_mesh + MaskBoundaryCriterion

**Files:**
- Modify: `src/staq_dic/mesh/generate_mesh.py:45-132`
- Test: existing `tests/test_mesh/test_generate_mesh.py` (must still pass)

### Step 1: Run existing tests to confirm baseline

```bash
cd staq-dic-python && python -m pytest tests/test_mesh/test_generate_mesh.py -v
```
Expected: All PASS

### Step 2: Refactor generate_mesh() to delegate to refine_mesh

Replace the internal while-loop with a call to `refine_mesh()`:

```python
def generate_mesh(
    mesh: DICMesh,
    para: DICPara,
    Df: ImageGradients,
    U0: NDArray[np.float64],
) -> tuple[DICMesh, NDArray[np.float64]]:
    """Generate an adaptive quadtree mesh from a uniform starting mesh.

    This is the legacy entry point. Internally delegates to refine_mesh()
    with MaskBoundaryCriterion for backward compatibility.
    """
    from .refinement import RefinementContext, refine_mesh
    from .criteria.mask_boundary import MaskBoundaryCriterion

    mask = para.img_ref_mask
    if mask is None:
        raise ValueError("DICPara.img_ref_mask must be set for quadtree mesh generation")

    criterion = MaskBoundaryCriterion(min_element_size=mesh.element_min_size)
    ctx = RefinementContext(mesh=mesh, mask=mask, Df=Df)

    return refine_mesh(
        mesh, [criterion], ctx, U0,
        mask=mask, img_size=Df.img_size,
    )
```

### Step 3: Run existing tests — must still pass

```bash
cd staq-dic-python && python -m pytest tests/test_mesh/test_generate_mesh.py -v
```
Expected: All PASS (behavior identical)

### Step 4: Commit

```bash
git add src/staq_dic/mesh/generate_mesh.py
git commit -m "refactor: generate_mesh delegates to refine_mesh + MaskBoundaryCriterion"
```

---

## Task 5: Integrate pre-solve refinement into pipeline.py

**Files:**
- Modify: `src/staq_dic/core/pipeline.py:341-350` (run_aldic signature)
- Modify: `src/staq_dic/core/pipeline.py:504-591` (Section 3)
- Test: `tests/test_integration/test_refinement_pipeline.py`

### Step 1: Write failing test

```python
# tests/test_integration/test_refinement_pipeline.py
"""End-to-end tests for pipeline + adaptive refinement."""
import numpy as np
import pytest

from staq_dic.core.pipeline import run_aldic
from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICPara, GridxyROIRange
from staq_dic.mesh.refinement import RefinementPolicy
from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion


def _make_speckle(h, w, seed=42):
    """Generate synthetic speckle image."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w))
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=1.5)


class TestPreSolveRefinement:
    def test_no_policy_uses_uniform_mesh(self):
        """Without refinement_policy, pipeline uses uniform mesh."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)  # zero displacement
        mask = np.ones((h, w), dtype=np.float64)
        para = dicpara_default(DICPara(
            winstepsize=16, winsize=32, winsize_min=8,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        ))
        result = run_aldic(para, [ref, defm], [mask, mask],
                          compute_strain=False, refinement_policy=None)
        # Uniform mesh: all elements same size
        mesh = result.result_fe_mesh_each_frame[0]
        corners = mesh.elements_fem[:, :4]
        sizes = (mesh.coordinates_fem[corners[:, 2], 0] -
                 mesh.coordinates_fem[corners[:, 0], 0])
        assert np.all(np.abs(sizes - sizes[0]) < 1e-6), "Should be uniform"

    def test_mask_criterion_refines_near_hole(self):
        """MaskBoundaryCriterion should produce non-uniform mesh."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)
        mask = np.ones((h, w), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]
        mask[(xx - 64)**2 + (yy - 64)**2 < 25**2] = 0.0
        para = dicpara_default(DICPara(
            winstepsize=16, winsize=32, winsize_min=4,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        ))
        policy = RefinementPolicy(
            pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
        )
        result = run_aldic(para, [ref, defm], [mask, mask],
                          compute_strain=False, refinement_policy=policy)
        mesh = result.result_fe_mesh_each_frame[0]
        corners = mesh.elements_fem[:, :4]
        sizes = (mesh.coordinates_fem[corners[:, 2], 0] -
                 mesh.coordinates_fem[corners[:, 0], 0])
        # Non-uniform: at least two different element sizes
        assert len(np.unique(np.round(sizes))) > 1, "Should be non-uniform near hole"

    def test_backward_compatible_no_policy(self):
        """Omitting refinement_policy produces same result as before."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)
        mask = np.ones((h, w), dtype=np.float64)
        para = dicpara_default(DICPara(
            winstepsize=16, winsize=32, winsize_min=8,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        ))
        r1 = run_aldic(para, [ref, defm], [mask, mask], compute_strain=False)
        r2 = run_aldic(para, [ref, defm], [mask, mask], compute_strain=False,
                      refinement_policy=None)
        np.testing.assert_allclose(r1.result_disp[0].U, r2.result_disp[0].U)
```

### Step 2: Run tests to verify they fail

```bash
cd staq-dic-python && python -m pytest tests/test_integration/test_refinement_pipeline.py -v
```
Expected: FAIL — `TypeError: run_aldic() got an unexpected keyword argument 'refinement_policy'`

### Step 3: Implement pipeline changes

In `pipeline.py`, modify `run_aldic` signature (line ~341):

```python
from ..mesh.refinement import RefinementContext, RefinementPolicy, refine_mesh

def run_aldic(
    para: DICPara,
    images: list[NDArray[np.float64]],
    masks: list[NDArray[np.float64]],
    progress_fn: Callable[[float, str], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    compute_strain: bool = True,
    mesh: DICMesh | None = None,
    U0: NDArray[np.float64] | None = None,
    refinement_policy: RefinementPolicy | None = None,
) -> PipelineResult:
```

In Section 3 (after `mesh_setup` and mask NaN application, ~line 542), add:

```python
            # --- Pre-solve refinement ---
            if refinement_policy is not None and refinement_policy.has_pre_solve:
                logger.info("Applying pre-solve refinement (%d criteria)...",
                           len(refinement_policy.pre_solve))
                ref_ctx = RefinementContext(
                    mesh=dic_mesh, mask=f_mask, Df=Df,
                )
                dic_mesh, current_U0 = refine_mesh(
                    dic_mesh, refinement_policy.pre_solve, ref_ctx, current_U0,
                    mask=f_mask, img_size=(img_h, img_w),
                )
                n_nodes = dic_mesh.coordinates_fem.shape[0]
                progress(frac, f"Frame {frame_idx + 1}: refined to {n_nodes} nodes")
```

Also add the import at top of pipeline.py:

```python
from ..mesh.refinement import RefinementContext, RefinementPolicy, refine_mesh
```

### Step 4: Run tests

```bash
cd staq-dic-python && python -m pytest tests/test_integration/test_refinement_pipeline.py -v
```
Expected: All PASS

### Step 5: Run full test suite for regression

```bash
cd staq-dic-python && python -m pytest tests/ -v --timeout=120
```
Expected: All existing tests still PASS

### Step 6: Commit

```bash
git add src/staq_dic/core/pipeline.py tests/test_integration/test_refinement_pipeline.py
git commit -m "feat: integrate pre-solve refinement into pipeline via RefinementPolicy"
```

---

## Task 6: ManualSelectionCriterion

**Files:**
- Create: `src/staq_dic/mesh/criteria/manual_selection.py`
- Modify: `src/staq_dic/mesh/criteria/__init__.py`
- Test: `tests/test_mesh/test_criteria_manual.py`

### Step 1: Write failing tests

```python
# tests/test_mesh/test_criteria_manual.py
"""Tests for ManualSelectionCriterion."""
import numpy as np
import pytest

from staq_dic.mesh.criteria.manual_selection import ManualSelectionCriterion
from staq_dic.mesh.refinement import RefinementContext, RefinementCriterion
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICPara


@pytest.fixture
def mesh_4x4():
    para = DICPara(winstepsize=16, winsize=32, winsize_min=4)
    x0 = np.arange(16, 64, 16, dtype=np.float64)
    y0 = np.arange(16, 64, 16, dtype=np.float64)
    return mesh_setup(x0, y0, para)


class TestManualSelectionCriterion:
    def test_implements_protocol(self):
        criterion = ManualSelectionCriterion(element_indices=np.array([0, 1]))
        assert isinstance(criterion, RefinementCriterion)

    def test_marks_specified_elements(self, mesh_4x4):
        n_elem = mesh_4x4.elements_fem.shape[0]
        selected = np.array([0, 2])
        criterion = ManualSelectionCriterion(element_indices=selected)
        ctx = RefinementContext(mesh=mesh_4x4)
        marks = criterion.mark(ctx)
        assert marks[0] and marks[2]
        assert not marks[1]
        if n_elem > 3:
            assert not marks[3]

    def test_empty_selection_no_marks(self, mesh_4x4):
        criterion = ManualSelectionCriterion(element_indices=np.array([], dtype=np.int64))
        ctx = RefinementContext(mesh=mesh_4x4)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_from_user_marks_in_context(self, mesh_4x4):
        """Can also read from ctx.user_marks."""
        criterion = ManualSelectionCriterion()
        ctx = RefinementContext(
            mesh=mesh_4x4,
            user_marks=np.array([1, 3]),
        )
        marks = criterion.mark(ctx)
        assert marks[1] and marks[3]
        assert not marks[0]

    def test_out_of_range_ignored(self, mesh_4x4):
        """Element indices beyond n_elements are silently ignored."""
        n_elem = mesh_4x4.elements_fem.shape[0]
        criterion = ManualSelectionCriterion(element_indices=np.array([0, 9999]))
        ctx = RefinementContext(mesh=mesh_4x4)
        marks = criterion.mark(ctx)
        assert marks[0]
        assert marks.sum() == 1
```

### Step 2: Run tests — should fail

### Step 3: Implement

```python
# src/staq_dic/mesh/criteria/manual_selection.py
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
        n_elem = ctx.mesh.elements_fem.shape[0]
        marks = np.zeros(n_elem, dtype=np.bool_)

        # Combine construction-time and runtime indices
        all_indices = []
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
```

Update `__init__.py`:
```python
from .mask_boundary import MaskBoundaryCriterion
from .manual_selection import ManualSelectionCriterion

__all__ = ["MaskBoundaryCriterion", "ManualSelectionCriterion"]
```

### Step 4: Run tests — all PASS

### Step 5: Commit

```bash
git add src/staq_dic/mesh/criteria/manual_selection.py src/staq_dic/mesh/criteria/__init__.py tests/test_mesh/test_criteria_manual.py
git commit -m "feat: ManualSelectionCriterion — user-specified element refinement"
```

---

## Task 7: PosteriorErrorCriterion (stub + framework)

**Files:**
- Create: `src/staq_dic/mesh/criteria/posterior_error.py`
- Modify: `src/staq_dic/mesh/criteria/__init__.py`
- Test: `tests/test_mesh/test_criteria_posterior.py`

**Note:** The exact error metric will be investigated later. This task implements the framework with a configurable metric, defaulting to IC-GN convergence iterations.

### Step 1: Write failing tests

```python
# tests/test_mesh/test_criteria_posterior.py
"""Tests for PosteriorErrorCriterion."""
import numpy as np
import pytest

from staq_dic.mesh.criteria.posterior_error import PosteriorErrorCriterion
from staq_dic.mesh.refinement import RefinementContext, RefinementCriterion
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICPara


@pytest.fixture
def mesh_4x4():
    para = DICPara(winstepsize=16, winsize=32, winsize_min=4)
    x0 = np.arange(16, 64, 16, dtype=np.float64)
    y0 = np.arange(16, 64, 16, dtype=np.float64)
    return mesh_setup(x0, y0, para)


class TestPosteriorErrorCriterion:
    def test_implements_protocol(self):
        criterion = PosteriorErrorCriterion()
        assert isinstance(criterion, RefinementCriterion)

    def test_no_data_no_marks(self, mesh_4x4):
        """Without U/conv_iterations, marks nothing."""
        ctx = RefinementContext(mesh=mesh_4x4)
        criterion = PosteriorErrorCriterion()
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_uniform_convergence_no_marks(self, mesh_4x4):
        """All nodes converge equally → no outliers → no marks."""
        n_nodes = mesh_4x4.coordinates_fem.shape[0]
        conv = np.full(n_nodes, 5, dtype=np.int64)  # all 5 iterations
        ctx = RefinementContext(mesh=mesh_4x4, conv_iterations=conv)
        criterion = PosteriorErrorCriterion(sigma_factor=1.0)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_high_iteration_nodes_get_marked(self, mesh_4x4):
        """Nodes with high IC-GN iterations → their elements get marked."""
        n_nodes = mesh_4x4.coordinates_fem.shape[0]
        conv = np.full(n_nodes, 5, dtype=np.int64)
        # Make first 2 nodes converge slowly
        conv[0] = 50
        conv[1] = 45
        ctx = RefinementContext(mesh=mesh_4x4, conv_iterations=conv)
        criterion = PosteriorErrorCriterion(sigma_factor=1.0)
        marks = criterion.mark(ctx)
        assert marks.any(), "Should mark elements containing slow-converging nodes"

    def test_sigma_factor_controls_sensitivity(self, mesh_4x4):
        """Higher sigma_factor → fewer marks (less sensitive)."""
        n_nodes = mesh_4x4.coordinates_fem.shape[0]
        conv = np.full(n_nodes, 5, dtype=np.int64)
        conv[0] = 20  # moderately slow
        ctx = RefinementContext(mesh=mesh_4x4, conv_iterations=conv)

        strict = PosteriorErrorCriterion(sigma_factor=0.5)
        lenient = PosteriorErrorCriterion(sigma_factor=3.0)
        marks_strict = strict.mark(ctx)
        marks_lenient = lenient.mark(ctx)

        assert marks_strict.sum() >= marks_lenient.sum()
```

### Step 2: Run tests — should fail

### Step 3: Implement

```python
# src/staq_dic/mesh/criteria/posterior_error.py
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
        n_elem = ctx.mesh.elements_fem.shape[0]
        marks = np.zeros(n_elem, dtype=np.bool_)

        # Get per-node quality values
        node_values = self._get_node_values(ctx)
        if node_values is None:
            return marks

        # Statistical outlier detection: value > mean + sigma * std
        mean_val = np.nanmean(node_values)
        std_val = np.nanstd(node_values)
        if std_val < 1e-10:
            return marks  # no variation → no outliers

        threshold = mean_val + self.sigma_factor * std_val
        bad_nodes = np.where(node_values > threshold)[0]

        if len(bad_nodes) == 0:
            return marks

        # Mark elements that contain any bad node
        bad_set = set(bad_nodes)
        corners = ctx.mesh.elements_fem[:, :4]
        for i in range(n_elem):
            if any(int(n) in bad_set for n in corners[i]):
                marks[i] = True

        return marks

    def _get_node_values(
        self, ctx: RefinementContext,
    ) -> NDArray[np.float64] | None:
        """Extract per-node quality metric from context."""
        if self.metric == "conv_iterations":
            if ctx.conv_iterations is None:
                return None
            return ctx.conv_iterations.astype(np.float64)
        return None
```

Update `__init__.py`:
```python
from .mask_boundary import MaskBoundaryCriterion
from .manual_selection import ManualSelectionCriterion
from .posterior_error import PosteriorErrorCriterion

__all__ = [
    "MaskBoundaryCriterion",
    "ManualSelectionCriterion",
    "PosteriorErrorCriterion",
]
```

### Step 4: Run tests — all PASS

### Step 5: Commit

```bash
git add src/staq_dic/mesh/criteria/posterior_error.py src/staq_dic/mesh/criteria/__init__.py tests/test_mesh/test_criteria_posterior.py
git commit -m "feat: PosteriorErrorCriterion — convergence-based post-solve refinement"
```

---

## Task 8: Integrate post-solve refinement loop into pipeline

**Files:**
- Modify: `src/staq_dic/core/pipeline.py` (after Section 6, before cumulative)
- Modify: `src/staq_dic/solver/local_icgn.py` (return conv_iterations)
- Test: `tests/test_integration/test_refinement_pipeline.py` (add post-solve tests)

### Step 1: Write failing test

Add to `tests/test_integration/test_refinement_pipeline.py`:

```python
class TestPostSolveRefinement:
    def test_post_solve_refines_on_poor_convergence(self):
        """Post-solve criterion should trigger re-solve with finer mesh."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        # Large displacement to create convergence difficulty
        from scipy.ndimage import shift
        defm = shift(ref, [0, 8.0], order=3, mode='reflect')
        mask = np.ones((h, w), dtype=np.float64)
        para = dicpara_default(DICPara(
            winstepsize=16, winsize=32, winsize_min=4,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
            size_of_fft_search_region=15,
        ))
        policy = RefinementPolicy(
            post_solve=[PosteriorErrorCriterion(sigma_factor=0.5, min_element_size=4)],
            max_post_solve_cycles=1,
        )
        result = run_aldic(para, [ref, defm], [mask, mask],
                          compute_strain=False, refinement_policy=policy)
        # Pipeline should complete without error
        assert result.result_disp[0] is not None

    def test_zero_cycles_skips_post_solve(self):
        """max_post_solve_cycles=0 should skip post-solve even with criteria."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)
        mask = np.ones((h, w), dtype=np.float64)
        para = dicpara_default(DICPara(
            winstepsize=16, winsize=32, winsize_min=4,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        ))
        policy = RefinementPolicy(
            post_solve=[PosteriorErrorCriterion()],
            max_post_solve_cycles=0,  # skip!
        )
        r1 = run_aldic(para, [ref, defm], [mask, mask],
                       compute_strain=False, refinement_policy=None)
        r2 = run_aldic(para, [ref, defm], [mask, mask],
                       compute_strain=False, refinement_policy=policy)
        # Same result since post-solve is skipped
        np.testing.assert_allclose(r1.result_disp[0].U, r2.result_disp[0].U)
```

### Step 2: Implement post-solve hook in pipeline

After Section 6 / local-only result (line ~786), before storing frame results, add:

```python
        # =============================================================
        # Post-solve refinement loop
        # =============================================================
        if (
            refinement_policy is not None
            and refinement_policy.has_post_solve
        ):
            for cycle in range(refinement_policy.max_post_solve_cycles):
                logger.info(
                    "Post-solve refinement cycle %d/%d",
                    cycle + 1, refinement_policy.max_post_solve_cycles,
                )
                post_ctx = RefinementContext(
                    mesh=dic_mesh,
                    mask=f_mask,
                    Df=Df,
                    U=U_final,
                    F=F_final,
                    conv_iterations=conv_iter_s4,
                )
                new_mesh, new_U0 = refine_mesh(
                    dic_mesh, refinement_policy.post_solve, post_ctx,
                    U_final, mask=f_mask, img_size=(img_h, img_w),
                )
                if new_mesh.coordinates_fem.shape[0] == dic_mesh.coordinates_fem.shape[0]:
                    logger.info("No elements marked — stopping post-solve refinement")
                    break

                # Update mesh and re-solve
                dic_mesh = new_mesh
                current_U0 = new_U0
                n_nodes = dic_mesh.coordinates_fem.shape[0]

                # Recompute node region map
                node_region_map = precompute_node_regions(
                    dic_mesh.coordinates_fem, f_mask, (img_h, img_w),
                )

                # Re-run Section 4 (local IC-GN)
                (
                    U_subpb1, F_subpb1, local_time, conv_iter_s4,
                    bad_pt_num_s4, mark_hole_strain,
                ) = local_icgn(
                    current_U0, dic_mesh.coordinates_fem, Df,
                    f_img_raw, g_img, para, tol,
                )

                if para.use_global_step:
                    # Invalidate subpb2 cache (mesh changed)
                    subpb2_cache_obj = None
                    subpb2_cache_beta = None

                    # Re-run Section 5-6
                    # (same code as above — extract helper if needed)
                    # ... subpb2_solver, ADMM loop ...

                    # For now: single subpb2 solve + abbreviated ADMM
                    beta_val = _auto_tune_beta(
                        dic_mesh, para, para.mu, U_subpb1, F_subpb1,
                    )
                    grad_dual = np.zeros(4 * n_nodes, dtype=np.float64)
                    disp_dual = np.zeros(2 * n_nodes, dtype=np.float64)

                    U_subpb2 = subpb2_solver(
                        dic_mesh, para.gauss_pt_order, beta_val, para.mu,
                        U_subpb1, F_subpb1, grad_dual, disp_dual,
                        para.alpha, para.winstepsize,
                    )
                    F_subpb2 = global_nodal_strain_fem(dic_mesh, para, U_subpb2)
                    U_subpb2, F_subpb2 = _apply_post_solve_corrections(
                        U_subpb2, F_subpb2, U_subpb1, F_subpb1,
                        dic_mesh, para, node_region_map, mark_hole_strain,
                    )

                    U_final = U_subpb2
                    F_final = F_subpb2
                else:
                    U_final = U_subpb1
                    F_final = F_subpb1

                logger.info(
                    "Post-solve cycle %d: %d nodes",
                    cycle + 1, n_nodes,
                )
```

### Step 3: Ensure local_icgn returns conv_iterations

Check `local_icgn()` return signature — it already returns `conv_iter_s4` (the `ConvItPerEle` array). Verify it contains per-node iteration counts:

```bash
cd staq-dic-python && python -c "
from staq_dic.solver.local_icgn import local_icgn
import inspect
sig = inspect.signature(local_icgn)
print('Returns:', sig.return_annotation)
"
```

### Step 4: Run tests

```bash
cd staq-dic-python && python -m pytest tests/test_integration/test_refinement_pipeline.py -v --timeout=120
```

### Step 5: Run full suite

```bash
cd staq-dic-python && python -m pytest tests/ -v --timeout=120
```

### Step 6: Commit

```bash
git add src/staq_dic/core/pipeline.py tests/test_integration/test_refinement_pipeline.py
git commit -m "feat: integrate post-solve refinement loop into pipeline"
```

---

## Task 9: Per-frame independent mesh (incremental mode support)

**Files:**
- Modify: `src/staq_dic/core/pipeline.py` (frame loop mesh reset logic)
- Test: `tests/test_integration/test_refinement_pipeline.py`

### Step 1: Write failing test

```python
class TestPerFrameRefinement:
    def test_incremental_mode_independent_meshes(self):
        """Each frame in incremental mode should have its own mesh."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        f2 = _make_speckle(h, w, seed=2)
        f3 = _make_speckle(h, w, seed=3)
        mask = np.ones((h, w), dtype=np.float64)
        mask[30:50, 30:50] = 0.0  # small hole
        para = dicpara_default(DICPara(
            winstepsize=16, winsize=32, winsize_min=4,
            reference_mode='incremental',
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        ))
        policy = RefinementPolicy(
            pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
        )
        result = run_aldic(para, [ref, f2, f3], [mask, mask, mask],
                          compute_strain=False, refinement_policy=policy)
        # Both frames should complete
        assert len(result.result_disp) == 2
        assert result.result_disp[0] is not None
        assert result.result_disp[1] is not None
```

### Step 2: Implement per-frame mesh independence

Key change: when `refinement_policy` is set, force `dic_mesh = None` at the start of each frame iteration so that each frame builds and refines its own mesh. In the current pipeline, `dic_mesh` persists across frames — this needs to change only when a refinement policy is active.

In the frame loop (line ~472), after determining ref_idx:

```python
        # Per-frame mesh independence when refinement policy is active
        if refinement_policy is not None and refinement_policy.has_pre_solve:
            dic_mesh = None
            current_U0 = None
```

This forces Section 3 to re-run FFT search and mesh generation for each frame. The pre-solve refinement then produces a frame-specific mesh.

### Step 3: Run tests — all PASS

### Step 4: Commit

```bash
git add src/staq_dic/core/pipeline.py tests/test_integration/test_refinement_pipeline.py
git commit -m "feat: per-frame independent mesh for refinement policy"
```

---

## Implementation Order & Dependencies

```
Task 1: Framework core (Protocol, Context, Policy, refine_mesh)
  ↓
Task 2: Verify generate_mesh helpers are importable
  ↓
Task 3: MaskBoundaryCriterion
  ↓
Task 4: Refactor generate_mesh → delegate to refine_mesh
  ↓
Task 5: Pipeline pre-solve integration
  ↓
Task 6: ManualSelectionCriterion     (independent of 5)
  ↓
Task 7: PosteriorErrorCriterion     (independent of 5-6)
  ↓
Task 8: Pipeline post-solve integration  (depends on 5, 7)
  ↓
Task 9: Per-frame mesh independence    (depends on 5)
```

Tasks 6 and 7 can be done in parallel after Task 5.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| refine_mesh reimplements generate_mesh logic | Task 4 makes generate_mesh delegate to refine_mesh — single source of truth |
| Post-solve re-solve is expensive | max_post_solve_cycles default = 0 (opt-in only) |
| Subpb2 cache invalidation on mesh change | Explicit cache reset in post-solve loop |
| Per-frame mesh breaks cumulative composition | Already handled by scattered_interpolant in tree-based composition |
| PosteriorErrorCriterion metric TBD | Stub with conv_iterations; metric field is extensible |
