# Brush Region Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `BrushRegionCriterion` that refines elements overlapping with a user-painted binary mask, plus a `build_refinement_policy()` factory function that builds a `RefinementPolicy` from two simple user inputs: `refine_boundary: bool` and `refinement_mask: NDArray | None`.

**Architecture:** New criterion follows the existing `RefinementCriterion` Protocol pattern (frozen dataclass with `min_element_size` attribute and `mark(ctx)` method). Factory function lives in `refinement.py` alongside `RefinementPolicy`. No changes to pipeline, `refine_mesh()`, or existing criteria.

**Tech Stack:** Python, NumPy, pytest, matplotlib (for PDF report)

---

### Task 1: BrushRegionCriterion — Test

**Files:**
- Create: `tests/test_mesh/test_criteria_brush.py`

**Step 1: Write tests**

```python
"""Tests for BrushRegionCriterion."""
import numpy as np
import pytest

from staq_dic.mesh.criteria.brush_region import BrushRegionCriterion
from staq_dic.mesh.refinement import RefinementContext, RefinementCriterion
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICMesh, DICPara


@pytest.fixture
def mesh_64x64():
    """4x4 uniform mesh on 64x64 image, step=16."""
    para = DICPara(winstepsize=16, winsize=32, winsize_min=4)
    x0 = np.arange(16, 64, 16, dtype=np.float64)
    y0 = np.arange(16, 64, 16, dtype=np.float64)
    return mesh_setup(x0, y0, para)


class TestBrushRegionCriterion:
    def test_implements_protocol(self):
        """BrushRegionCriterion satisfies RefinementCriterion protocol."""
        mask = np.zeros((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=mask, min_element_size=4)
        assert isinstance(criterion, RefinementCriterion)

    def test_full_overlap_marks_all(self, mesh_64x64):
        """All-ones refinement mask -> all elements marked."""
        rmask = np.ones((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        ctx = RefinementContext(mesh=mesh_64x64)
        marks = criterion.mark(ctx)
        assert marks.all(), "Every element overlaps with all-ones mask"

    def test_no_overlap_marks_none(self, mesh_64x64):
        """All-zeros refinement mask -> no elements marked."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        ctx = RefinementContext(mesh=mesh_64x64)
        marks = criterion.mark(ctx)
        assert not marks.any(), "No element overlaps with all-zeros mask"

    def test_partial_overlap(self, mesh_64x64):
        """Paint a small region -> only overlapping elements marked."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        # Paint a 10x10 region in the top-left corner
        rmask[10:20, 10:20] = 1.0
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        ctx = RefinementContext(mesh=mesh_64x64)
        marks = criterion.mark(ctx)
        # At least one element should be marked (overlaps 10:20 region)
        assert marks.any(), "Should mark elements overlapping painted region"
        # Not all elements should be marked (painted region is small)
        assert not marks.all(), "Should not mark all elements"

    def test_min_element_size_prevents_marking(self, mesh_64x64):
        """Large min_element_size prevents any marking."""
        rmask = np.ones((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=999)
        ctx = RefinementContext(mesh=mesh_64x64)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_empty_elements(self):
        """Empty element array -> empty boolean array."""
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
        elems = np.empty((0, 8), dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elems, element_min_size=4)
        rmask = np.ones((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        ctx = RefinementContext(mesh=mesh)
        marks = criterion.mark(ctx)
        assert marks.shape == (0,)
        assert marks.dtype == np.bool_

    def test_frozen_dataclass(self):
        """Should be immutable."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        with pytest.raises(AttributeError):
            criterion.min_element_size = 10  # type: ignore[misc]

    def test_default_min_element_size(self):
        """Default min_element_size is 4."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        criterion = BrushRegionCriterion(refinement_mask=rmask)
        assert criterion.min_element_size == 4

    def test_brush_stroke_line(self, mesh_64x64):
        """A thin brush stroke (horizontal line) marks elements it crosses."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        # Paint a horizontal line at y=32 (middle), 3 pixels wide
        rmask[31:34, 10:55] = 1.0
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        ctx = RefinementContext(mesh=mesh_64x64)
        marks = criterion.mark(ctx)
        # Should mark middle-row elements but not top/bottom rows
        assert marks.any()
        assert not marks.all()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mesh/test_criteria_brush.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'staq_dic.mesh.criteria.brush_region'`

**Step 3: Commit failing test**

```bash
git add tests/test_mesh/test_criteria_brush.py
git commit -m "test: add failing tests for BrushRegionCriterion"
```

---

### Task 2: BrushRegionCriterion — Implementation

**Files:**
- Create: `src/staq_dic/mesh/criteria/brush_region.py`
- Modify: `src/staq_dic/mesh/criteria/__init__.py`

**Step 1: Implement BrushRegionCriterion**

```python
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
```

**Step 2: Update `criteria/__init__.py`**

Add `BrushRegionCriterion` to the exports:

```python
"""Built-in mesh refinement criteria."""
from .mask_boundary import MaskBoundaryCriterion
from .manual_selection import ManualSelectionCriterion
from .brush_region import BrushRegionCriterion

__all__ = [
    "MaskBoundaryCriterion",
    "ManualSelectionCriterion",
    "BrushRegionCriterion",
]
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_mesh/test_criteria_brush.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/staq_dic/mesh/criteria/brush_region.py src/staq_dic/mesh/criteria/__init__.py
git commit -m "feat: BrushRegionCriterion — refine elements overlapping user-painted regions"
```

---

### Task 3: build_refinement_policy — Test

**Files:**
- Create: `tests/test_mesh/test_build_policy.py`

**Step 1: Write tests**

```python
"""Tests for build_refinement_policy factory function."""
import numpy as np
import pytest

from staq_dic.mesh.refinement import RefinementPolicy, build_refinement_policy
from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from staq_dic.mesh.criteria.brush_region import BrushRegionCriterion


class TestBuildRefinementPolicy:
    def test_no_options_returns_none(self):
        """No refinement requested -> returns None."""
        policy = build_refinement_policy()
        assert policy is None

    def test_boundary_only(self):
        """refine_boundary=True -> policy with MaskBoundaryCriterion."""
        policy = build_refinement_policy(refine_boundary=True)
        assert policy is not None
        assert len(policy.pre_solve) == 1
        assert isinstance(policy.pre_solve[0], MaskBoundaryCriterion)

    def test_brush_only(self):
        """refinement_mask provided -> policy with BrushRegionCriterion."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(refinement_mask=rmask)
        assert policy is not None
        assert len(policy.pre_solve) == 1
        assert isinstance(policy.pre_solve[0], BrushRegionCriterion)

    def test_both_combined(self):
        """Both options -> policy with two criteria."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(
            refine_boundary=True, refinement_mask=rmask,
        )
        assert policy is not None
        assert len(policy.pre_solve) == 2
        types = {type(c) for c in policy.pre_solve}
        assert MaskBoundaryCriterion in types
        assert BrushRegionCriterion in types

    def test_min_element_size_propagated(self):
        """min_element_size should be set on all criteria."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(
            refine_boundary=True, refinement_mask=rmask,
            min_element_size=6,
        )
        assert policy is not None
        for crit in policy.pre_solve:
            assert crit.min_element_size == 6

    def test_empty_refinement_mask_returns_none(self):
        """All-zero refinement mask is still a valid mask (returns policy)."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        policy = build_refinement_policy(refinement_mask=rmask)
        # Even all-zero mask is a valid user intent — the criterion
        # will simply mark no elements at runtime.
        assert policy is not None

    def test_returns_refinement_policy_type(self):
        """Return type is RefinementPolicy."""
        policy = build_refinement_policy(refine_boundary=True)
        assert isinstance(policy, RefinementPolicy)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mesh/test_build_policy.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_refinement_policy'`

**Step 3: Commit**

```bash
git add tests/test_mesh/test_build_policy.py
git commit -m "test: add failing tests for build_refinement_policy"
```

---

### Task 4: build_refinement_policy — Implementation

**Files:**
- Modify: `src/staq_dic/mesh/refinement.py` (add function at bottom, add imports)
- Modify: `src/staq_dic/mesh/__init__.py` (add exports)

**Step 1: Add build_refinement_policy to refinement.py**

Add at the bottom of `refinement.py`, after `_to_q8_placeholder`:

```python
def build_refinement_policy(
    *,
    refine_boundary: bool = False,
    refinement_mask: NDArray[np.float64] | None = None,
    min_element_size: int = 8,
) -> RefinementPolicy | None:
    """Build a RefinementPolicy from user configuration.

    Convenience factory that translates simple user inputs into the
    appropriate combination of refinement criteria.

    Args:
        refine_boundary: If True, adds a ``MaskBoundaryCriterion`` that
            refines elements straddling the mask boundary.
        refinement_mask: Optional (H, W) binary image where 1.0 marks
            regions the user wants refined (e.g. from a brush tool).
            Adds a ``BrushRegionCriterion`` when provided.
        min_element_size: Minimum element size passed to all criteria.

    Returns:
        A ``RefinementPolicy``, or ``None`` if no refinement is requested.
    """
    from .criteria.mask_boundary import MaskBoundaryCriterion
    from .criteria.brush_region import BrushRegionCriterion

    criteria: list[RefinementCriterion] = []

    if refine_boundary:
        criteria.append(MaskBoundaryCriterion(min_element_size=min_element_size))

    if refinement_mask is not None:
        criteria.append(
            BrushRegionCriterion(
                refinement_mask=refinement_mask,
                min_element_size=min_element_size,
            )
        )

    if not criteria:
        return None

    return RefinementPolicy(pre_solve=criteria)
```

**Step 2: Update `mesh/__init__.py` exports**

```python
"""Quadtree mesh generation and refinement."""

from .generate_mesh import generate_mesh
from .mesh_setup import mesh_setup
from .refinement import (
    RefinementContext,
    RefinementCriterion,
    RefinementPolicy,
    build_refinement_policy,
    refine_mesh,
)
from .criteria import BrushRegionCriterion

__all__ = [
    "mesh_setup",
    "generate_mesh",
    "refine_mesh",
    "RefinementPolicy",
    "RefinementContext",
    "RefinementCriterion",
    "build_refinement_policy",
    "BrushRegionCriterion",
]
```

**Step 3: Run all tests**

Run: `python -m pytest tests/test_mesh/test_criteria_brush.py tests/test_mesh/test_build_policy.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/staq_dic/mesh/refinement.py src/staq_dic/mesh/__init__.py
git commit -m "feat: build_refinement_policy() factory for simple user configuration"
```

---

### Task 5: Run full test suite to verify no regressions

**Step 1: Run all mesh tests**

Run: `python -m pytest tests/test_mesh/ -v`
Expected: ALL PASS

**Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration/test_refinement_pipeline.py -v`
Expected: ALL PASS

**Step 3: Run full suite**

Run: `python -m pytest tests/ -x --timeout=300`
Expected: ALL PASS

---

### Task 6: Visual PDF Report

**Files:**
- Create: `scripts/report_brush_refinement.py`

**Step 1: Write report script**

Creates a 4-page PDF in `reports/`:
1. **Page 1**: Reference image + mask + brush strokes overlay
2. **Page 2**: 4 mesh comparisons (uniform / boundary-only / brush-only / boundary+brush)
3. **Page 3**: Element count and size distribution bar charts
4. **Page 4**: Summary table

Uses a 256x256 square mask with a central circular hole. Brush region is a diagonal band across the image (simulating user painting where they expect strain concentration).

**Step 2: Run report**

Run: `python scripts/report_brush_refinement.py`
Expected: PDF saved to `reports/brush_refinement_report.pdf`

---

## Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `src/staq_dic/mesh/criteria/brush_region.py` | **Create** | `BrushRegionCriterion` — marks elements overlapping user-painted regions |
| `src/staq_dic/mesh/criteria/__init__.py` | **Modify** | Export `BrushRegionCriterion` |
| `src/staq_dic/mesh/refinement.py` | **Modify** | Add `build_refinement_policy()` factory |
| `src/staq_dic/mesh/__init__.py` | **Modify** | Export `BrushRegionCriterion` + `build_refinement_policy` |
| `tests/test_mesh/test_criteria_brush.py` | **Create** | 9 unit tests for BrushRegionCriterion |
| `tests/test_mesh/test_build_policy.py` | **Create** | 7 unit tests for build_refinement_policy |
| `scripts/report_brush_refinement.py` | **Create** | Visual PDF comparing refinement configurations |

**Not modified**: `pipeline.py`, `refine_mesh()`, `MaskBoundaryCriterion`, `ManualSelectionCriterion`, any existing tests.
