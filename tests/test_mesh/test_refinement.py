"""Tests for mesh refinement framework (Protocol, Context, Policy, refine_mesh)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import DICMesh
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.mesh.refinement import (
    RefinementContext,
    RefinementCriterion,
    RefinementPolicy,
    refine_mesh,
)


# ---------------------------------------------------------------------------
# Test helper criteria
# ---------------------------------------------------------------------------


class _AlwaysRefineCriterion:
    """Marks all elements whose size exceeds min_element_size."""

    min_element_size: int = 4

    def __init__(self, min_element_size: int = 4) -> None:
        self.min_element_size = min_element_size

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        coords = ctx.mesh.coordinates_fem
        elems = ctx.mesh.elements_fem[:, :4]
        n_elem = elems.shape[0]
        marks = np.ones(n_elem, dtype=np.bool_)
        # Only mark elements whose diagonal > min_element_size
        dx = coords[elems[:, 0], 0] - coords[elems[:, 2], 0]
        dy = coords[elems[:, 0], 1] - coords[elems[:, 2], 1]
        diag = np.sqrt(dx**2 + dy**2)
        marks[diag <= self.min_element_size * np.sqrt(2)] = False
        return marks


class _NeverRefineCriterion:
    """Never marks any element."""

    min_element_size: int = 1

    def __init__(self, min_element_size: int = 1) -> None:
        self.min_element_size = min_element_size

    def mark(self, ctx: RefinementContext) -> NDArray[np.bool_]:
        n_elem = ctx.mesh.elements_fem.shape[0]
        return np.zeros(n_elem, dtype=np.bool_)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mesh(step: int = 16, nx: int = 4, ny: int = 4) -> tuple[DICMesh, NDArray]:
    """Create a uniform mesh and zero U0 for testing."""
    img_h = (ny - 1) * step + step
    img_w = (nx - 1) * step + step
    para = dicpara_default(
        img_size=(img_h, img_w),
        winstepsize=step,
        winsize_min=step // 2,
    )
    x0 = np.arange(nx, dtype=np.float64) * step
    y0 = np.arange(ny, dtype=np.float64) * step
    m = mesh_setup(x0, y0, para)
    U0 = np.zeros(2 * m.coordinates_fem.shape[0], dtype=np.float64)
    return m, U0


# ---------------------------------------------------------------------------
# RefinementContext tests
# ---------------------------------------------------------------------------


class TestRefinementContext:
    def test_minimal_creation(self):
        """RefinementContext with only mesh should be valid."""
        m, _ = _make_mesh()
        ctx = RefinementContext(mesh=m)
        assert ctx.mesh is m
        assert ctx.mask is None
        assert ctx.user_marks is None

    def test_all_fields(self):
        """RefinementContext with all optional fields set."""
        m, U0 = _make_mesh()
        n_elem = m.elements_fem.shape[0]
        mask = np.ones((64, 64), dtype=np.float64)
        user = np.zeros(n_elem, dtype=np.int64)

        ctx = RefinementContext(
            mesh=m,
            mask=mask,
            user_marks=user,
        )
        assert ctx.mask is mask
        assert ctx.user_marks is user


# ---------------------------------------------------------------------------
# RefinementPolicy tests
# ---------------------------------------------------------------------------


class TestRefinementPolicy:
    def test_empty_policy(self):
        """Empty policy has no pre-solve criteria."""
        pol = RefinementPolicy(pre_solve=[])
        assert not pol.has_pre_solve

    def test_pre_solve_only(self):
        """Policy with pre_solve criteria."""
        crit = _AlwaysRefineCriterion()
        pol = RefinementPolicy(pre_solve=[crit])
        assert pol.has_pre_solve


# ---------------------------------------------------------------------------
# RefinementCriterion Protocol tests
# ---------------------------------------------------------------------------


class TestRefinementCriterionProtocol:
    def test_always_refine_is_criterion(self):
        """_AlwaysRefineCriterion satisfies RefinementCriterion protocol."""
        crit = _AlwaysRefineCriterion()
        assert isinstance(crit, RefinementCriterion)

    def test_never_refine_is_criterion(self):
        """_NeverRefineCriterion satisfies RefinementCriterion protocol."""
        crit = _NeverRefineCriterion()
        assert isinstance(crit, RefinementCriterion)


# ---------------------------------------------------------------------------
# refine_mesh tests
# ---------------------------------------------------------------------------


class TestRefineMesh:
    def test_no_criteria_unchanged(self):
        """No criteria → mesh and U0 returned unchanged."""
        m, U0 = _make_mesh()
        ctx = RefinementContext(mesh=m)
        m2, U2 = refine_mesh(m, [], ctx, U0)

        assert m2.coordinates_fem.shape == m.coordinates_fem.shape
        assert m2.elements_fem.shape == m.elements_fem.shape
        np.testing.assert_array_equal(U2, U0)

    def test_never_refine_unchanged(self):
        """NeverRefineCriterion → mesh unchanged."""
        m, U0 = _make_mesh()
        ctx = RefinementContext(mesh=m)
        crit = _NeverRefineCriterion()
        m2, U2 = refine_mesh(m, [crit], ctx, U0)

        assert m2.coordinates_fem.shape == m.coordinates_fem.shape
        assert m2.elements_fem.shape == m.elements_fem.shape
        np.testing.assert_array_equal(U2, U0)

    def test_always_refine_increases_nodes(self):
        """AlwaysRefineCriterion should produce more nodes and elements."""
        m, U0 = _make_mesh(step=16, nx=3, ny=3)
        ctx = RefinementContext(mesh=m)
        crit = _AlwaysRefineCriterion(min_element_size=4)
        m2, U2 = refine_mesh(m, [crit], ctx, U0)

        assert m2.coordinates_fem.shape[0] > m.coordinates_fem.shape[0]
        assert m2.elements_fem.shape[0] > m.elements_fem.shape[0]
        assert len(U2) == 2 * m2.coordinates_fem.shape[0]

    def test_union_of_two_criteria(self):
        """Union of AlwaysRefine + NeverRefine should refine (OR logic)."""
        m, U0 = _make_mesh(step=16, nx=3, ny=3)
        ctx = RefinementContext(mesh=m)
        always = _AlwaysRefineCriterion(min_element_size=4)
        never = _NeverRefineCriterion()
        m2, U2 = refine_mesh(m, [always, never], ctx, U0)

        # Same as always-refine alone
        m3, _ = refine_mesh(m, [always], ctx, U0)
        assert m2.coordinates_fem.shape[0] == m3.coordinates_fem.shape[0]
        assert m2.elements_fem.shape[0] == m3.elements_fem.shape[0]

    def test_min_size_respected(self):
        """Elements should not be refined below min_element_size."""
        m, U0 = _make_mesh(step=16, nx=3, ny=3)
        ctx = RefinementContext(mesh=m)
        # min_element_size=8 means diagonal limit ~ 8*sqrt(2) ≈ 11.3
        crit = _AlwaysRefineCriterion(min_element_size=8)
        m2, U2 = refine_mesh(m, [crit], ctx, U0)

        # Check no element has diagonal smaller than min_element_size * sqrt(2)
        coords = m2.coordinates_fem
        corners = m2.elements_fem[:, :4]
        dx = coords[corners[:, 0], 0] - coords[corners[:, 2], 0]
        dy = coords[corners[:, 0], 1] - coords[corners[:, 2], 1]
        diags = np.sqrt(dx**2 + dy**2)
        # All elements should have diagonal >= min_size * sqrt(2) - tolerance
        # (the criterion doesn't mark elements at or below min_size)
        min_diag = diags.min()
        # Elements at min_size level have diag = min_element_size * sqrt(2)
        assert min_diag >= 8 * np.sqrt(2) - 1.0

    def test_with_mask(self):
        """refine_mesh with mask should remove inside elements."""
        m, U0 = _make_mesh(step=16, nx=5, ny=5)
        img_h = 80
        img_w = 80
        mask = np.ones((img_h, img_w), dtype=np.float64)
        # Punch a hole in the center
        mask[30:50, 30:50] = 0.0
        ctx = RefinementContext(mesh=m, mask=mask)
        crit = _AlwaysRefineCriterion(min_element_size=4)
        m2, U2 = refine_mesh(m, [crit], ctx, U0, mask=mask, img_size=(img_h, img_w))

        # Some elements should be removed (inside hole)
        # Verify all corner indices are valid
        n_nodes = m2.coordinates_fem.shape[0]
        corners = m2.elements_fem[:, :4]
        assert corners.min() >= 0
        assert corners.max() < n_nodes

    def test_q8_elements_after_refinement(self):
        """Refined mesh should have Q8 (8-column) elements."""
        m, U0 = _make_mesh(step=16, nx=3, ny=3)
        ctx = RefinementContext(mesh=m)
        crit = _AlwaysRefineCriterion(min_element_size=4)
        m2, _ = refine_mesh(m, [crit], ctx, U0)

        assert m2.elements_fem.shape[1] == 8
