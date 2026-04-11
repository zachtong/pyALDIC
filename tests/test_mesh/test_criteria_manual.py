"""Tests for ManualSelectionCriterion."""
import numpy as np
import pytest

from al_dic.mesh.criteria.manual_selection import ManualSelectionCriterion
from al_dic.mesh.refinement import (
    RefinementContext,
    RefinementCriterion,
    refine_mesh,
)
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.core.data_structures import DICPara


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

    def test_user_marks_not_reapplied_after_refinement(self, mesh_4x4):
        """user_marks should only fire once, not recursively over-refine.

        Before fix: user_marks=[0] would refine element 0 in round 1,
        then re-mark whatever element sits at index 0 in round 2, etc.,
        creating asymmetric over-refinement until min_element_size stops it.

        After fix: user_marks is cleared after the first round, so only
        one level of subdivision occurs for the selected element.
        """
        n_elem_before = mesh_4x4.elements_fem.shape[0]
        U0 = np.zeros(2 * mesh_4x4.coordinates_fem.shape[0], dtype=np.float64)

        criterion = ManualSelectionCriterion(
            element_indices=np.array([0], dtype=np.int64),
            min_element_size=1,
        )
        ctx = RefinementContext(mesh=mesh_4x4)
        refined, _ = refine_mesh(mesh_4x4, [criterion], ctx, U0)

        # One element subdivided into 4: net gain of 3 elements
        n_elem_after = refined.elements_fem.shape[0]
        assert n_elem_after == n_elem_before + 3, (
            f"Expected exactly 1 subdivision (+3 elements), "
            f"got {n_elem_after - n_elem_before:+d}. "
            f"user_marks may be leaking into subsequent iterations."
        )
