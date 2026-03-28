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
