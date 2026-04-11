"""Tests for BrushRegionCriterion."""
import numpy as np
import pytest

from al_dic.mesh.criteria.brush_region import BrushRegionCriterion
from al_dic.mesh.refinement import RefinementContext, RefinementCriterion
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.core.data_structures import DICMesh, DICPara


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
        assert marks.any(), "Should mark elements overlapping painted region"
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
        mesh = DICMesh(
            coordinates_fem=coords, elements_fem=elems, element_min_size=4
        )
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

    def test_brush_stroke_corner(self, mesh_64x64):
        """A small brush stroke in one corner marks only nearby elements."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        # Paint a small patch in the bottom-right corner (y=42:48, x=42:48)
        # Only the bottom-right element (nodes near x=48,y=48) should overlap
        rmask[42:48, 42:48] = 1.0
        criterion = BrushRegionCriterion(refinement_mask=rmask, min_element_size=4)
        ctx = RefinementContext(mesh=mesh_64x64)
        marks = criterion.mark(ctx)
        assert marks.any(), "Should mark element overlapping bottom-right patch"
        assert marks.sum() < marks.size, "Should not mark all elements"
