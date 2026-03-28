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
        """Solid mask (all ones) -> no elements marked."""
        mesh, para = mesh_64x64
        mask = np.ones((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_empty_mask_no_marks(self, mesh_64x64):
        """All-zero mask -> no elements marked (all uniformly zero)."""
        mesh, para = mesh_64x64
        mask = np.zeros((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_center_hole_marks_boundary_elements(self, mesh_64x64):
        """Circular hole in center -> marks elements straddling boundary."""
        mesh, para = mesh_64x64
        mask = np.ones((64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        mask[(xx - 32) ** 2 + (yy - 32) ** 2 < 15**2] = 0.0
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

    def test_empty_elements(self):
        """Empty element array -> empty boolean array."""
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
        elems = np.empty((0, 8), dtype=np.int64)
        mesh = DICMesh(
            coordinates_fem=coords,
            elements_fem=elems,
            element_min_size=4,
        )
        mask = np.ones((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        criterion = MaskBoundaryCriterion(min_element_size=4)
        marks = criterion.mark(ctx)
        assert marks.shape == (0,)
        assert marks.dtype == np.bool_

    def test_default_min_element_size(self):
        """Default min_element_size is 8."""
        criterion = MaskBoundaryCriterion()
        assert criterion.min_element_size == 8

    def test_frozen_dataclass(self):
        """MaskBoundaryCriterion should be immutable (frozen)."""
        criterion = MaskBoundaryCriterion(min_element_size=4)
        with pytest.raises(AttributeError):
            criterion.min_element_size = 10  # type: ignore[misc]
