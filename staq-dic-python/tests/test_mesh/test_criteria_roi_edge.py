"""Tests for ROIEdgeCriterion with diverse mask topologies.

Mask topologies tested:
  - Simple square ROI (single connected domain, no holes)
  - Square ROI with one central hole
  - Square ROI with multiple holes
  - L-shaped ROI (single connected domain, concave)
  - Annular ROI (single hole, mask has two connected components of mask=0)
  - Multiple disconnected ROI regions (multi-connected domain)
"""
import numpy as np
import pytest

from staq_dic.mesh.criteria.roi_edge import (
    ROIEdgeCriterion,
    _compute_outer_region,
    _find_peripheral_elements,
)
from staq_dic.mesh.refinement import RefinementContext, RefinementCriterion
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICMesh, DICPara


# ---------------------------------------------------------------------------
# Fixtures — diverse mask topologies
# ---------------------------------------------------------------------------

def _make_mesh(h, w, step=16):
    """Build a uniform mesh well inside the image bounds."""
    para = DICPara(winstepsize=step, winsize=step * 2, winsize_min=4)
    # Start grid at step+8 to create asymmetric gap from edges
    x0 = np.arange(step + 8, w - step, step, dtype=np.float64)
    y0 = np.arange(step + 8, h - step, step, dtype=np.float64)
    return mesh_setup(x0, y0, para)


@pytest.fixture
def square_no_hole():
    """256x256, square ROI (margin=20), no holes."""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float64)
    mask[20:236, 20:236] = 1.0
    return _make_mesh(h, w), mask


@pytest.fixture
def square_one_hole():
    """256x256, square ROI (margin=20), one central circular hole."""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float64)
    mask[20:236, 20:236] = 1.0
    yy, xx = np.mgrid[0:h, 0:w]
    mask[(xx - 128) ** 2 + (yy - 128) ** 2 < 30**2] = 0.0
    return _make_mesh(h, w), mask


@pytest.fixture
def square_multi_hole():
    """256x256, square ROI, three holes at different positions."""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float64)
    mask[20:236, 20:236] = 1.0
    yy, xx = np.mgrid[0:h, 0:w]
    # Three small holes
    mask[(xx - 80) ** 2 + (yy - 80) ** 2 < 15**2] = 0.0
    mask[(xx - 180) ** 2 + (yy - 128) ** 2 < 20**2] = 0.0
    mask[(xx - 100) ** 2 + (yy - 190) ** 2 < 12**2] = 0.0
    return _make_mesh(h, w), mask


@pytest.fixture
def l_shaped():
    """256x256, L-shaped ROI (concave, single connected domain, no holes)."""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float64)
    mask[20:236, 20:140] = 1.0   # left vertical bar
    mask[140:236, 20:236] = 1.0  # bottom horizontal bar
    return _make_mesh(h, w), mask


@pytest.fixture
def annular():
    """256x256, annular (ring-shaped) ROI: large circle minus small circle."""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    mask[(xx - 128) ** 2 + (yy - 128) ** 2 < 100**2] = 1.0
    mask[(xx - 128) ** 2 + (yy - 128) ** 2 < 40**2] = 0.0
    return _make_mesh(h, w), mask


@pytest.fixture
def multi_region():
    """256x256, two disconnected rectangular ROI regions."""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float64)
    mask[30:110, 30:110] = 1.0   # top-left block
    mask[140:230, 140:230] = 1.0  # bottom-right block
    return _make_mesh(h, w), mask


# ---------------------------------------------------------------------------
# _compute_outer_region tests
# ---------------------------------------------------------------------------

class TestComputeOuterRegion:
    def test_full_mask_no_outer(self):
        mask = np.ones((64, 64), dtype=np.float64)
        assert not _compute_outer_region(mask).any()

    def test_border_zeros_are_outer(self):
        mask = np.ones((64, 64), dtype=np.float64)
        mask[:5, :] = 0.0
        outer = _compute_outer_region(mask)
        assert outer[:5, :].all()
        assert not outer[5:, :].any()

    def test_internal_hole_not_outer(self):
        mask = np.ones((64, 64), dtype=np.float64)
        mask[25:35, 25:35] = 0.0
        assert not _compute_outer_region(mask).any()

    def test_mixed_outer_and_holes(self):
        mask = np.zeros((64, 64), dtype=np.float64)
        mask[10:54, 10:54] = 1.0
        mask[25:35, 25:35] = 0.0  # internal hole
        outer = _compute_outer_region(mask)
        assert outer[0, 0], "Border region is outer"
        assert not outer[30, 30], "Internal hole is NOT outer"

    def test_annular_mask(self):
        """Annular mask: outer background + inner hole -> only outer is 'outer'."""
        h, w = 128, 128
        mask = np.zeros((h, w), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]
        mask[(xx - 64) ** 2 + (yy - 64) ** 2 < 50**2] = 1.0
        mask[(xx - 64) ** 2 + (yy - 64) ** 2 < 20**2] = 0.0
        outer = _compute_outer_region(mask)
        # Outer background (corners)
        assert outer[0, 0]
        # Inner hole (center)
        assert not outer[64, 64]

    def test_multi_region_single_outer(self):
        """Two disconnected ROI regions: only one outer component."""
        mask = np.zeros((128, 128), dtype=np.float64)
        mask[10:50, 10:50] = 1.0
        mask[70:120, 70:120] = 1.0
        outer = _compute_outer_region(mask)
        # The gap between regions IS connected to the border -> outer
        assert outer[60, 60]
        # Image corner -> outer
        assert outer[0, 0]

    def test_multiple_holes(self):
        """Three internal holes: none are 'outer'."""
        mask = np.ones((128, 128), dtype=np.float64)
        mask[20:30, 20:30] = 0.0
        mask[50:60, 50:60] = 0.0
        mask[80:90, 80:90] = 0.0
        outer = _compute_outer_region(mask)
        assert not outer.any(), "No mask=0 region touches border"


# ---------------------------------------------------------------------------
# _find_peripheral_elements tests
# ---------------------------------------------------------------------------

class TestFindPeripheralElements:
    def test_single_element_all_peripheral(self):
        """A single element is always peripheral."""
        elems = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        assert _find_peripheral_elements(elems)[0]

    def test_2x2_grid_all_peripheral(self):
        """In a 2x2 grid, all 4 elements are peripheral."""
        # 3x3 nodes: (0,1,2), (3,4,5), (6,7,8)
        elems = np.array([
            [0, 1, 4, 3, -1, -1, -1, -1],
            [1, 2, 5, 4, -1, -1, -1, -1],
            [3, 4, 7, 6, -1, -1, -1, -1],
            [4, 5, 8, 7, -1, -1, -1, -1],
        ], dtype=np.int64)
        peripheral = _find_peripheral_elements(elems)
        assert peripheral.all(), "All elements in 2x2 grid are peripheral"

    def test_3x3_grid_has_interior(self):
        """In a 3x3 grid, the center element is NOT peripheral."""
        # 4x4 nodes
        elems_list = []
        for r in range(3):
            for c in range(3):
                n0 = r * 4 + c
                elems_list.append([n0, n0 + 1, n0 + 5, n0 + 4,
                                   -1, -1, -1, -1])
        elems = np.array(elems_list, dtype=np.int64)
        peripheral = _find_peripheral_elements(elems)
        # Center element (index 4) should NOT be peripheral
        assert not peripheral[4], "Center element in 3x3 should be interior"
        # All others should be peripheral
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert peripheral[i], f"Element {i} should be peripheral"

    def test_empty_elements(self):
        elems = np.empty((0, 8), dtype=np.int64)
        assert _find_peripheral_elements(elems).shape == (0,)


# ---------------------------------------------------------------------------
# ROIEdgeCriterion — Protocol and basic
# ---------------------------------------------------------------------------

class TestROIEdgeCriterionBasic:
    def test_implements_protocol(self):
        criterion = ROIEdgeCriterion(half_win=16, min_element_size=4)
        assert isinstance(criterion, RefinementCriterion)

    def test_full_mask_no_marks(self):
        """All-ones mask -> no outer region -> no marks."""
        para = DICPara(winstepsize=16, winsize=32, winsize_min=4)
        x0 = np.arange(16, 64, 16, dtype=np.float64)
        y0 = np.arange(16, 64, 16, dtype=np.float64)
        mesh = mesh_setup(x0, y0, para)
        mask = np.ones((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        marks = ROIEdgeCriterion(half_win=16, min_element_size=4).mark(ctx)
        assert not marks.any()

    def test_min_element_size_prevents_marking(self, square_no_hole):
        mesh, mask = square_no_hole
        ctx = RefinementContext(mesh=mesh, mask=mask)
        marks = ROIEdgeCriterion(half_win=16, min_element_size=999).mark(ctx)
        assert not marks.any()

    def test_raises_without_mask(self, square_no_hole):
        mesh, _ = square_no_hole
        ctx = RefinementContext(mesh=mesh, mask=None)
        with pytest.raises(ValueError, match="mask"):
            ROIEdgeCriterion(half_win=16, min_element_size=4).mark(ctx)

    def test_empty_elements(self):
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
        elems = np.empty((0, 8), dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elems,
                       element_min_size=4)
        mask = np.ones((64, 64), dtype=np.float64)
        ctx = RefinementContext(mesh=mesh, mask=mask)
        marks = ROIEdgeCriterion(half_win=16, min_element_size=4).mark(ctx)
        assert marks.shape == (0,)

    def test_frozen_dataclass(self):
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        with pytest.raises(AttributeError):
            crit.min_element_size = 10  # type: ignore[misc]

    def test_larger_half_win_marks_more(self, square_no_hole):
        mesh, mask = square_no_hole
        ctx = RefinementContext(mesh=mesh, mask=mask)
        small = ROIEdgeCriterion(half_win=4, min_element_size=4).mark(ctx)
        large = ROIEdgeCriterion(half_win=32, min_element_size=4).mark(ctx)
        assert large.sum() >= small.sum()


# ---------------------------------------------------------------------------
# ROIEdgeCriterion — Symmetric marking on all 4 sides
# ---------------------------------------------------------------------------

class TestSymmetricMarking:
    """Verify that all four outer edges get marked, not just left/top."""

    def _check_all_four_sides(self, mesh, mask, criterion):
        """Assert that marked elements exist near all 4 mask edges."""
        ctx = RefinementContext(mesh=mesh, mask=mask)
        marks = criterion.mark(ctx)
        assert marks.any(), "Should mark some elements"

        coords = mesh.coordinates_fem
        elems = mesh.elements_fem
        h, w = mask.shape

        # Find mask boundary (where mask transitions)
        mask_x_min = np.argmax(mask[h // 2, :] > 0.5)
        mask_x_max = w - 1 - np.argmax(mask[h // 2, ::-1] > 0.5)
        mask_y_min = np.argmax(mask[:, w // 2] > 0.5)
        mask_y_max = h - 1 - np.argmax(mask[::-1, w // 2] > 0.5)

        # Classify marked elements by proximity to each edge
        near_left = near_right = near_top = near_bottom = False
        band = 30  # pixels from mask edge to consider "near"

        for i in range(elems.shape[0]):
            if not marks[i]:
                continue
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            if cx - mask_x_min < band:
                near_left = True
            if mask_x_max - cx < band:
                near_right = True
            if cy - mask_y_min < band:
                near_top = True
            if mask_y_max - cy < band:
                near_bottom = True

        return near_left, near_right, near_top, near_bottom

    def test_square_no_hole_all_sides(self, square_no_hole):
        """Simple square ROI: all 4 sides should have marked elements."""
        mesh, mask = square_no_hole
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        left, right, top, bottom = self._check_all_four_sides(mesh, mask, crit)
        assert left, "Left edge should be marked"
        assert right, "Right edge should be marked"
        assert top, "Top edge should be marked"
        assert bottom, "Bottom edge should be marked"

    def test_square_one_hole_all_sides(self, square_one_hole):
        """Square ROI + hole: all 4 outer sides should be marked."""
        mesh, mask = square_one_hole
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        left, right, top, bottom = self._check_all_four_sides(mesh, mask, crit)
        assert left and right and top and bottom

    def test_square_multi_hole_all_sides(self, square_multi_hole):
        """Square ROI + 3 holes: all 4 outer sides should be marked."""
        mesh, mask = square_multi_hole
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        left, right, top, bottom = self._check_all_four_sides(mesh, mask, crit)
        assert left and right and top and bottom


# ---------------------------------------------------------------------------
# ROIEdgeCriterion — Hole exclusion
# ---------------------------------------------------------------------------

class TestHoleExclusion:
    """Elements near holes should NOT be marked by ROIEdgeCriterion."""

    def test_hole_elements_not_marked(self, square_one_hole):
        """Elements near the central hole should not be marked."""
        mesh, mask = square_one_hole
        ctx = RefinementContext(mesh=mesh, mask=mask)
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        marks = crit.mark(ctx)

        coords = mesh.coordinates_fem
        elems = mesh.elements_fem
        h, w = mask.shape

        for i in range(elems.shape[0]):
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            dist_center = np.sqrt((cx - 128) ** 2 + (cy - 128) ** 2)
            dist_border = min(cx - 20, 236 - cx, cy - 20, 236 - cy)
            # Element near hole AND far from border
            if dist_center < 45 and dist_border > 50:
                assert not marks[i], (
                    f"Element {i} near hole (d_hole={dist_center:.0f}, "
                    f"d_border={dist_border:.0f}) should not be marked"
                )

    def test_multi_hole_exclusion(self, square_multi_hole):
        """None of the 3 holes should trigger ROIEdgeCriterion."""
        mesh, mask = square_multi_hole
        ctx = RefinementContext(mesh=mesh, mask=mask)
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        marks = crit.mark(ctx)

        coords = mesh.coordinates_fem
        elems = mesh.elements_fem
        hole_centers = [(80, 80), (180, 128), (100, 190)]

        for i in range(elems.shape[0]):
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            dist_border = min(cx - 20, 236 - cx, cy - 20, 236 - cy)
            for hx, hy in hole_centers:
                dist_hole = np.sqrt((cx - hx) ** 2 + (cy - hy) ** 2)
                if dist_hole < 35 and dist_border > 50:
                    assert not marks[i], (
                        f"Element {i} near hole ({hx},{hy}), "
                        f"d_hole={dist_hole:.0f}, d_border={dist_border:.0f}"
                    )

    def test_annular_inner_ring_not_marked(self, annular):
        """Annular ROI: inner ring boundary = hole, should NOT be marked."""
        mesh, mask = annular
        ctx = RefinementContext(mesh=mesh, mask=mask)
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        marks = crit.mark(ctx)

        coords = mesh.coordinates_fem
        elems = mesh.elements_fem

        for i in range(elems.shape[0]):
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            dist_center = np.sqrt((cx - 128) ** 2 + (cy - 128) ** 2)
            # Near inner ring (hole radius=40) and far from outer ring (radius=100)
            if dist_center < 55 and dist_center > 30:
                assert not marks[i], (
                    f"Element {i} near annular inner ring "
                    f"(d_center={dist_center:.0f}) should not be marked"
                )


# ---------------------------------------------------------------------------
# ROIEdgeCriterion — Special topologies
# ---------------------------------------------------------------------------

class TestSpecialTopologies:
    def test_l_shaped_marks_outer_only(self, l_shaped):
        """L-shaped ROI: marks outer edges, not the concave inner corner."""
        mesh, mask = l_shaped
        ctx = RefinementContext(mesh=mesh, mask=mask)
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        marks = crit.mark(ctx)
        assert marks.any(), "Should mark elements near L-shaped outer edges"
        # Interior of the L should not be marked
        coords = mesh.coordinates_fem
        elems = mesh.elements_fem
        for i in range(elems.shape[0]):
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            # Deep interior of L: bottom-left region
            if 50 < cx < 100 and 160 < cy < 210:
                assert not marks[i], (
                    f"Element {i} in L interior ({cx:.0f},{cy:.0f}) "
                    f"should not be marked"
                )

    def test_annular_outer_ring_marked(self, annular):
        """Annular ROI: outer ring elements SHOULD be marked."""
        mesh, mask = annular
        ctx = RefinementContext(mesh=mesh, mask=mask)
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        marks = crit.mark(ctx)

        # Should have marks near the outer circle (radius ~100)
        coords = mesh.coordinates_fem
        elems = mesh.elements_fem
        near_outer = False
        for i in range(elems.shape[0]):
            if not marks[i]:
                continue
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            dist = np.sqrt((cx - 128) ** 2 + (cy - 128) ** 2)
            if dist > 70:
                near_outer = True
                break
        assert near_outer, "Should mark elements near annular outer ring"

    def test_multi_region_both_regions_marked(self, multi_region):
        """Two disconnected ROIs: both should have outer-edge marking."""
        mesh, mask = multi_region
        ctx = RefinementContext(mesh=mesh, mask=mask)
        crit = ROIEdgeCriterion(half_win=16, min_element_size=4)
        marks = crit.mark(ctx)

        coords = mesh.coordinates_fem
        elems = mesh.elements_fem
        in_region_1 = False
        in_region_2 = False

        for i in range(elems.shape[0]):
            if not marks[i]:
                continue
            corners = elems[i, :4]
            if np.any(corners < 0):
                continue
            cx = coords[corners, 0].mean()
            cy = coords[corners, 1].mean()
            if cx < 120 and cy < 120:
                in_region_1 = True
            if cx > 130 and cy > 130:
                in_region_2 = True

        assert in_region_1, "Region 1 (top-left) should have outer-edge marks"
        assert in_region_2, "Region 2 (bottom-right) should have outer-edge marks"
