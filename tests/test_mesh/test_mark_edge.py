"""Tests for mark_edge and mark_inside modules."""

import numpy as np
import pytest

from al_dic.mesh.mark_edge import mark_edge
from al_dic.mesh.mark_inside import mark_inside


class TestMarkEdge:
    def _make_simple_mesh(self):
        """Create a 2x2 element mesh on a 3x3 grid (spacing=16)."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
            [0, 32], [16, 32], [32, 32],
        ], dtype=np.float64)
        elems = np.array([
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ], dtype=np.int64)
        return coords, elems

    def test_uniform_mask_no_marks(self):
        """All-ones mask should mark no elements."""
        coords, elems = self._make_simple_mesh()
        mask = np.ones((33, 33), dtype=np.float64)
        marks = mark_edge(coords, elems, mask, min_size=4)
        assert marks.shape == (4,)
        assert not marks.any()

    def test_hole_in_center_marks_surrounding(self):
        """A hole in the center should mark elements that straddle the boundary."""
        coords, elems = self._make_simple_mesh()
        mask = np.ones((33, 33), dtype=np.float64)
        # Create a circular hole at center
        yy, xx = np.mgrid[0:33, 0:33]
        mask[((xx - 16)**2 + (yy - 16)**2) < 8**2] = 0.0

        marks = mark_edge(coords, elems, mask, min_size=4)
        # At least some elements should be marked (they straddle the hole boundary)
        assert marks.any()

    def test_min_size_prevents_marking(self):
        """Elements smaller than min_size should not be marked."""
        coords, elems = self._make_simple_mesh()
        mask = np.ones((33, 33), dtype=np.float64)
        mask[0:10, 0:10] = 0.0  # Hole in corner

        # With min_size larger than element size → no marks
        marks = mark_edge(coords, elems, mask, min_size=100)
        assert not marks.any()

    def test_empty_elements(self):
        """Empty element array should return empty marks."""
        coords = np.array([[0, 0]], dtype=np.float64)
        elems = np.empty((0, 4), dtype=np.int64)
        mask = np.ones((10, 10))
        marks = mark_edge(coords, elems, mask, min_size=1)
        assert len(marks) == 0


class TestMarkInside:
    def test_all_outside(self):
        """All-ones mask: no elements inside."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
        ], dtype=np.float64)
        elems = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64)
        mask = np.ones((33, 33))
        inside, outside = mark_inside(coords, elems, mask)
        assert len(inside) == 0
        assert len(outside) == 2

    def test_element_inside_hole(self):
        """Element fully inside a hole should be classified as inside."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
        ], dtype=np.float64)
        elems = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64)
        mask = np.ones((33, 33))
        # Zero out the region of element 0
        mask[0:17, 0:17] = 0.0

        inside, outside = mark_inside(coords, elems, mask)
        assert 0 in inside

    def test_empty_elements(self):
        coords = np.array([[0, 0]], dtype=np.float64)
        elems = np.empty((0, 4), dtype=np.int64)
        mask = np.ones((10, 10))
        inside, outside = mark_inside(coords, elems, mask)
        assert len(inside) == 0
        assert len(outside) == 0
