"""Tests for qrefine_r (quadtree red refinement)."""

import numpy as np
import pytest

from al_dic.mesh.qrefine_r import qrefine_r
from al_dic.mesh.geometric_data import provide_geometric_data


class TestProvideGeometricData:
    def test_single_element(self):
        """Single Q4 element should produce 4 edges."""
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        irregular = np.empty((0, 3), dtype=np.int64)

        edge2nodes, irr2edges, elem2edges = provide_geometric_data(irregular, elements)

        assert edge2nodes.shape[0] == 4  # 4 unique edges
        assert elem2edges.shape == (1, 4)
        assert irr2edges.shape == (0, 3)

    def test_two_elements_shared_edge(self):
        """Two adjacent elements should share one edge."""
        #  3 --- 2 --- 5
        #  |  0  |  1  |
        #  0 --- 1 --- 4
        elements = np.array([
            [0, 1, 2, 3],
            [1, 4, 5, 2],
        ], dtype=np.int64)
        irregular = np.empty((0, 3), dtype=np.int64)

        edge2nodes, _, elem2edges = provide_geometric_data(irregular, elements)

        # 7 unique edges (4 + 4 - 1 shared)
        assert edge2nodes.shape[0] == 7
        assert elem2edges.shape == (2, 4)

        # Shared edge between elements: edge (1, 2)
        # Both elements should reference the same edge index for this edge
        # Element 0 edge (1->2) and Element 1 edge (2->1) should be same
        edge01 = elem2edges[0]  # edges of element 0
        edge11 = elem2edges[1]  # edges of element 1
        # The shared edge index should appear in both
        shared = np.intersect1d(edge01, edge11)
        assert len(shared) == 1


class TestQrefineR:
    def _make_single_element(self):
        """Single element: nodes at corners of a 16x16 square."""
        coords = np.array([
            [0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0],
        ])
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        irregular = np.empty((0, 3), dtype=np.int64)
        return coords, elements, irregular

    def test_no_marking(self):
        """Empty marked_elements should return unchanged mesh."""
        coords, elements, irregular = self._make_single_element()
        marked = np.array([], dtype=np.int64)

        new_coords, new_elems, new_irr = qrefine_r(coords, elements, irregular, marked)

        np.testing.assert_array_equal(new_coords, coords)
        np.testing.assert_array_equal(new_elems, elements)

    def test_single_element_red_refinement(self):
        """Marking the only element should produce 4 children + 5 new nodes."""
        coords, elements, irregular = self._make_single_element()
        marked = np.array([0], dtype=np.int64)

        new_coords, new_elems, new_irr = qrefine_r(coords, elements, irregular, marked)

        # 4 original + 4 edge midpoints + 1 center = 9 nodes
        assert new_coords.shape[0] == 9
        # 4 child elements
        assert new_elems.shape[0] == 4
        # No irregular nodes (fully refined)
        assert new_irr.shape[0] == 0

    def test_red_refinement_midpoints(self):
        """Midpoints should be at the correct positions."""
        coords, elements, irregular = self._make_single_element()
        marked = np.array([0], dtype=np.int64)

        new_coords, new_elems, new_irr = qrefine_r(coords, elements, irregular, marked)

        # Check that midpoints exist
        expected_midpoints = [
            [8.0, 0.0],   # mid(0,1)
            [16.0, 8.0],  # mid(1,2)
            [8.0, 16.0],  # mid(2,3)
            [0.0, 8.0],   # mid(3,0)
            [8.0, 8.0],   # center
        ]
        for mp in expected_midpoints:
            dists = np.linalg.norm(new_coords - mp, axis=1)
            assert dists.min() < 1e-10, f"Midpoint {mp} not found in coordinates"

    def test_two_elements_partial_refinement(self):
        """Marking one of two adjacent elements should create irregular nodes."""
        #  3 --- 2 --- 5
        #  |  0  |  1  |
        #  0 --- 1 --- 4
        coords = np.array([
            [0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0],
            [32.0, 0.0], [32.0, 16.0],
        ])
        elements = np.array([
            [0, 1, 2, 3],
            [1, 4, 5, 2],
        ], dtype=np.int64)
        irregular = np.empty((0, 3), dtype=np.int64)

        # Mark only element 0
        marked = np.array([0], dtype=np.int64)
        new_coords, new_elems, new_irr = qrefine_r(coords, elements, irregular, marked)

        # Element 0 is refined into 4 children, element 1 stays
        # Total: 4 + 1 = 5 elements
        assert new_elems.shape[0] == 5
        # There should be irregular nodes on the shared edge
        assert new_irr.shape[0] > 0

    def test_all_indices_valid(self):
        """All element node indices should be within coordinate bounds."""
        coords, elements, irregular = self._make_single_element()
        marked = np.array([0], dtype=np.int64)

        new_coords, new_elems, new_irr = qrefine_r(coords, elements, irregular, marked)

        assert new_elems.min() >= 0
        assert new_elems.max() < len(new_coords)
