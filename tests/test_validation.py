"""Tests for al_dic.utils.validation — data integrity assertions."""

import numpy as np
import pytest

from al_dic.utils.validation import (
    assert_displacement_vector,
    assert_def_grad_vector,
    assert_mesh_consistent,
)


# ---------------------------------------------------------------------------
# assert_displacement_vector
# ---------------------------------------------------------------------------

class TestAssertDisplacementVector:
    def test_valid_shape(self):
        U = np.zeros(20, dtype=np.float64)
        assert_displacement_vector(U, n_nodes=10)  # should not raise

    def test_wrong_shape(self):
        U = np.zeros(21, dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            assert_displacement_vector(U, n_nodes=10)

    def test_all_nan_raises(self):
        U = np.full(20, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="NaN"):
            assert_displacement_vector(U, n_nodes=10)

    def test_partial_nan_ok(self):
        U = np.zeros(20, dtype=np.float64)
        U[0] = np.nan
        assert_displacement_vector(U, n_nodes=10)  # should not raise


# ---------------------------------------------------------------------------
# assert_def_grad_vector
# ---------------------------------------------------------------------------

class TestAssertDefGradVector:
    def test_valid_shape(self):
        F = np.zeros(40, dtype=np.float64)
        assert_def_grad_vector(F, n_nodes=10)

    def test_wrong_shape(self):
        F = np.zeros(41, dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            assert_def_grad_vector(F, n_nodes=10)


# ---------------------------------------------------------------------------
# assert_mesh_consistent
# ---------------------------------------------------------------------------

class TestAssertMeshConsistent:
    def test_valid_mesh(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        assert_mesh_consistent(coords, elems)  # should not raise

    def test_wrong_coords_shape(self):
        coords = np.zeros((4, 3), dtype=np.float64)  # must be (N, 2)
        elems = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        with pytest.raises(ValueError, match="coordinates_fem"):
            assert_mesh_consistent(coords, elems)

    def test_wrong_elements_shape(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)  # must be (M, 8)
        with pytest.raises(ValueError, match="elements_fem"):
            assert_mesh_consistent(coords, elems)

    def test_index_out_of_bounds(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 2, 99, -1, -1, -1, -1]], dtype=np.int64)
        with pytest.raises(ValueError, match="Element index"):
            assert_mesh_consistent(coords, elems)

    def test_negative_placeholder_accepted(self):
        """Q8 meshes use -1 as placeholder for unused mid-edge nodes.

        This was a bug (validation.py:78 rejected all negatives).
        After the fix, -1 placeholders are accepted.
        """
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
            [0, 32], [16, 32], [32, 32],
        ], dtype=np.float64)
        elems = np.array([
            [0, 1, 4, 3, -1, -1, -1, -1],
            [1, 2, 5, 4, -1, -1, -1, -1],
            [3, 4, 7, 6, -1, -1, -1, -1],
            [4, 5, 8, 7, -1, -1, -1, -1],
        ], dtype=np.int64)
        assert_mesh_consistent(coords, elems)  # should NOT raise

    def test_negative_non_placeholder_raises(self):
        """Negative values other than -1 should still raise."""
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 2, 3, -2, -1, -1, -1]], dtype=np.int64)
        with pytest.raises(ValueError, match="negative"):
            assert_mesh_consistent(coords, elems)
