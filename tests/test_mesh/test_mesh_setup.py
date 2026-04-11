"""Tests for mesh_setup module."""

import numpy as np
import pytest

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.mesh.mesh_setup import mesh_setup


class TestMeshSetup:
    def test_basic_3x3(self):
        """3x3 grid should produce 9 nodes and 4 elements."""
        x0 = np.array([0.0, 16.0, 32.0])
        y0 = np.array([0.0, 16.0, 32.0])
        para = dicpara_default(img_size=(64, 64))
        mesh = mesh_setup(x0, y0, para)

        assert mesh.coordinates_fem.shape == (9, 2)
        assert mesh.elements_fem.shape == (4, 8)
        assert mesh.elements_fem[:, :4].min() >= 0
        assert mesh.elements_fem[:, :4].max() < 9
        # Midside nodes should be -1 (no hanging nodes for uniform mesh)
        np.testing.assert_array_equal(mesh.elements_fem[:, 4:], -1)

    def test_node_coordinates(self):
        """Node coordinates should match the grid."""
        x0 = np.array([10.0, 26.0])
        y0 = np.array([5.0, 21.0, 37.0])
        para = dicpara_default(img_size=(64, 64))
        mesh = mesh_setup(x0, y0, para)

        # 2 * 3 = 6 nodes
        assert mesh.coordinates_fem.shape == (6, 2)

        # Check a few known nodes
        # Node (i=0, j=0): x=10, y=5
        assert mesh.coordinates_fem[0, 0] == 10.0
        assert mesh.coordinates_fem[0, 1] == 5.0

    def test_element_connectivity_counterclockwise(self):
        """Element corners should be counter-clockwise."""
        x0 = np.array([0.0, 16.0, 32.0])
        y0 = np.array([0.0, 16.0, 32.0])
        para = dicpara_default(img_size=(64, 64))
        mesh = mesh_setup(x0, y0, para)

        # First element: (0,0), (16,0), (16,16), (0,16)
        elem0 = mesh.elements_fem[0, :4]
        c0 = mesh.coordinates_fem[elem0]
        # Bottom-left, bottom-right, top-right, top-left
        assert c0[0, 0] < c0[1, 0]  # node0.x < node1.x
        assert c0[1, 1] < c0[2, 1]  # node1.y < node2.y
        assert c0[3, 0] < c0[2, 0]  # node3.x < node2.x

    def test_q8_midside_absent(self):
        """Q8 midside nodes (columns 4-7) should be -1 for uniform mesh."""
        x0 = np.array([0.0, 16.0, 32.0])
        y0 = np.array([0.0, 16.0, 32.0])
        para = dicpara_default(img_size=(64, 64))
        mesh = mesh_setup(x0, y0, para)

        np.testing.assert_array_equal(mesh.elements_fem[:, 4:], -1)

    def test_too_few_points(self):
        """Should raise ValueError with < 2 points."""
        x0 = np.array([0.0])
        y0 = np.array([0.0, 16.0])
        para = dicpara_default(img_size=(64, 64))
        with pytest.raises(ValueError, match="Need >= 2"):
            mesh_setup(x0, y0, para)

    def test_world_coordinates(self):
        """World coordinates should flip y-axis."""
        x0 = np.array([0.0, 16.0])
        y0 = np.array([0.0, 16.0])
        para = dicpara_default(img_size=(64, 64))
        mesh = mesh_setup(x0, y0, para)

        # For node at (0, 0): world = (0, 64 - 0) = (0, 64)
        assert mesh.coordinates_fem_world[0, 0] == 0.0
        assert mesh.coordinates_fem_world[0, 1] == 64.0

    def test_5x4_grid(self):
        """5x4 grid should produce 20 nodes and 12 elements."""
        x0 = np.arange(5) * 16.0
        y0 = np.arange(4) * 16.0
        para = dicpara_default(img_size=(64, 80))
        mesh = mesh_setup(x0, y0, para)

        assert mesh.coordinates_fem.shape == (20, 2)
        assert mesh.elements_fem.shape == (12, 8)
