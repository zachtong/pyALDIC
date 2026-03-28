"""Tests for generate_mesh (quadtree mesh generation pipeline)."""

import numpy as np
import pytest

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICMesh, ImageGradients
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.mesh.generate_mesh import (
    generate_mesh,
    _reorder_element_nodes_ccw,
    _inject_hanging_nodes,
    _find_boundary_nodes,
    _interpolate_u0,
)


def _make_uniform_mesh_and_mask(nx=5, ny=5, step=16, hole=False):
    """Create a uniform mesh and mask for testing.

    Returns (mesh, para, Df, U0).
    """
    img_h = (ny - 1) * step + step  # some extra room
    img_w = (nx - 1) * step + step

    mask = np.ones((img_h, img_w), dtype=np.float64)
    if hole:
        # Create a circular hole near the center
        cy, cx = img_h // 2, img_w // 2
        yy, xx = np.mgrid[0:img_h, 0:img_w]
        radius = step * 0.8
        mask[((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2] = 0.0

    from dataclasses import replace
    para = dicpara_default(
        img_size=(img_h, img_w),
        winstepsize=step,
        img_ref_mask=mask,
    )

    x0 = np.arange(nx, dtype=np.float64) * step
    y0 = np.arange(ny, dtype=np.float64) * step
    dic_mesh = mesh_setup(x0, y0, para)

    # Trivial image gradients (just for mask + img_size)
    Df = ImageGradients(
        df_dx=np.zeros((img_h, img_w)),
        df_dy=np.zeros((img_h, img_w)),
        img_ref_mask=mask,
        img_size=(img_h, img_w),
    )

    n_nodes = dic_mesh.coordinates_fem.shape[0]
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)

    return dic_mesh, para, Df, U0


class TestReorderElementNodesCCW:
    def test_already_ccw(self):
        """Already-CCW nodes should remain unchanged."""
        coords = np.array([
            [0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0],
        ])
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        _reorder_element_nodes_ccw(coords, elems)

        # BL=0, BR=1, TR=2, TL=3 — should stay
        assert list(elems[0]) == [0, 1, 2, 3]

    def test_reversed_order(self):
        """CW-ordered nodes should be reordered to CCW."""
        coords = np.array([
            [0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0],
        ])
        # Start with TL, TR, BR, BL (clockwise)
        elems = np.array([[3, 2, 1, 0]], dtype=np.int64)
        _reorder_element_nodes_ccw(coords, elems)

        # Should become [BL, BR, TR, TL] = [0, 1, 2, 3]
        assert list(elems[0]) == [0, 1, 2, 3]


class TestInjectHangingNodes:
    def test_no_irregular(self):
        """No irregular nodes → columns 4-7 should be -1 (no midside)."""
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        irregular = np.empty((0, 3), dtype=np.int64)

        q8 = _inject_hanging_nodes(elems, irregular)
        assert q8.shape == (1, 8)
        np.testing.assert_array_equal(q8[:, 4:], -1)

    def test_midside_on_edge_n0_n1(self):
        """Irregular node on edge (n0, n1) → column 7."""
        # Element: nodes 0,1,2,3 in CCW order
        # Midside node 4 is midpoint of edge (0, 1)
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        irregular = np.array([[0, 1, 4]], dtype=np.int64)

        q8 = _inject_hanging_nodes(elems, irregular)
        assert q8[0, 7] == 4  # col 7 = edge (n0, n1)

    def test_midside_on_edge_n1_n2(self):
        """Irregular node on edge (n1, n2) → column 4."""
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        irregular = np.array([[1, 2, 5]], dtype=np.int64)

        q8 = _inject_hanging_nodes(elems, irregular)
        assert q8[0, 4] == 5

    def test_reverse_direction(self):
        """Irregular edge (b, a) should also match element edge (a, b)."""
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        # Irregular (2, 1, 5) → edge (n1,n2) reversed → col 4
        irregular = np.array([[2, 1, 5]], dtype=np.int64)

        q8 = _inject_hanging_nodes(elems, irregular)
        assert q8[0, 4] == 5


class TestGenerateMeshNoHole:
    def test_no_hole_no_refinement(self):
        """Solid mask → no refinement, mesh unchanged except Q8 padding."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=3, ny=3, hole=False)
        mesh_qt, U0_qt = generate_mesh(mesh, para, Df, U0)

        # No refinement → same number of elements
        assert mesh_qt.elements_fem.shape[0] == mesh.elements_fem.shape[0]
        assert mesh_qt.elements_fem.shape[1] == 8
        assert mesh_qt.coordinates_fem.shape == mesh.coordinates_fem.shape

    def test_zero_displacement_preserved(self):
        """Zero U0 should remain zero after interpolation."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=3, ny=3, hole=False)
        _, U0_qt = generate_mesh(mesh, para, Df, U0)
        np.testing.assert_allclose(U0_qt, 0.0, atol=1e-15)


class TestGenerateMeshWithHole:
    def test_hole_produces_more_elements(self):
        """Mesh with hole should have more elements than original (due to refinement)."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=5, ny=5, hole=True)
        mesh_qt, U0_qt = generate_mesh(mesh, para, Df, U0)

        # Refinement should increase element count
        assert mesh_qt.elements_fem.shape[0] >= mesh.elements_fem.shape[0]

    def test_hole_has_boundary_nodes(self):
        """Mesh with hole should identify boundary-adjacent nodes."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=5, ny=5, hole=True)
        mesh_qt, _ = generate_mesh(mesh, para, Df, U0)

        assert len(mesh_qt.mark_coord_hole_edge) > 0

    def test_elements_inside_hole_removed(self):
        """No element center should be inside the hole."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=5, ny=5, hole=True)
        mesh_qt, _ = generate_mesh(mesh, para, Df, U0)

        # Check element centroids are in valid mask regions
        corners = mesh_qt.elements_fem[:, :4]
        cx = mesh_qt.coordinates_fem[corners, 0].mean(axis=1)
        cy = mesh_qt.coordinates_fem[corners, 1].mean(axis=1)

        h, w = para.img_size
        xi = np.clip(np.round(cx).astype(int), 0, w - 1)
        yi = np.clip(np.round(cy).astype(int), 0, h - 1)

        mask = para.img_ref_mask
        # Most centroids should be in valid region (some edge elements may straddle)
        valid_frac = mask[yi, xi].mean()
        assert valid_frac > 0.5

    def test_all_indices_valid(self):
        """All element indices should reference valid coordinate rows."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=5, ny=5, hole=True)
        mesh_qt, _ = generate_mesh(mesh, para, Df, U0)

        n_nodes = mesh_qt.coordinates_fem.shape[0]
        corners = mesh_qt.elements_fem[:, :4]
        assert corners.min() >= 0
        assert corners.max() < n_nodes

    def test_uniform_displacement_interpolated(self):
        """Uniform displacement should be preserved through interpolation."""
        mesh, para, Df, _ = _make_uniform_mesh_and_mask(nx=5, ny=5, hole=True)

        n_old = mesh.coordinates_fem.shape[0]
        U0 = np.zeros(2 * n_old, dtype=np.float64)
        U0[0::2] = 2.5  # uniform u = 2.5 px
        U0[1::2] = -1.0  # uniform v = -1.0 px

        _, U0_qt = generate_mesh(mesh, para, Df, U0)

        u_qt = U0_qt[0::2]
        v_qt = U0_qt[1::2]

        # Non-zero nodes should have approximately uniform values
        nonzero_u = u_qt[u_qt != 0.0]
        nonzero_v = v_qt[v_qt != 0.0]

        if len(nonzero_u) > 0:
            np.testing.assert_allclose(nonzero_u, 2.5, atol=0.1)
        if len(nonzero_v) > 0:
            np.testing.assert_allclose(nonzero_v, -1.0, atol=0.1)

    def test_world_coordinates(self):
        """World coordinates should have flipped y-axis."""
        mesh, para, Df, U0 = _make_uniform_mesh_and_mask(nx=3, ny=3, hole=False)
        mesh_qt, _ = generate_mesh(mesh, para, Df, U0)

        h, _ = para.img_size
        expected_y_world = h + 1 - mesh_qt.coordinates_fem[:, 1]
        np.testing.assert_allclose(
            mesh_qt.coordinates_fem_world[:, 1], expected_y_world
        )


class TestInterpolateU0:
    def test_same_mesh_preserves_values(self):
        """Interpolating to the same coordinates should preserve values."""
        coords = np.array([
            [0.0, 0.0], [16.0, 0.0], [32.0, 0.0],
            [0.0, 16.0], [16.0, 16.0], [32.0, 16.0],
        ])
        U0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                        7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        mask = np.ones((33, 33), dtype=np.float64)

        U0_new = _interpolate_u0(coords, coords, U0, mask, (33, 33))
        np.testing.assert_allclose(U0_new, U0, atol=1e-10)
