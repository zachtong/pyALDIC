"""Unit tests for tree-based cumulative displacement composition.

Tests the _compute_cumulative_displacements_tree function with
analytically verifiable displacement fields on a simple regular grid.

Key math:
    For chain A -> B -> C on the mesh:
        coords_B = coords_A + u_AB(coords_A)
        coords_C = coords_B + u_BC(coords_B)
        u_AC     = coords_C - coords_A

    For uniform translations this simplifies to vector addition.
    For spatially-varying fields, scattered interpolation is required.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from al_dic.core.data_structures import DICMesh, FrameResult, FrameSchedule

# Import the function under test
from al_dic.core.pipeline import _compute_cumulative_displacements_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regular_mesh(nx: int = 5, ny: int = 5, step: float = 16.0) -> DICMesh:
    """Create a regular grid mesh for testing."""
    xs = np.arange(nx, dtype=np.float64) * step
    ys = np.arange(ny, dtype=np.float64) * step
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Minimal Q4 elements (not used by composition, but required by DICMesh)
    elements = np.empty((0, 8), dtype=np.int64)

    return DICMesh(
        coordinates_fem=coords,
        elements_fem=elements,
        x0=xs,
        y0=ys,
    )


def _make_frame_result(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    ref_frame: int = 0,
) -> FrameResult:
    """Build FrameResult from u, v arrays."""
    n = len(u)
    U = np.empty(2 * n, dtype=np.float64)
    U[0::2] = u
    U[1::2] = v
    return FrameResult(U=U, ref_frame=ref_frame)


# ---------------------------------------------------------------------------
# Tests: zero displacement
# ---------------------------------------------------------------------------


class TestZeroDisplacement:
    """Zero displacement composition should produce zero cumulative."""

    def test_single_frame_zero(self):
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]
        u_zero = np.zeros(n)
        v_zero = np.zeros(n)

        result_disp = [_make_frame_result(u_zero, v_zero)]
        result_mesh = [DICMesh(
            coordinates_fem=mesh.coordinates_fem.copy(),
            elements_fem=mesh.elements_fem.copy(),
        )]

        schedule = FrameSchedule(ref_indices=(0,))
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 2, schedule,
        )

        U_accum = result[0].U_accum
        assert U_accum is not None
        np.testing.assert_allclose(U_accum, 0.0, atol=1e-12)

    def test_chain_of_zeros(self):
        """3 frames, all zero displacement, incremental."""
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]
        u_zero = np.zeros(n)
        v_zero = np.zeros(n)

        result_disp = [
            _make_frame_result(u_zero, v_zero, ref_frame=0),
            _make_frame_result(u_zero, v_zero, ref_frame=1),
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
        ]

        schedule = FrameSchedule.from_mode("incremental", 3)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 3, schedule,
        )

        for i in range(2):
            assert result[i].U_accum is not None
            np.testing.assert_allclose(result[i].U_accum, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: uniform translation chain
# ---------------------------------------------------------------------------


class TestTranslationChain:
    """Uniform translations compose by simple addition."""

    def test_two_frame_translation(self):
        """Single pair: accumulative, u=2.5, v=-1.0."""
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]
        u = np.full(n, 2.5)
        v = np.full(n, -1.0)

        result_disp = [_make_frame_result(u, v)]
        result_mesh = [DICMesh(
            coordinates_fem=mesh.coordinates_fem.copy(),
            elements_fem=mesh.elements_fem.copy(),
        )]

        schedule = FrameSchedule(ref_indices=(0,))
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 2, schedule,
        )

        U_accum = result[0].U_accum
        np.testing.assert_allclose(U_accum[0::2], 2.5, atol=1e-10)
        np.testing.assert_allclose(U_accum[1::2], -1.0, atol=1e-10)

    def test_incremental_chain_3_frames(self):
        """Incremental: frame1->frame2 = (1,0), frame2->frame3 = (1,0).

        Cumulative: frame2 = (1,0), frame3 = (2,0).
        """
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]

        result_disp = [
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=0),
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=1),
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
        ]

        schedule = FrameSchedule.from_mode("incremental", 3)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 3, schedule,
        )

        # Frame 2: cumulative = (1, 0)
        np.testing.assert_allclose(result[0].U_accum[0::2], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[0].U_accum[1::2], 0.0, atol=1e-10)

        # Frame 3: cumulative = (2, 0)
        np.testing.assert_allclose(result[1].U_accum[0::2], 2.0, atol=1e-10)
        np.testing.assert_allclose(result[1].U_accum[1::2], 0.0, atol=1e-10)

    def test_accumulative_3_frames(self):
        """Accumulative: both frames directly reference frame 0.

        U_accum = U for each frame.
        """
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]

        result_disp = [
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=0),
            _make_frame_result(np.full(n, 2.0), np.zeros(n), ref_frame=0),
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
        ]

        schedule = FrameSchedule.from_mode("accumulative", 3)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 3, schedule,
        )

        np.testing.assert_allclose(result[0].U_accum[0::2], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1].U_accum[0::2], 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: tree branching
# ---------------------------------------------------------------------------


class TestTreeBranching:
    """Test branching schedules (not just linear chains)."""

    def test_skip2_keyframe(self):
        """Schedule (0, 0, 2): frames 1,2 ref 0; frame 3 refs frame 2.

        If frame 1 has u=1.0 and frame 2 has u=2.0 (both vs frame 0),
        and frame 3 has u=0.5 vs frame 2, then:
            cumulative frame 3 = u_02 + u_23 = 2.0 + 0.5 = 2.5
        """
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]

        result_disp = [
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=0),
            _make_frame_result(np.full(n, 2.0), np.zeros(n), ref_frame=0),
            _make_frame_result(np.full(n, 0.5), np.zeros(n), ref_frame=2),
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
        ]

        schedule = FrameSchedule(ref_indices=(0, 0, 2))
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 4, schedule,
        )

        # Frame 1: directly refs 0, cumulative = 1.0
        np.testing.assert_allclose(result[0].U_accum[0::2], 1.0, atol=1e-10)
        # Frame 2: directly refs 0, cumulative = 2.0
        np.testing.assert_allclose(result[1].U_accum[0::2], 2.0, atol=1e-10)
        # Frame 3: refs frame 2, cumulative = 2.0 + 0.5 = 2.5
        np.testing.assert_allclose(result[2].U_accum[0::2], 2.5, atol=1e-10)

    def test_mixed_schedule(self):
        """Schedule (0, 1, 0): frame 1 refs 0, frame 2 refs 1, frame 3 refs 0.

        Translation: all increments = (1, 0).
        Frame 1 cumulative: 1.0
        Frame 2 cumulative: 1.0 + 1.0 = 2.0 (chained)
        Frame 3 cumulative: 1.0 (direct from 0)
        """
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]

        result_disp = [
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=0),
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=1),
            _make_frame_result(np.full(n, 1.0), np.zeros(n), ref_frame=0),
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
        ]

        schedule = FrameSchedule(ref_indices=(0, 1, 0))
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 4, schedule,
        )

        np.testing.assert_allclose(result[0].U_accum[0::2], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1].U_accum[0::2], 2.0, atol=1e-10)
        np.testing.assert_allclose(result[2].U_accum[0::2], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: spatially-varying displacement
# ---------------------------------------------------------------------------


class TestSpatiallyVarying:
    """Composition with non-uniform displacement fields.

    For affine expansion u(x) = eps * (x - cx), the composition
    of two identical steps is:
        u_02(X) = eps * (X - cx) + eps * ((X + eps*(X-cx)) - cx)
                = eps * (X - cx) + eps * (X - cx) + eps^2 * (X - cx)
                = (2*eps + eps^2) * (X - cx)
    """

    def test_affine_chain_composition(self):
        """Two incremental affine steps should compose correctly."""
        mesh = _make_regular_mesh(nx=8, ny=8, step=10.0)
        n = mesh.coordinates_fem.shape[0]
        cx = 35.0  # approximate center

        eps = 0.02
        x = mesh.coordinates_fem[:, 0]

        # Each incremental step: u = eps * (x - cx)
        u_inc = eps * (x - cx)
        v_inc = np.zeros(n)

        result_disp = [
            _make_frame_result(u_inc, v_inc, ref_frame=0),
            _make_frame_result(u_inc.copy(), v_inc.copy(), ref_frame=1),
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy()),
        ]

        schedule = FrameSchedule.from_mode("incremental", 3)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 3, schedule,
        )

        # Frame 2: cumulative should be (2*eps + eps^2) * (x - cx)
        expected_u = (2 * eps + eps**2) * (x - cx)
        actual_u = result[1].U_accum[0::2]

        # Scattered interpolation introduces small errors
        np.testing.assert_allclose(actual_u, expected_u, atol=0.05, rtol=0.01)


# ---------------------------------------------------------------------------
# Tests: accumulative mode equivalence
# ---------------------------------------------------------------------------


class TestAccumulativeEquivalence:
    """For accumulative schedule, U_accum should equal U directly."""

    def test_direct_reference_no_composition(self):
        """All frames reference frame 0 -> U_accum = U."""
        mesh = _make_regular_mesh()
        n = mesh.coordinates_fem.shape[0]

        disps = [
            np.full(n, 1.0),
            np.full(n, 3.0),
            np.full(n, 5.0),
        ]

        result_disp = [
            _make_frame_result(d, np.zeros(n), ref_frame=0) for d in disps
        ]
        result_mesh = [
            DICMesh(coordinates_fem=mesh.coordinates_fem.copy(),
                    elements_fem=mesh.elements_fem.copy())
            for _ in range(3)
        ]

        schedule = FrameSchedule.from_mode("accumulative", 4)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_mesh, 4, schedule,
        )

        for i, d in enumerate(disps):
            np.testing.assert_allclose(
                result[i].U_accum[0::2], d, atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Tests: crack-aware composition (masks + crack_radius)
# ---------------------------------------------------------------------------


def _make_cut_mesh(h=120, w=200, step=16.0):
    """Regular mesh with REAL Q4 elements, plus a variant with a horizontal
    band of elements removed (an open crack cut)."""
    xs = np.arange(16.0, w - 15.0, step)
    ys = np.arange(16.0, h - 15.0, step)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    nx, ny = len(xs), len(ys)
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    elems = np.full((len(ii), 8), -1, np.int64)
    elems[:, 0] = ii * ny + jj
    elems[:, 1] = (ii + 1) * ny + jj
    elems[:, 2] = (ii + 1) * ny + (jj + 1)
    elems[:, 3] = ii * ny + (jj + 1)
    return coords, elems, xs, ys


class TestCrackAwareComposition:
    """masks + crack_radius: near-gap increments come from the crack-cut
    elements (never mixing faces); deep-gap nodes die (NaN); masks=None or an
    all-ones mask keeps the legacy result bit-for-bit."""

    def test_all_ones_mask_is_byte_identical_to_legacy(self):
        coords, elems, _, _ = _make_cut_mesh()
        n = len(coords)
        rng = np.random.default_rng(11)
        u = rng.uniform(-2, 2, n)
        v = rng.uniform(-2, 2, n)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elems)

        def run(**kw):
            disp = [
                _make_frame_result(u, v, ref_frame=0),
                _make_frame_result(u * 0.5, v * 0.5, ref_frame=1),
            ]
            meshes = [
                DICMesh(coordinates_fem=coords.copy(),
                        elements_fem=elems.copy()),
                DICMesh(coordinates_fem=coords.copy(),
                        elements_fem=elems.copy()),
            ]
            schedule = FrameSchedule.from_mode("incremental", 3)
            return _compute_cumulative_displacements_tree(
                disp, meshes, 3, schedule, **kw,
            )

        legacy = run()
        ones = [np.ones((120, 200))] * 3
        aware = run(masks=ones, crack_radius=32.0)
        for i in range(2):
            np.testing.assert_array_equal(
                legacy[i].U_accum, aware[i].U_accum,
            )

    def _crack_setup(self):
        """2 deformed frames, incremental.  Pair 1->2 opens a crack: its
        reference mask (frame 1) has an open band and its mesh is cut there.
        The pair-1->2 increment is a step across the crack."""
        coords, elems, xs, ys = _make_cut_mesh()
        n = len(coords)
        h, w = 120, 200
        cy = 64.0  # crack line: between node rows 48 and 80 (node row 64 dies)

        # Frame-0 and frame-1 masks: all material (crack not yet open).
        m_full = np.ones((h, w))
        # Frame-1 mask (reference of pair 1->2): open gap rows 56..72.
        m_open = np.ones((h, w))
        m_open[56:73, :] = 0.0

        # Pair 0->1: zero increment, full mesh.
        u01 = np.zeros(n)
        v01 = np.zeros(n)

        # Pair 1->2: rigid faces step across the crack; mesh cut at the gap.
        v12 = np.where(coords[:, 1] < cy, -6.0, 6.0)
        u12 = np.zeros(n)
        cent = coords[elems[:, :4]].mean(axis=1)
        cut = (np.abs(cent[:, 1] - cy) < 16.0)
        elems_cut = elems[~cut]

        disp = [
            _make_frame_result(u01, v01, ref_frame=0),
            _make_frame_result(u12, v12, ref_frame=1),
        ]
        meshes = [
            DICMesh(coordinates_fem=coords.copy(), elements_fem=elems.copy()),
            DICMesh(coordinates_fem=coords.copy(),
                    elements_fem=elems_cut.copy()),
        ]
        schedule = FrameSchedule.from_mode("incremental", 3)
        masks = [m_full, m_open, m_open]
        return coords, disp, meshes, schedule, masks, cy

    def test_face_nodes_get_face_pure_values(self):
        coords, disp, meshes, schedule, masks, cy = self._crack_setup()
        result = _compute_cumulative_displacements_tree(
            disp, meshes, 3, schedule, masks=masks, crack_radius=32.0,
        )
        v_acc = result[1].U_accum[1::2]
        # Node rows adjacent to the gap (y=48 upper, y=80 lower) must carry
        # the pure face value -- NOT a cross-gap blend.
        upper = np.isclose(coords[:, 1], 48.0)
        lower = np.isclose(coords[:, 1], 80.0)
        np.testing.assert_allclose(v_acc[upper], -6.0, atol=1e-9)
        np.testing.assert_allclose(v_acc[lower], 6.0, atol=1e-9)

    def test_mid_gap_node_dies_and_stays_dead(self):
        coords, disp, meshes, schedule, masks, cy = self._crack_setup()
        result = _compute_cumulative_displacements_tree(
            disp, meshes, 3, schedule, masks=masks, crack_radius=32.0,
        )
        mid = np.isclose(coords[:, 1], 64.0)  # node row inside the open gap
        v2 = result[1].U_accum[1::2]
        assert np.isnan(v2[mid]).all()
        # The frame-1 mask already shows the open band, so the child-side
        # death check kills the mid-band nodes at frame 1 as well.
        v1 = result[0].U_accum[1::2]
        assert np.isnan(v1[mid]).all()
        # Off-band nodes stay finite on every frame.
        off = ~mid
        assert np.isfinite(result[0].U_accum[1::2][off]).all()
        assert np.isfinite(v2[off]).all()

    def test_born_dead_nodes_masked_at_frame0(self):
        coords, disp, meshes, schedule, masks, cy = self._crack_setup()
        # Frame-0 mask now masks the row-64 band from the start.
        m0 = np.ones((120, 200))
        m0[62:67, :] = 0.0
        masks = [m0] + masks[1:]
        result = _compute_cumulative_displacements_tree(
            disp, meshes, 3, schedule, masks=masks, crack_radius=32.0,
        )
        mid = np.isclose(coords[:, 1], 64.0)
        for i in range(2):
            assert np.isnan(result[i].U_accum[0::2][mid]).all()
            assert np.isnan(result[i].U_accum[1::2][mid]).all()
        # Off-band nodes unaffected.
        off = np.isclose(coords[:, 1], 16.0)
        assert np.isfinite(result[1].U_accum[1::2][off]).all()


# ---------------------------------------------------------------------------
# Tests: bounded Delaunay cache (D24)
# ---------------------------------------------------------------------------


def _cycling_lattice_run(n_pairs: int = 7):
    """Accumulative run whose pair meshes cycle through 3 distinct lattices.

    Per-reference grids give pairs their own lattices; here pair i lives on
    lattice i % 3 (jittered, so frame-0 queries are not nodes), so a
    one-entry cache must rebuild every triangulation while the default
    cache keeps all three.
    """
    rng = np.random.default_rng(7)
    base = _make_regular_mesh(6, 6, 16.0).coordinates_fem
    lattices = [base + rng.uniform(-3.0, 3.0, base.shape) for _ in range(3)]
    meshes, disp = [], []
    for i in range(n_pairs):
        pts = lattices[i % 3]
        meshes.append(DICMesh(
            coordinates_fem=pts.copy(),
            elements_fem=np.empty((0, 8), dtype=np.int64),
        ))
        u = 0.01 * (i + 1) * pts[:, 0] + 0.3
        v = -0.02 * pts[:, 1] + 0.1 * i
        disp.append(_make_frame_result(u, v, ref_frame=0))
    schedule = FrameSchedule.from_mode("accumulative", n_pairs + 1)
    return disp, meshes, schedule, n_pairs + 1


def _compose_counting_delaunay(monkeypatch, cache_size=None):
    import al_dic.core.pipeline as pipeline_mod
    import al_dic.utils.interpolation as interp_mod

    builds: list[int] = []
    real_delaunay = interp_mod.Delaunay

    def counting(points, *args, **kwargs):
        builds.append(len(points))
        return real_delaunay(points, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(interp_mod, "Delaunay", counting)
        if cache_size is not None:
            mp.setattr(pipeline_mod, "_TRI_CACHE_SIZE", cache_size)
        disp, meshes, schedule, n_frames = _cycling_lattice_run()
        result = _compute_cumulative_displacements_tree(
            disp, meshes, n_frames, schedule,
        )
    return [r.U_accum for r in result], len(builds)


def test_tri_cache_bound_is_value_neutral(monkeypatch):
    """A one-entry cache rebuilds every triangulation, values unchanged."""
    default, n_default = _compose_counting_delaunay(monkeypatch)
    tiny, n_tiny = _compose_counting_delaunay(monkeypatch, cache_size=1)
    assert n_default == 3  # one build per distinct lattice
    assert n_tiny == 7  # the 3-lattice cycle evicts every entry
    for a, b in zip(default, tiny):
        np.testing.assert_array_equal(a, b)
