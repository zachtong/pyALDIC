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

from staq_dic.core.data_structures import DICMesh, FrameResult, FrameSchedule

# Import the function under test
from staq_dic.core.pipeline import _compute_cumulative_displacements_tree


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
