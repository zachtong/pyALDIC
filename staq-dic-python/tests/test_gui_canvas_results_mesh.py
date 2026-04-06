"""Tests for canvas_area._show_results_mesh.

Focus: the deformed-mesh display path.  The key invariant is that
``PipelineResult.result_disp[i].U_accum`` is always defined on
frame 0's canonical mesh, *not* on ``result_fe_mesh_each_frame[i]``.
When incremental + refined + per-frame ROIs produce differently sized
per-frame meshes, reading ``meshes[frame]`` and adding ``U_accum`` to
it crashes with a shape-broadcast error.

Regression case: incremental + refined + per-frame ROIs with frame 0
mesh = N0 nodes and frame K mesh = NK != N0 nodes.  The display must
use the canonical (frame 0) mesh so ``coords.shape[0] == len(U_accum)/2``
unconditionally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QApplication

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import (
    DICMesh,
    FrameResult,
    PipelineResult,
)
from staq_dic.gui.app_state import AppState


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    for sig in (
        state.images_changed,
        state.current_frame_changed,
        state.roi_changed,
        state.params_changed,
        state.run_state_changed,
        state.progress_updated,
        state.results_changed,
        state.display_changed,
        state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


def _make_q4_mesh(n_nodes: int) -> DICMesh:
    """Build a dummy Q8-padded mesh with ``n_nodes`` corner nodes.

    The connectivity is a single element referencing the first four
    nodes; extra nodes are placeholders so the canonical vs per-frame
    node counts can differ cleanly.
    """
    coords = np.column_stack(
        [
            np.linspace(0, 100, n_nodes, dtype=np.float64),
            np.linspace(0, 50, n_nodes, dtype=np.float64),
        ]
    )
    # Single Q8-padded element using nodes 0..3 (cols 4-7 = -1)
    elements = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
    return DICMesh(coordinates_fem=coords, elements_fem=elements)


def _make_frame_result(n_canonical: int, scale: float = 0.1) -> FrameResult:
    """FrameResult whose ``U_accum`` is always sized to the canonical
    (frame 0) mesh — that is the unbroken invariant established by
    ``_compute_cumulative_displacements_tree``.
    """
    u = np.arange(n_canonical, dtype=np.float64) * scale
    v = np.arange(n_canonical, dtype=np.float64) * (scale * 0.5)
    U = np.empty(2 * n_canonical, dtype=np.float64)
    U[0::2] = u
    U[1::2] = v
    return FrameResult(U=U.copy(), U_accum=U.copy())


class _MeshOverlaySpy:
    """Captures the data passed through to MeshOverlay."""

    def __init__(self) -> None:
        self.last_coords: np.ndarray | None = None
        self.last_elements: np.ndarray | None = None
        self.last_valid: np.ndarray | None = None
        self.visible: bool = False

    def set_mesh(self, coords, elements, valid=None):
        self.last_coords = coords
        self.last_elements = elements
        self.last_valid = valid

    def set_view_transform(self, _vt):
        pass

    def setVisible(self, on):  # noqa: N802 — matches Qt convention
        self.visible = bool(on)


class _CanvasStub:
    def viewportTransform(self):  # noqa: N802
        return QTransform()


def _make_canvas_shim(state: AppState):
    """Bind just enough of CanvasArea to exercise _show_results_mesh."""
    from staq_dic.gui.panels.canvas_area import CanvasArea

    class _Shim:
        def __init__(self, state):
            self._state = state
            self._mesh_overlay = _MeshOverlaySpy()
            self._canvas = _CanvasStub()
            self._hover_mesh_coords = None
            self._hover_valid = None

    # Bind the methods under test
    _Shim._show_results_mesh = CanvasArea._show_results_mesh
    _Shim._node_valid_mask = CanvasArea._node_valid_mask
    return _Shim(state)


def _install_result(
    state: AppState,
    *,
    frame0_nodes: int,
    frame_node_counts: list[int],
    current_frame: int,
    show_deformed: bool,
) -> PipelineResult:
    """Build a PipelineResult with per-frame meshes of specified sizes.

    ``frame_node_counts[k]`` is the node count of
    ``result_fe_mesh_each_frame[k]`` (k=0 is frame 0).  Every
    ``U_accum`` is sized to ``2 * frame0_nodes`` — the canonical mesh
    contract.
    """
    meshes = [_make_q4_mesh(n) for n in frame_node_counts]

    # result_disp has length (n_frames - 1); deformed frame K uses
    # entry K-1.  We construct it for all deformed frames 1..N.
    n_deformed = len(frame_node_counts) - 1
    disps = [_make_frame_result(frame0_nodes) for _ in range(n_deformed)]

    result = PipelineResult(
        dic_para=dicpara_default(),
        dic_mesh=meshes[0],   # canonical = frame 0
        result_disp=disps,
        result_def_grad=[],
        result_strain=[],
        result_fe_mesh_each_frame=meshes,
    )

    state.set_results(result)
    state.image_files = [f"f{i}.tif" for i in range(len(frame_node_counts))]
    state.current_frame = current_frame
    state.show_deformed = show_deformed
    state.show_mesh = True
    return result


class TestResultsMeshCanonical:
    """Pin: mesh overlay must always use the frame-0 canonical mesh."""

    def test_incremental_refined_mismatched_nodes_does_not_crash(self, qapp):
        """The historical crash case.

        Frame 0: 10 canonical nodes.  Frame 1: 15 nodes (different per-frame
        mesh due to refinement + different mask).  3 images → result_disp
        has length 2, so ``current_frame=2`` (→ ``frame=1``) enters the
        deformed branch.  Pre-fix: per-frame coords(15) + U_accum(10) →
        ``ValueError`` broadcast mismatch.  Post-fix: canonical mesh is
        used, coords(10) + U_accum(10) → success.
        """
        state = AppState.instance()
        _install_result(
            state,
            frame0_nodes=10,
            frame_node_counts=[10, 15, 12],
            current_frame=2,  # image index 2 → frame=1 → result_disp[1]
            show_deformed=True,
        )

        shim = _make_canvas_shim(state)
        # This used to raise ValueError: operands could not be broadcast
        shim._show_results_mesh()

        overlay = shim._mesh_overlay
        assert overlay.last_coords is not None
        # Canonical (frame 0) mesh has 10 nodes — that is what the
        # overlay must receive regardless of the current frame index.
        assert overlay.last_coords.shape == (10, 2)
        assert overlay.visible is True

    def test_deformed_coords_equal_canonical_plus_u_accum(self, qapp):
        """Deformed coords must equal canonical_coords + U_accum, so the
        overlay draws where frame 0's nodes actually ended up.
        """
        state = AppState.instance()
        # 4 images → result_disp length 3 → current_frame=3 → frame=2
        result = _install_result(
            state,
            frame0_nodes=10,
            frame_node_counts=[10, 15, 12, 8],
            current_frame=3,
            show_deformed=True,
        )

        shim = _make_canvas_shim(state)
        shim._show_results_mesh()

        canonical = result.result_fe_mesh_each_frame[0].coordinates_fem
        u_accum = result.result_disp[2].U_accum
        expected = canonical + np.column_stack([u_accum[0::2], u_accum[1::2]])

        np.testing.assert_allclose(shim._mesh_overlay.last_coords, expected)

    def test_reference_view_uses_canonical_mesh(self, qapp):
        """Reference (undeformed) view should also use the canonical mesh
        so the topology is stable across frames when the user toggles
        the deformed switch.

        ``current_frame=2`` selects image index 2; pre-fix this would
        show ``meshes[2]`` (12 nodes); post-fix it must show
        ``meshes[0]`` (10 nodes) like every other frame.
        """
        state = AppState.instance()
        result = _install_result(
            state,
            frame0_nodes=10,
            frame_node_counts=[10, 15, 12],
            current_frame=2,
            show_deformed=False,
        )

        shim = _make_canvas_shim(state)
        shim._show_results_mesh()

        canonical = result.result_fe_mesh_each_frame[0].coordinates_fem
        np.testing.assert_allclose(
            shim._mesh_overlay.last_coords, canonical
        )

    def test_matching_node_counts_still_work(self, qapp):
        """Accumulative-style case: every per-frame mesh already equals
        the canonical mesh.  Must keep working after the fix.
        """
        state = AppState.instance()
        _install_result(
            state,
            frame0_nodes=12,
            frame_node_counts=[12, 12, 12],
            current_frame=2,
            show_deformed=True,
        )

        shim = _make_canvas_shim(state)
        shim._show_results_mesh()

        assert shim._mesh_overlay.last_coords.shape == (12, 2)
        assert shim._mesh_overlay.visible is True
