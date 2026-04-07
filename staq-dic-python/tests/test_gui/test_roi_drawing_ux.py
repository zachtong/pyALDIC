"""Q1-Q6 ROI drawing UX contracts.

These tests pin the post-fix invariant that ``state.current_frame`` is the
single source of truth for "which frame's ROI am I editing".  See
``docs/plans/2026-04-06-roi-drawing-ux-fixes.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

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
        state.images_changed, state.current_frame_changed,
        state.roi_changed, state.params_changed,
        state.run_state_changed, state.progress_updated,
        state.results_changed, state.display_changed,
        state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


class _StubImageController:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._img_rgb = np.zeros((*shape, 3), dtype=np.uint8)

    def read_image_rgb(self, _idx: int) -> np.ndarray:
        return self._img_rgb.copy()

    def read_image(self, _idx: int) -> np.ndarray:
        return np.zeros(self._img_rgb.shape[:2], dtype=np.float64)


def _make_main_window(qapp):
    """Build a real MainWindow for end-to-end signal-flow tests."""
    from staq_dic.gui.app import MainWindow
    win = MainWindow()
    win._image_ctrl = _StubImageController((128, 128))
    state = AppState.instance()
    state.image_files = [f"/fake/img{i}.tif" for i in range(5)]
    state.images_changed.emit()  # triggers _init_roi_controller via stub
    return win


class TestQ1Q2_FrameSync:
    def test_image_list_edit_button_syncs_current_frame(self, qapp):
        """Clicking 'Edit' on frame K must set current_frame=K."""
        win = _make_main_window(qapp)
        state = AppState.instance()
        assert state.current_frame == 0

        # Simulate clicking "Edit" on frame 3
        win._on_roi_edit_for_frame(3)

        assert state.current_frame == 3
        assert state.roi_editing is True

    def test_draw_button_targets_current_frame(self, qapp):
        """After navigating to frame 2 and clicking Draw Rect, drawing must
        commit to frame 2 -- not the previous edit target or frame 0.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # User clicks frame 2 row in image list
        state.set_current_frame(2)

        # User clicks "Draw Rect" toolbar button
        win._on_draw_requested("rect", "add")

        # Now whatever the user draws should commit to frame 2.
        # Simulate by stamping a rect into the controller and finishing.
        win._roi_ctrl.add_rectangle(10, 10, 50, 50, "add")
        win._canvas_area.canvas._finish_drawing()

        assert 2 in state.per_frame_rois
        # And no other frame got the mask
        assert 0 not in state.per_frame_rois
        assert 1 not in state.per_frame_rois


class TestQ3_OverlaySource:
    """The blue ROI overlay must reflect per_frame_rois[current_frame],
    not the in-memory roi_ctrl.mask buffer.
    """

    def test_external_mutation_of_per_frame_rois_repaints_overlay(
        self, qapp, monkeypatch
    ):
        """Simulating a batch import that writes to per_frame_rois[3]
        directly must update the canvas overlay when current_frame=3.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()
        state.set_current_frame(3)

        canvas = win._canvas_area.canvas
        captured: dict[str, object] = {}

        # Spy on update_roi_overlay to record that it was invoked
        original = canvas.update_roi_overlay
        def spy():
            captured["called"] = True
            original()
        monkeypatch.setattr(canvas, "update_roi_overlay", spy)

        # External mutation (simulates batch import)
        new_mask = np.ones((128, 128), dtype=bool)
        state.per_frame_rois[3] = new_mask
        state.roi_changed.emit()

        # The overlay must have been refreshed
        assert captured.get("called") is True

        # And the data must have come from per_frame_rois[3], not an
        # empty buffer: new_mask is all-ones, so any pixel should be
        # painted.
        img = canvas._roi_item.pixmap().toImage()
        assert img.pixelColor(64, 64).alpha() > 0

    def test_overlay_data_source_is_per_frame_rois(self, qapp):
        """Direct contract test: update_roi_overlay must use
        per_frame_rois[current_frame] as its data source, not
        the transient roi_ctrl.mask buffer.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # Put a small mask in per_frame_rois and a completely different
        # (full-image) mask in the controller buffer.  If the overlay
        # reads the buffer, pixel (50, 50) will be painted; if it reads
        # per_frame_rois[2], pixel (50, 50) will be transparent because
        # the persisted mask only covers a 10x10 corner.
        state.set_current_frame(2)
        own_mask = np.zeros((128, 128), dtype=bool)
        own_mask[0:10, 0:10] = True
        state.per_frame_rois[2] = own_mask

        # Pollute the controller buffer with something different
        win._roi_ctrl.mask = np.ones((128, 128), dtype=bool)

        canvas = win._canvas_area.canvas
        canvas.update_roi_overlay()

        pixmap = canvas._roi_item.pixmap()
        assert not pixmap.isNull()
        img = pixmap.toImage()
        # Pixel (50, 50) should be transparent under the per_frame_rois reading
        assert img.pixelColor(50, 50).alpha() == 0
        # And a pixel inside the 10x10 corner should be painted (alpha > 0)
        assert img.pixelColor(2, 2).alpha() > 0


class TestQ1_BufferFollowsFrame:
    def test_navigating_during_roi_editing_reloads_buffer(self, qapp):
        """Editing frame 0, then arrow-key to frame 3, must reload the
        ROI controller buffer from per_frame_rois[3].
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # Seed two different masks
        m0 = np.zeros((128, 128), dtype=bool); m0[0:5, 0:5] = True
        m3 = np.zeros((128, 128), dtype=bool); m3[100:110, 100:110] = True
        state.per_frame_rois[0] = m0
        state.per_frame_rois[3] = m3

        # Enter editing on frame 0
        win._on_roi_edit_for_frame(0)
        assert win._roi_ctrl.mask[0, 0] == True
        assert win._roi_ctrl.mask[100, 100] == False

        # Navigate to frame 3 (without re-clicking Edit)
        state.set_current_frame(3)

        # Buffer must now hold m3, not m0
        assert win._roi_ctrl.mask[0, 0] == False
        assert win._roi_ctrl.mask[100, 100] == True


class TestQ4_MeshDuringEditing:
    def test_roi_editing_routes_to_preview_mesh(self, qapp, monkeypatch):
        """Even when results exist, entering ROI editing must show the
        preview mesh (which reflects the in-progress ROI), not the
        results mesh (which reflects whatever DIC ran on).
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # Stub results so the "results path" branch would normally win
        from staq_dic.core.data_structures import (
            DICMesh, FrameResult, PipelineResult,
        )
        from staq_dic.core.config import dicpara_default

        coords = np.array([[10, 10], [20, 10], [20, 20], [10, 20]],
                          dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elements)
        result = PipelineResult(
            dic_para=dicpara_default(),
            dic_mesh=mesh,
            result_disp=[FrameResult(U=np.zeros(8), U_accum=np.zeros(8))],
            result_def_grad=[],
            result_strain=[],
            result_fe_mesh_each_frame=[mesh],
        )
        state.set_results(result)

        # Seed an ROI on frame 0 and enter editing.  Order matters:
        # ``_on_frame_changed`` exits ROI editing when results exist and
        # the user navigates frames, so we must set current_frame BEFORE
        # flipping roi_editing on (mirroring _on_roi_edit_for_frame).
        state.per_frame_rois[0] = np.ones((128, 128), dtype=bool)
        state.set_current_frame(0)
        state.roi_editing = True

        canvas_area = win._canvas_area

        # Monkeypatch results path so we can detect if it was wrongly taken
        called = {"results": False}
        monkeypatch.setattr(
            canvas_area, "_show_results_mesh",
            lambda: called.__setitem__("results", True),
        )
        # Monkeypatch timer.start so we detect the preview path deterministically
        timer_started = {"v": False}
        monkeypatch.setattr(
            canvas_area._mesh_preview_timer, "start",
            lambda *a, **kw: timer_started.__setitem__("v", True),
        )

        canvas_area._refresh_mesh_overlay()

        assert timer_started["v"] is True, (
            "ROI editing on frame 0 must start the mesh preview timer"
        )
        assert called["results"] is False, (
            "Results path must NOT be taken while ROI editing is active"
        )


class TestQ5_MeshHiddenForNonRefEditing:
    def test_mesh_hidden_when_editing_non_frame_zero_roi(
        self, qapp, monkeypatch
    ):
        """The preview mesh only models frame-0 geometry.  When the
        user is editing a per-frame ROI for K != 0, hide the mesh
        entirely so they don't get a misleading overlay.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()
        # Seed frame 0 too, otherwise the old code would accidentally
        # take the final ``else`` branch (roi_mask is None) and hide.
        # With frame 0 seeded, the old code would hit the preview-timer
        # branch and display the mesh, failing this assertion.
        state.per_frame_rois[0] = np.ones((128, 128), dtype=bool)
        state.per_frame_rois[3] = np.ones((128, 128), dtype=bool)
        state.set_current_frame(3)
        state.roi_editing = True

        canvas_area = win._canvas_area
        # Block the preview timer from actually running the preview
        # pipeline inside the test; we only care about overlay visibility.
        timer_started = {"v": False}
        monkeypatch.setattr(
            canvas_area._mesh_preview_timer, "start",
            lambda *a, **kw: timer_started.__setitem__("v", True),
        )

        canvas_area._refresh_mesh_overlay()

        assert timer_started["v"] is False, (
            "Preview timer must NOT start when editing a non-frame-0 ROI"
        )
        assert canvas_area._mesh_overlay.isVisible() is False
