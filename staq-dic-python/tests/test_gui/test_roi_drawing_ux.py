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
