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
