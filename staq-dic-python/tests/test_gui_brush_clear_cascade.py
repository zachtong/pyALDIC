"""Test that clearing the ROI cascades to clearing the brush refinement mask.

User feedback: brush refinement is meaningless without an ROI on frame 0,
so clearing frame 0's ROI (via the toolbar Clear button or via right-click
->Clear on frame 0 in the image list) must also drop refine_brush_mask.
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


def _build_main_window(monkeypatch):
    """Construct a MainWindow with image dimensions stubbed out.

    Avoids hitting the filesystem by stubbing
    ImageController.read_image_rgb to return a fake RGB array.
    """
    import tempfile
    from pathlib import Path

    from staq_dic.gui.app import MainWindow
    from staq_dic.gui.controllers.image_controller import ImageController

    fake_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        ImageController, "read_image_rgb", lambda self, idx: fake_rgb
    )

    state = AppState.instance()
    tmp = Path(tempfile.gettempdir()) / "_brush_clear_test"
    tmp.mkdir(exist_ok=True)
    # Use 4 fake files so set_current_frame(2) is in range
    state.image_files = [str(tmp / f"img{i}.bmp") for i in range(4)]

    win = MainWindow()
    # Force the list/canvas to rebuild from the seeded image_files
    state.images_changed.emit()
    win._init_roi_controller()
    return win, state


def test_toolbar_clear_on_frame_0_drops_brush_mask(qapp, monkeypatch):
    """Toolbar Clear button on frame 0 must wipe both ROI and brush."""
    win, state = _build_main_window(monkeypatch)

    # Seed an ROI + brush mask
    roi = np.zeros((64, 64), dtype=bool)
    roi[10:50, 10:50] = True
    state.per_frame_rois[0] = roi
    win._roi_ctrl.mask = roi.copy()

    brush = np.zeros((64, 64), dtype=bool)
    brush[20:30, 20:30] = True
    win._brush_ctrl.mask = brush.copy()
    state.set_refine_brush_mask(brush.copy())

    state.set_current_frame(0)

    win._on_roi_clear()

    assert state.per_frame_rois.get(0) is None
    assert state.refine_brush_mask is None, (
        "Clearing the frame 0 ROI must cascade to the brush mask"
    )
    assert not win._brush_ctrl.mask.any()


def test_toolbar_clear_on_frame_2_keeps_brush_mask(qapp, monkeypatch):
    """Brush lives only on frame 0, so clearing frame 2's ROI must not affect it."""
    win, state = _build_main_window(monkeypatch)

    roi0 = np.zeros((64, 64), dtype=bool)
    roi0[10:50, 10:50] = True
    state.per_frame_rois[0] = roi0

    roi2 = np.zeros((64, 64), dtype=bool)
    roi2[15:45, 15:45] = True
    state.per_frame_rois[2] = roi2
    win._roi_ctrl.mask = roi2.copy()

    brush = np.zeros((64, 64), dtype=bool)
    brush[20:30, 20:30] = True
    win._brush_ctrl.mask = brush.copy()
    state.set_refine_brush_mask(brush.copy())

    state.set_current_frame(2)

    win._on_roi_clear()

    assert state.per_frame_rois.get(2) is None
    assert state.refine_brush_mask is not None, (
        "Clearing a non-frame-0 ROI must not touch the brush mask"
    )
    assert state.refine_brush_mask.any()


def test_image_list_clear_on_frame_0_drops_brush(qapp, monkeypatch):
    """Right-click ->Clear ROI in image list, when frame 0 selected, drops brush."""
    win, state = _build_main_window(monkeypatch)

    roi = np.zeros((64, 64), dtype=bool)
    roi[10:50, 10:50] = True
    state.per_frame_rois[0] = roi

    brush = np.zeros((64, 64), dtype=bool)
    brush[20:30, 20:30] = True
    state.set_refine_brush_mask(brush.copy())

    # Simulate selecting frame 0 in the tree and invoking the menu action
    image_list = win._left_sidebar._image_list
    item = image_list._tree.topLevelItem(0)
    image_list._tree.setCurrentItem(item)
    item.setSelected(True)
    image_list._clear_roi_selected()

    assert 0 not in state.per_frame_rois
    assert state.refine_brush_mask is None, (
        "Right-click clear on frame 0 must cascade to the brush mask"
    )
