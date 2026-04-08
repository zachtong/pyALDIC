"""Test brush refinement lockout: only locked out on non-zero frames.

Brush painting is allowed at any time on frame 0 (reference frame),
including after a completed Run.  This is consistent with how ROI edits
and mesh-parameter changes work -- the old results remain visible until
the next Run replaces them.

The only hard lock is ``current_frame != 0``:

  - Entry guard:  ``_on_brush_requested`` rejects entering brush mode on
    non-reference frames.
  - Active layer: switching frame drops the active brush tool back to
    ``select`` and emits ``drawing_finished`` so the toolbar resets.
  - Mouse layer:  stray mouse press / move on non-zero frames is rejected.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
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


def _build_main_window(monkeypatch):
    from staq_dic.gui.app import MainWindow
    from staq_dic.gui.controllers.image_controller import ImageController

    fake_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        ImageController, "read_image_rgb", lambda self, idx: fake_rgb
    )

    state = AppState.instance()
    tmp = Path(tempfile.gettempdir()) / "_brush_lockout_test"
    tmp.mkdir(exist_ok=True)
    state.image_files = [str(tmp / f"img{i}.bmp") for i in range(4)]

    win = MainWindow()
    state.images_changed.emit()
    win._init_roi_controller()

    # Seed an ROI on frame 0 so brush entry guards pass
    roi = np.zeros((64, 64), dtype=bool)
    roi[10:50, 10:50] = True
    state.per_frame_rois[0] = roi
    state.set_current_frame(0)

    return win, state


def _seed_results(state: AppState) -> None:
    """Inject a fake PipelineResult to mimic a completed Run."""
    coords = np.array(
        [[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float64
    )
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


def test_brush_entry_allowed_when_results_exist(qapp, monkeypatch):
    """Brush must be paintable on frame 0 even after a completed Run."""
    win, state = _build_main_window(monkeypatch)
    _seed_results(state)

    win._on_brush_requested("paint", 16)

    assert win._canvas_area.canvas._current_tool == "brush", (
        "Brush must be allowed on frame 0 even when results exist"
    )


def test_frame_switch_during_brush_drops_tool(qapp, monkeypatch):
    """Switching to a non-zero frame while in brush mode must reset the tool."""
    win, state = _build_main_window(monkeypatch)

    # Enter brush mode (frame 0, ROI exists, no results)
    win._on_brush_requested("paint", 16)
    assert win._canvas_area.canvas._current_tool == "brush"

    # Track drawing_finished emissions so we know the toolbar will reset
    finished_emissions: list[bool] = []
    win._canvas_area.canvas.drawing_finished.connect(
        lambda: finished_emissions.append(True)
    )

    # User navigates to frame 2
    state.set_current_frame(2)

    assert win._canvas_area.canvas._current_tool != "brush", (
        "Frame switch must drop the active brush tool"
    )
    assert len(finished_emissions) >= 1, (
        "drawing_finished must fire so the toolbar Refine button highlight resets"
    )


def test_run_completion_keeps_active_brush_tool(qapp, monkeypatch):
    """A completed Run must NOT interrupt an active brush session on frame 0."""
    win, state = _build_main_window(monkeypatch)

    # Enter brush mode
    win._on_brush_requested("paint", 16)
    assert win._canvas_area.canvas._current_tool == "brush"

    # Pipeline completes -- results land
    _seed_results(state)

    assert win._canvas_area.canvas._current_tool == "brush", (
        "Brush must stay active on frame 0 when a Run completes"
    )


def test_brush_mouse_press_rejected_on_non_zero_frame(qapp, monkeypatch):
    """Defensive: even if the tool is somehow still 'brush' on frame 2, the
    mouse handler must refuse to mutate the brush mask."""
    win, state = _build_main_window(monkeypatch)
    canvas = win._canvas_area.canvas

    # Enter brush mode legitimately
    win._on_brush_requested("paint", 16)
    assert canvas._current_tool == "brush"

    # Force the canvas tool back to brush AFTER changing frame
    # (simulates a corner case where the active-layer guard somehow missed)
    state.current_frame = 2  # bypass set_current_frame so guard doesn't fire
    canvas._current_tool = "brush"
    assert canvas._brush_ctrl is not None
    canvas._brush_ctrl.clear()

    # Synthesise a mouse press at scene (20, 20)
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPoint(20, 20),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press)

    assert not canvas._brush_ctrl.mask.any(), (
        "Brush mouse press on a non-frame-0 view must not mutate the buffer"
    )
