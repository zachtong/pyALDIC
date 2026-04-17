"""Regression: brush refinement overlay is gated to frame 1.

The cyan brush mask lives in frame-0 (aka 1-based 'frame 1') pixel
coordinates. Showing it on any later frame paints the strokes at the
wrong material points (the material has moved between frames). This
test pins that gating down: after painting on frame 0 and navigating
to frame 2, the overlay must be cleared.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState
from al_dic.gui.panels.canvas_area import CanvasArea
from al_dic.gui.controllers.image_controller import ImageController


@pytest.fixture(autouse=True)
def reset_state():
    AppState._instance = None
    yield
    AppState._instance = None


@pytest.fixture
def canvas_area():
    state = AppState.instance()
    image_ctrl = ImageController(state)
    # Return the whole CanvasArea so its internal QTimers and sub-widgets
    # outlive the test (returning just `.canvas` lets the parent get GC'd).
    return CanvasArea(image_ctrl)


def test_refine_overlay_visible_on_frame_0(canvas_area):
    canvas = canvas_area.canvas
    state = AppState.instance()
    state.current_frame = 0
    # Paint a trivial brush mask
    state.refine_brush_mask = np.zeros((64, 64), dtype=bool)
    state.refine_brush_mask[10:20, 10:20] = True
    canvas.update_refine_overlay()
    # On frame 0 the pixmap must be non-empty
    pm = canvas._refine_item.pixmap()
    assert isinstance(pm, QPixmap)
    assert not pm.isNull()


def test_refine_overlay_hidden_on_frame_2(canvas_area):
    canvas = canvas_area.canvas
    state = AppState.instance()
    state.refine_brush_mask = np.zeros((64, 64), dtype=bool)
    state.refine_brush_mask[10:20, 10:20] = True
    # Paint first on frame 0
    state.current_frame = 0
    canvas.update_refine_overlay()
    assert not canvas._refine_item.pixmap().isNull()

    # Now navigate to frame 2 — overlay must hide
    state.current_frame = 2
    canvas.update_refine_overlay()
    assert canvas._refine_item.pixmap().isNull()


def test_refine_overlay_reappears_when_returning_to_frame_0(canvas_area):
    canvas = canvas_area.canvas
    state = AppState.instance()
    state.refine_brush_mask = np.zeros((64, 64), dtype=bool)
    state.refine_brush_mask[10:20, 10:20] = True

    state.current_frame = 3
    canvas.update_refine_overlay()
    assert canvas._refine_item.pixmap().isNull()

    state.current_frame = 0
    canvas.update_refine_overlay()
    assert not canvas._refine_item.pixmap().isNull()
