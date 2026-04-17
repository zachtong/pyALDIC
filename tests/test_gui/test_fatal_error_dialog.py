"""Tests for the fatal_error signal -> modal dialog flow.

The pipeline controller emits ``state.fatal_error(title, message)`` when
a run cannot start or the worker throws. RightSidebar listens and shows
a QMessageBox.critical modal so the user cannot miss the failure even
if the console log is scrolled out of view.

These tests verify the signal contract, not the modal itself (we do not
want to pop a real dialog in CI).
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.controllers.pipeline_controller import PipelineController


@pytest.fixture
def ctrl():
    state = AppState.instance()
    # Minimum reset so a no-images run trips the first validation
    state.image_files = []
    state.per_frame_rois = {}
    image_ctrl = ImageController(state)
    return PipelineController(state, image_ctrl)


def test_fatal_error_signal_is_on_app_state():
    """AppState must expose a fatal_error(str, str) signal."""
    state = AppState.instance()
    assert hasattr(state, "fatal_error")


def test_no_images_emits_fatal_error(ctrl):
    """Starting a run with <2 images must emit fatal_error with a
    user-facing title + message and NOT launch any worker."""
    received: list[tuple[str, str]] = []
    AppState.instance().fatal_error.connect(
        lambda title, msg: received.append((title, msg))
    )
    ctrl.start()
    assert len(received) == 1
    title, msg = received[0]
    assert title == "Cannot start analysis"
    assert "at least 2 images" in msg.lower()


def test_no_roi_emits_fatal_error(ctrl):
    """Two images but no ROI must emit fatal_error, not silently refuse."""
    state = AppState.instance()
    # Stub image list so the first guard passes
    state.image_files = ["a.tif", "b.tif"]
    # Force roi_mask None (no per-frame ROI)
    state.per_frame_rois = {}
    received: list[tuple[str, str]] = []
    state.fatal_error.connect(
        lambda title, msg: received.append((title, msg))
    )
    ctrl.start()
    assert len(received) == 1
    title, msg = received[0]
    assert title == "Cannot start analysis"
    assert "region of interest" in msg.lower()
