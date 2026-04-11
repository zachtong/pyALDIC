"""Tests for GUI AppState signal behavior."""

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState, RunState


def _safe_disconnect(signal) -> None:
    """Disconnect all slots from a signal, ignoring errors if none connected."""
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
    """Reset singleton state and disconnect signals between tests."""
    state = AppState.instance()
    # Disconnect all signals to prevent cross-test contamination
    _safe_disconnect(state.images_changed)
    _safe_disconnect(state.current_frame_changed)
    _safe_disconnect(state.roi_changed)
    _safe_disconnect(state.params_changed)
    _safe_disconnect(state.run_state_changed)
    _safe_disconnect(state.progress_updated)
    _safe_disconnect(state.results_changed)
    _safe_disconnect(state.display_changed)
    _safe_disconnect(state.log_message)
    state.reset()
    yield


class TestAppState:
    def test_singleton(self, qapp):
        s1 = AppState.instance()
        s2 = AppState.instance()
        assert s1 is s2

    def test_image_folder_signal(self, qapp):
        state = AppState.instance()
        received = []
        state.images_changed.connect(lambda: received.append(True))
        state.set_image_files(["/fake/img1.tif", "/fake/img2.tif"])
        assert len(received) == 1
        assert len(state.image_files) == 2

    def test_current_frame_signal(self, qapp):
        state = AppState.instance()
        state.set_image_files([f"/img{i}.tif" for i in range(5)])
        received = []
        state.current_frame_changed.connect(lambda idx: received.append(idx))
        state.set_current_frame(3)
        assert received == [3]
        assert state.current_frame == 3

    def test_current_frame_clamps(self, qapp):
        state = AppState.instance()
        state.set_image_files([f"/img{i}.tif" for i in range(5)])
        state.set_current_frame(99)
        assert state.current_frame == 4

    def test_run_state_transitions(self, qapp):
        state = AppState.instance()
        received = []
        state.run_state_changed.connect(lambda s: received.append(s))
        state.set_run_state(RunState.RUNNING)
        state.set_run_state(RunState.PAUSED)
        state.set_run_state(RunState.DONE)
        assert received == [RunState.RUNNING, RunState.PAUSED, RunState.DONE]

    def test_roi_mask_signal(self, qapp):
        state = AppState.instance()
        received = []
        state.roi_changed.connect(lambda: received.append(True))
        mask = np.ones((100, 100), dtype=bool)
        state.set_roi_mask(mask)
        assert len(received) == 1
        assert state.roi_mask is not None

    def test_display_field_signal(self, qapp):
        state = AppState.instance()
        received = []
        state.display_changed.connect(lambda: received.append(True))
        state.set_display_field("disp_v")
        assert received == [True]
        assert state.display_field == "disp_v"

    def test_reset_clears_everything(self, qapp):
        state = AppState.instance()
        state.set_image_files(["/a.tif"])
        state.set_run_state(RunState.RUNNING)
        state.reset()
        assert state.image_files == []
        assert state.run_state == RunState.IDLE
        assert state.roi_mask is None
        assert state.results is None
