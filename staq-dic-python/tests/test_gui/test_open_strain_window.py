"""Tests for wiring 'Open Strain Window' between RightSidebar and MainWindow."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

app = QApplication.instance() or QApplication([])

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.controllers.pipeline_controller import PipelineController
from staq_dic.gui.panels.right_sidebar import RightSidebar

from ._helpers import make_synthetic_pipeline_result


@pytest.fixture(autouse=True)
def reset_state():
    """Each test gets a fresh AppState singleton (no leak between tests)."""
    AppState._instance = None
    yield
    AppState._instance = None


@pytest.fixture
def sidebar():
    state = AppState.instance()
    image_ctrl = ImageController(state)
    pipeline_ctrl = PipelineController(state, image_ctrl)
    return RightSidebar(pipeline_ctrl)


# ----------------------------------------------------------------------
# RightSidebar exposes the new button + signal
# ----------------------------------------------------------------------

def test_right_sidebar_has_strain_button(sidebar):
    assert hasattr(sidebar, "_strain_btn")
    assert isinstance(sidebar._strain_btn, QPushButton)
    assert "Strain" in sidebar._strain_btn.text()


def test_open_strain_window_signal_exists(sidebar):
    assert hasattr(sidebar, "open_strain_window_requested")


def test_strain_button_emits_signal(sidebar):
    received: list[bool] = []
    sidebar.open_strain_window_requested.connect(lambda: received.append(True))
    sidebar._strain_btn.click()
    assert received == [True]


# ----------------------------------------------------------------------
# MainWindow integration: lazy singleton + no-results warning
# ----------------------------------------------------------------------

def test_main_window_open_strain_warns_when_no_results():
    """Clicking the button with state.results == None must NOT crash;
    it must log a warning instead."""
    from staq_dic.gui.app import MainWindow

    win = MainWindow()
    state = AppState.instance()
    assert state.results is None

    received: list[tuple[str, str]] = []
    state.log_message.connect(lambda msg, lvl: received.append((msg, lvl)))

    win._on_open_strain_window()

    # No window created, log entry recorded.
    assert win._strain_window is None
    warns = [m for m, l in received if l == "warn"]
    assert len(warns) >= 1


def test_main_window_open_strain_lazy_singleton():
    """Opening the strain window twice reuses the same instance."""
    from staq_dic.gui.app import MainWindow

    win = MainWindow()
    state = AppState.instance()

    result, mask = make_synthetic_pipeline_result(
        n_frames=3, shear=0.01, img_shape=(128, 128), step=16,
    )
    state.results = result
    state.per_frame_rois[0] = mask.astype(bool)

    win._on_open_strain_window()
    first = win._strain_window
    assert first is not None

    win._on_open_strain_window()
    second = win._strain_window
    assert second is first  # singleton, not re-created
