"""Tests for the unified Cancel button in the right sidebar.

Pause + Stop used to be two buttons with fuzzy semantics (Pause was
not a true pause, Stop was a hard kill). This batch merged them into
a single Cancel button. These tests pin the new contract down.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.controllers.pipeline_controller import PipelineController
from al_dic.gui.panels.right_sidebar import RightSidebar


@pytest.fixture(autouse=True)
def reset_state():
    AppState._instance = None
    yield
    AppState._instance = None


@pytest.fixture
def sidebar():
    state = AppState.instance()
    image_ctrl = ImageController(state)
    pipeline_ctrl = PipelineController(state, image_ctrl)
    return RightSidebar(pipeline_ctrl)


def test_cancel_button_exists(sidebar):
    assert hasattr(sidebar, "_cancel_btn")
    assert sidebar._cancel_btn.text() == "Cancel"


def test_no_pause_or_stop_buttons(sidebar):
    """Pause and Stop have been merged away; their attrs should be gone."""
    assert not hasattr(sidebar, "_pause_btn")
    assert not hasattr(sidebar, "_stop_btn")


def test_cancel_disabled_in_idle_state(sidebar):
    AppState.instance().set_run_state(RunState.IDLE)
    assert sidebar._cancel_btn.isEnabled() is False


def test_cancel_enabled_while_running(sidebar):
    AppState.instance().set_run_state(RunState.RUNNING)
    assert sidebar._cancel_btn.isEnabled() is True


def test_cancel_calls_controller_stop(sidebar, monkeypatch):
    """Clicking Cancel must delegate to pipeline_controller.stop()."""
    stop_calls = []
    monkeypatch.setattr(
        sidebar._pipeline_ctrl, "stop",
        lambda: stop_calls.append(True),
    )
    AppState.instance().set_run_state(RunState.RUNNING)
    sidebar._cancel_btn.click()
    assert stop_calls == [True]
