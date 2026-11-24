"""The config overlay at the top-left of the preview follows the sidebar."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState
from al_dic.gui.widgets.canvas_config_overlay import CanvasConfigOverlay


@pytest.fixture(autouse=True)
def reset_state():
    AppState._instance = None
    yield
    AppState._instance = None


def _overlay() -> tuple[AppState, CanvasConfigOverlay]:
    state = AppState.instance()
    state.image_files = ["a.png", "b.png"]
    return state, CanvasConfigOverlay()


def test_overlay_shows_the_shape_function():
    state, overlay = _overlay()
    assert overlay._shape_lbl.text() == "1st order"

    state.set_param("shape_function_order", "second")
    assert overlay._shape_lbl.text() == "2nd order"


def test_overlay_shows_first_order_for_aldic():
    """AL-DIC always runs first order, so that is what the overlay reports
    even if second order is still selected for Local DIC."""
    state, overlay = _overlay()
    state.set_param("shape_function_order", "second")

    state.set_param("use_admm", True)
    assert overlay._shape_lbl.text() == "1st order"

    state.set_param("use_admm", False)
    assert overlay._shape_lbl.text() == "2nd order"
