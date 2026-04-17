"""Tests for ParamPanel -- main DIC parameter editor.

Focus: the Search Range spinbox was promoted from the ADVANCED
collapsible section to the main panel so users can find it without
expanding Advanced. These tests pin that promotion down.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState
from al_dic.gui.widgets.param_panel import ParamPanel


@pytest.fixture
def panel():
    # AppState is a singleton; reset the one field we mutate to a known value
    AppState.instance().search_range = 20
    return ParamPanel()


def test_search_range_spinbox_exists(panel):
    """Search Range must be a direct child widget of the main ParamPanel."""
    assert hasattr(panel, "_search_range"), (
        "ParamPanel must expose _search_range in the main section, not only "
        "inside the collapsed ADVANCED / InitGuessWidget panel."
    )


def test_search_range_default_matches_state(panel):
    assert panel._search_range.value() == AppState.instance().search_range


def test_search_range_range_and_step(panel):
    """4..512 px range (matches historical InitGuessWidget range) in px steps of 2."""
    assert panel._search_range.minimum() == 4
    assert panel._search_range.maximum() == 512
    assert panel._search_range.singleStep() == 2
    assert panel._search_range.suffix().strip() == "px"


def test_changing_search_range_updates_state(panel):
    panel._search_range.setValue(60)
    assert AppState.instance().search_range == 60
    panel._search_range.setValue(24)
    assert AppState.instance().search_range == 24
