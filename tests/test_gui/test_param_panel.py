"""Tests for ParamPanel -- main DIC parameter editor.

Focus: the Search Range spinbox was promoted from the ADVANCED
collapsible section to the main panel so users can find it without
expanding Advanced. These tests pin that promotion down.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QRect, Qt
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


@pytest.fixture
def starting_points_mode():
    """The search label reads "Starting Point Search" in this mode."""
    state = AppState.instance()
    before = state.init_guess_mode
    state.init_guess_mode = "seed_propagation"
    yield
    state.init_guess_mode = before


@pytest.mark.parametrize("lang", ["en", "de", "fr", "es", "ja", "zh_TW"])
def test_no_parameter_label_is_clipped(lang, starting_points_mode):
    """The labels were fixed at 120 px (R5) and lost their tails in German,
    French and Spanish ("Niveau de raffineme"). The column now takes its
    longest label, and wraps past a cap. Measured with each label's own font,
    so the check holds whatever fonts the machine has.
    """
    from al_dic.i18n import LanguageManager

    manager = LanguageManager(app)
    assert manager.load(lang)
    try:
        panel = ParamPanel()
        labels = panel._row_labels
        texts = [label.text() for label in labels]
        assert "Starting Point Search" in texts or lang != "en"
        widths = {label.maximumWidth() for label in labels}
        assert len(widths) == 1, f"labels no longer share one column: {widths}"
        width = widths.pop()
        for label in labels:
            # Qt's own line breaking: an unbreakable run wider than the
            # column comes back wider than the rectangle it was given.
            flags = Qt.TextFlag.TextWordWrap if label.wordWrap() else 0
            needed = label.fontMetrics().boundingRect(
                QRect(0, 0, width, 10_000), flags, label.text()).width()
            assert needed <= width, f"{label.text()!r} is cut at {width} px"
    finally:
        manager.load("en")
