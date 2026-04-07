"""Tests for StrainFieldSelector -- 7-button 3-row exclusive selector."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QGridLayout, QPushButton

app = QApplication.instance() or QApplication([])

from staq_dic.gui.widgets.strain_field_selector import (
    STRAIN_FIELD_NAMES,
    StrainFieldSelector,
)


@pytest.fixture
def selector():
    return StrainFieldSelector()


def test_default_field_is_exx(selector):
    assert selector.current_field() == "strain_exx"


def test_field_names_list_is_canonical():
    """The selector's known field set must match StrainResult attributes."""
    assert STRAIN_FIELD_NAMES == (
        "strain_exx",
        "strain_eyy",
        "strain_exy",
        "strain_principal_max",
        "strain_principal_min",
        "strain_maxshear",
        "strain_von_mises",
    )


def test_set_current_field_changes_active_button(selector):
    selector.set_current_field("strain_eyy")
    assert selector.current_field() == "strain_eyy"
    selector.set_current_field("strain_von_mises")
    assert selector.current_field() == "strain_von_mises"


def test_unknown_field_is_rejected(selector):
    with pytest.raises(ValueError):
        selector.set_current_field("disp_u")


def test_exclusive_selection(selector):
    """Only one button is checked at a time."""
    selector.set_current_field("strain_exy")
    btns = selector.findChildren(QPushButton)
    checked = [b for b in btns if b.isChecked()]
    assert len(checked) == 1


def test_field_changed_signal_emits(selector):
    received: list[str] = []
    selector.field_changed.connect(received.append)
    selector.set_current_field("strain_principal_max")
    assert received == ["strain_principal_max"]


def test_three_row_grid_layout(selector):
    """The widget uses a QGridLayout with at least 3 rows."""
    layout = selector.layout()
    assert isinstance(layout, QGridLayout)
    rows = layout.rowCount()
    assert rows >= 3


def test_signal_not_emitted_when_setting_same_field(selector):
    received: list[str] = []
    selector.field_changed.connect(received.append)
    selector.set_current_field("strain_exx")  # already current default
    assert received == []
