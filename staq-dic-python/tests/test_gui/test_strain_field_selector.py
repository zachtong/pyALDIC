"""Tests for StrainFieldSelector -- 9-button (2 disp + 7 strain) selector."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout

app = QApplication.instance() or QApplication([])

from staq_dic.gui.widgets.strain_field_selector import (
    DISP_FIELD_NAMES,
    FIELD_NAMES,
    STRAIN_FIELD_NAMES,
    StrainFieldSelector,
)


@pytest.fixture
def selector():
    return StrainFieldSelector()


def test_default_field_is_disp_u(selector):
    """Default selection is disp_u: users land on the displacement view
    that mirrors the displacement-only data they just computed."""
    assert selector.current_field() == "disp_u"


def test_field_names_canonical_order():
    """The selector exposes 9 fields: 2 displacement + 7 strain."""
    assert DISP_FIELD_NAMES == ("disp_u", "disp_v")
    assert STRAIN_FIELD_NAMES == (
        "strain_exx",
        "strain_eyy",
        "strain_exy",
        "strain_principal_max",
        "strain_principal_min",
        "strain_maxshear",
        "strain_von_mises",
    )
    assert FIELD_NAMES == DISP_FIELD_NAMES + STRAIN_FIELD_NAMES
    assert len(FIELD_NAMES) == 9


def test_set_current_field_to_disp_v(selector):
    selector.set_current_field("disp_v")
    assert selector.current_field() == "disp_v"


def test_set_current_field_to_strain(selector):
    selector.set_current_field("strain_eyy")
    assert selector.current_field() == "strain_eyy"
    selector.set_current_field("strain_von_mises")
    assert selector.current_field() == "strain_von_mises"


def test_unknown_field_is_rejected(selector):
    with pytest.raises(ValueError):
        selector.set_current_field("displacement_magnitude")


def test_exclusive_selection_across_groups(selector):
    """Selecting a strain field deactivates the disp button (and vice versa)."""
    btns = selector.findChildren(QPushButton)
    assert len(btns) == 9

    selector.set_current_field("disp_u")
    checked = [b for b in btns if b.isChecked()]
    assert len(checked) == 1

    selector.set_current_field("strain_exx")
    checked = [b for b in btns if b.isChecked()]
    assert len(checked) == 1

    selector.set_current_field("disp_v")
    checked = [b for b in btns if b.isChecked()]
    assert len(checked) == 1


def test_field_changed_signal_emits(selector):
    received: list[str] = []
    selector.field_changed.connect(received.append)
    selector.set_current_field("strain_principal_max")
    assert received == ["strain_principal_max"]


def test_signal_not_emitted_when_setting_same_field(selector):
    received: list[str] = []
    selector.field_changed.connect(received.append)
    selector.set_current_field("disp_u")  # already current default
    assert received == []


def test_top_level_layout_is_vertical(selector):
    """Outer layout is QVBoxLayout (DISPLACEMENT section above STRAIN)."""
    layout = selector.layout()
    assert isinstance(layout, QVBoxLayout)
