"""Tests for StrainFieldSelector -- 13-button selector (4 disp + 9 strain)."""

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
    """Default selection is disp_u: displacement is visible without Compute Strain."""
    assert selector.current_field() == "disp_u"


def test_field_names_canonical_order():
    """4 displacement fields + 9 strain fields = 13 total."""
    assert DISP_FIELD_NAMES == ("disp_u", "disp_v", "disp_magnitude", "velocity")
    assert STRAIN_FIELD_NAMES == (
        "strain_exx",
        "strain_eyy",
        "strain_exy",
        "strain_principal_max",
        "strain_principal_min",
        "strain_maxshear",
        "strain_von_mises",
        "strain_rotation",
    )
    assert FIELD_NAMES == DISP_FIELD_NAMES + STRAIN_FIELD_NAMES
    assert len(FIELD_NAMES) == 12


def test_set_current_field_to_disp_v(selector):
    selector.set_current_field("disp_v")
    assert selector.current_field() == "disp_v"


def test_set_current_field_to_magnitude(selector):
    selector.set_current_field("disp_magnitude")
    assert selector.current_field() == "disp_magnitude"


def test_set_current_field_to_velocity(selector):
    selector.set_current_field("velocity")
    assert selector.current_field() == "velocity"


def test_set_current_field_to_strain(selector):
    selector.set_current_field("strain_eyy")
    assert selector.current_field() == "strain_eyy"
    selector.set_current_field("strain_von_mises")
    assert selector.current_field() == "strain_von_mises"


def test_set_current_field_to_rotation(selector):
    selector.set_current_field("strain_rotation")
    assert selector.current_field() == "strain_rotation"


def test_unknown_field_is_rejected(selector):
    with pytest.raises(ValueError):
        selector.set_current_field("totally_unknown_field")


def test_exclusive_selection_across_groups(selector):
    """Selecting any field deactivates all others (single button group)."""
    btns = selector.findChildren(QPushButton)
    assert len(btns) == 12

    for field in ("disp_u", "strain_exx", "velocity", "strain_rotation"):
        selector.set_current_field(field)
        checked = [b for b in btns if b.isChecked()]
        assert len(checked) == 1, f"Expected 1 checked after {field}, got {len(checked)}"


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
