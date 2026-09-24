"""Tests for StrainVizPanel -- colormap / vmin / vmax / opacity / deformed controls."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.widgets.strain_viz_panel import StrainVizPanel


@pytest.fixture
def panel():
    return StrainVizPanel()


def test_default_state(panel):
    """jet, auto range, opacity 0.70, deformed geometry, background shown."""
    s = panel.get_state()
    assert s["colormap"] == "jet"
    assert s["use_percentile"] is True
    assert s["alpha"] == pytest.approx(0.70)
    assert s["show_deformed"] is True
    assert s["show_background"] is True
    assert "vmin" in s and "vmax" in s


def test_state_keys(panel):
    expected = {"colormap", "vmin", "vmax", "alpha", "use_percentile",
                "show_deformed", "show_background", "hidden_bg_color",
                "fill_trimmed_edges"}
    assert set(panel.get_state().keys()) == expected


def test_fill_trimmed_edges_default_off(panel):
    """The 'Fill trimmed edges' toggle is unchecked by default (trim visible)."""
    assert panel.get_state()["fill_trimmed_edges"] is False


def test_fill_trimmed_edges_toggle_updates_state_and_emits(panel):
    """Checking the fill toggle flips the state and re-renders (viz_changed)."""
    received: list[bool] = []
    panel.viz_changed.connect(lambda: received.append(True))
    panel._fill_edges_check.setChecked(True)
    assert panel.get_state()["fill_trimmed_edges"] is True
    assert len(received) >= 1
    panel._fill_edges_check.setChecked(False)
    assert panel.get_state()["fill_trimmed_edges"] is False


def test_change_colormap_emits_signal(panel):
    received: list[bool] = []
    panel.viz_changed.connect(lambda: received.append(True))
    idx = panel._cmap_combo.findText("seismic")
    assert idx >= 0
    panel._cmap_combo.setCurrentIndex(idx)
    assert panel.get_state()["colormap"] == "seismic"
    assert len(received) >= 1


def test_auto_toggle_disables_vmin_vmax_spins(panel):
    """When auto is ON, manual vmin/vmax spinboxes are disabled."""
    panel._auto_check.setChecked(True)
    assert panel._vmin_spin.isEnabled() is False
    assert panel._vmax_spin.isEnabled() is False
    panel._auto_check.setChecked(False)
    assert panel._vmin_spin.isEnabled() is True
    assert panel._vmax_spin.isEnabled() is True


def test_auto_disabled_signal_fires_on_manual_switch(panel):
    """auto_disabled fires exactly once when auto checkbox goes True -> False."""
    panel._auto_check.setChecked(True)
    fired: list[bool] = []
    panel.auto_disabled.connect(lambda: fired.append(True))
    panel._auto_check.setChecked(False)
    assert len(fired) == 1


def test_auto_disabled_signal_not_fired_when_enabling_auto(panel):
    panel._auto_check.setChecked(False)
    fired: list[bool] = []
    panel.auto_disabled.connect(lambda: fired.append(True))
    panel._auto_check.setChecked(True)
    assert fired == []


def test_set_range_populates_spinboxes_without_double_emit(panel):
    """set_range() sets vmin/vmax without extra auto_disabled signals."""
    panel._auto_check.setChecked(False)
    fired: list[bool] = []
    panel.auto_disabled.connect(lambda: fired.append(True))
    panel.set_range(-0.05, 0.05)
    assert panel.get_state()["vmin"] == pytest.approx(-0.05)
    assert panel.get_state()["vmax"] == pytest.approx(0.05)
    assert fired == []


def test_change_vmin_emits_signal(panel):
    panel._auto_check.setChecked(False)
    received: list[bool] = []
    panel.viz_changed.connect(lambda: received.append(True))
    panel._vmin_spin.setValue(-0.05)
    assert panel.get_state()["vmin"] == pytest.approx(-0.05)
    assert len(received) >= 1


def test_change_vmax_updates_state(panel):
    panel._auto_check.setChecked(False)
    panel._vmax_spin.setValue(0.05)
    assert panel.get_state()["vmax"] == pytest.approx(0.05)


def test_opacity_slider_returns_fraction(panel):
    """Opacity slider stores 0-100 internally but get_state() returns [0, 1]."""
    panel._opacity_slider.setValue(40)
    assert panel.get_state()["alpha"] == pytest.approx(0.40)
    panel._opacity_slider.setValue(100)
    assert panel.get_state()["alpha"] == pytest.approx(1.00)


def test_opacity_renamed_label(panel):
    """The opacity control is labeled 'Opacity', not 'Alpha'."""
    from PySide6.QtWidgets import QLabel, QFormLayout
    layout = panel.layout()
    assert isinstance(layout, QFormLayout)
    labels = []
    for i in range(layout.rowCount()):
        item = layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
        if item and item.widget() and isinstance(item.widget(), QLabel):
            labels.append(item.widget().text())
    assert any("Opacity" in t for t in labels), f"Labels found: {labels}"
    assert not any(t == "Alpha" for t in labels), f"'Alpha' should be renamed: {labels}"


def test_geometry_combo_selects_where_the_field_is_plotted(panel):
    """Deformed is first so it stays the default; reference is second."""
    panel._geometry_combo.setCurrentIndex(0)
    assert panel.get_state()["show_deformed"] is True
    panel._geometry_combo.setCurrentIndex(1)
    assert panel.get_state()["show_deformed"] is False


def test_background_toggle_is_independent_of_geometry(panel):
    """The whole point of splitting the old checkbox: all four combinations.

    Hiding the image used to be impossible, and asking for the reference
    image used to force reference geometry along with it.
    """
    panel._geometry_combo.setCurrentIndex(0)      # deformed
    panel._background_check.setChecked(False)
    s = panel.get_state()
    assert s["show_deformed"] is True and s["show_background"] is False

    panel._geometry_combo.setCurrentIndex(1)      # reference
    panel._background_check.setChecked(True)
    s = panel.get_state()
    assert s["show_deformed"] is False and s["show_background"] is True


def test_colormaps_list(panel):
    """jet must be first (default); diverging maps also present."""
    items = [panel._cmap_combo.itemText(i)
             for i in range(panel._cmap_combo.count())]
    assert items[0] == "jet"
    for name in ("RdBu_r", "seismic", "coolwarm"):
        assert name in items


def test_the_fill_is_only_editable_when_it_applies(panel):
    """Disabled, not hidden: the row must not jump as the box is toggled."""
    panel._background_check.setChecked(True)
    assert panel._hidden_bg_combo.isEnabled() is False
    panel._background_check.setChecked(False)
    assert panel._hidden_bg_combo.isEnabled() is True


def test_the_fill_reaches_get_state(panel):
    """It was readable by every render path and writable by none, so the
    display was permanently white whatever the setting said."""
    assert panel.get_state()["hidden_bg_color"] == "white"
    panel._hidden_bg_combo.setCurrentIndex(
        panel._hidden_bg_combo.findData("transparent"))
    assert panel.get_state()["hidden_bg_color"] == "transparent"
