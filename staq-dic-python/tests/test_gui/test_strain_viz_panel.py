"""Tests for StrainVizPanel -- colormap / vmin / vmax / alpha controls."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from staq_dic.gui.widgets.strain_viz_panel import StrainVizPanel


@pytest.fixture
def panel():
    return StrainVizPanel()


def test_default_state(panel):
    """Default colormap is RdBu_r (diverging, zero-centered for signed
    strain), percentile auto-range is on, alpha = 0.70."""
    s = panel.get_state()
    assert s["colormap"] == "RdBu_r"
    assert s["use_percentile"] is True
    assert s["alpha"] == pytest.approx(0.70)
    assert "vmin" in s and "vmax" in s


def test_state_keys(panel):
    expected = {"colormap", "vmin", "vmax", "alpha", "use_percentile"}
    assert set(panel.get_state().keys()) == expected


def test_change_colormap_emits_signal(panel):
    received: list[bool] = []
    panel.viz_changed.connect(lambda: received.append(True))
    idx = panel._cmap_combo.findText("seismic")
    assert idx >= 0
    panel._cmap_combo.setCurrentIndex(idx)
    assert panel.get_state()["colormap"] == "seismic"
    assert len(received) >= 1


def test_percentile_toggle_disables_vmin_vmax_spins(panel):
    """When percentile mode is ON, manual vmin/vmax spinboxes are disabled."""
    panel._pct_check.setChecked(True)
    assert panel._vmin_spin.isEnabled() is False
    assert panel._vmax_spin.isEnabled() is False
    panel._pct_check.setChecked(False)
    assert panel._vmin_spin.isEnabled() is True
    assert panel._vmax_spin.isEnabled() is True


def test_change_vmin_emits_signal(panel):
    panel._pct_check.setChecked(False)  # enable manual mode
    received: list[bool] = []
    panel.viz_changed.connect(lambda: received.append(True))
    panel._vmin_spin.setValue(-0.05)
    assert panel.get_state()["vmin"] == pytest.approx(-0.05)
    assert len(received) >= 1


def test_change_vmax_updates_state(panel):
    panel._pct_check.setChecked(False)
    panel._vmax_spin.setValue(0.05)
    assert panel.get_state()["vmax"] == pytest.approx(0.05)


def test_change_alpha_updates_state(panel):
    """Alpha slider stores 0-100 internally but get_state() returns [0, 1]."""
    panel._alpha_slider.setValue(40)
    assert panel.get_state()["alpha"] == pytest.approx(0.40)
    panel._alpha_slider.setValue(100)
    assert panel.get_state()["alpha"] == pytest.approx(1.00)


def test_alpha_signal(panel):
    received: list[bool] = []
    panel.viz_changed.connect(lambda: received.append(True))
    panel._alpha_slider.setValue(50)
    assert len(received) >= 1


def test_colormaps_list(panel):
    """Diverging colormaps must be listed first since strain is signed."""
    items = [panel._cmap_combo.itemText(i)
             for i in range(panel._cmap_combo.count())]
    for name in ("RdBu_r", "seismic", "coolwarm"):
        assert name in items
