"""Tests for StrainParamPanel -- strain-only parameter editor."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.controllers.strain_controller import ALLOWED_OVERRIDES
from al_dic.gui.widgets.strain_param_panel import StrainParamPanel, _SMOOTH_PRESETS


@pytest.fixture
def panel():
    return StrainParamPanel()


def test_default_override(panel):
    """Default: plane fitting (method 2), vsg=41 -> rad=20, no post-grad smoothing,
    strain_type 0 (infinitesimal)."""
    o = panel.get_override()
    assert o == {
        "method_to_compute_strain": 2,
        "strain_plane_fit_rad": 20.0,
        "strain_smoothness": 0.0,
        "strain_type": 0,
    }


def test_override_keys_match_whitelist(panel):
    o = panel.get_override()
    assert set(o.keys()) == ALLOWED_OVERRIDES


def test_only_two_methods_in_combo(panel):
    """Only plane fitting (2) and FEM nodal (3) are available."""
    assert panel._method_codes == (2, 3)
    assert panel._method_combo.count() == 2


def test_default_method_is_2(panel):
    assert panel.get_override()["method_to_compute_strain"] == 2


def test_changing_method_to_fem(panel):
    """Selecting FEM nodal (index 1 = method 3) updates override."""
    panel._method_combo.setCurrentIndex(1)   # method 3
    assert panel.get_override()["method_to_compute_strain"] == 3


def test_vsg_spin_enabled_only_for_plane_fitting(panel):
    """VSG size is only enabled when plane fitting (method 2) is selected."""
    panel._method_combo.setCurrentIndex(0)   # plane fitting (method 2)
    assert panel._vsg_spin.isEnabled() is True

    panel._method_combo.setCurrentIndex(1)   # FEM nodal (method 3)
    assert panel._vsg_spin.isEnabled() is False


def test_changing_vsg_updates_override(panel):
    """VSG -> rad conversion: rad = (VSG - 1) / 2."""
    panel._vsg_spin.setValue(21)   # rad = (21-1)/2 = 10
    assert panel.get_override()["strain_plane_fit_rad"] == 10.0

    panel._vsg_spin.setValue(41)   # rad = (41-1)/2 = 20
    assert panel.get_override()["strain_plane_fit_rad"] == 20.0


def test_changing_strain_type_updates_override(panel):
    # 0 = infinitesimal, 1 = Eulerian, 2 = Green-Lagrangian
    panel._type_combo.setCurrentIndex(2)
    assert panel.get_override()["strain_type"] == 2


def test_smoothing_default_is_off(panel):
    """Default selection is 'Off' (index 0), yielding zero smoothness."""
    assert panel._smooth_combo.currentIndex() == 0
    assert panel._smooth_combo.currentText() == "Off"
    assert panel.get_override()["strain_smoothness"] == 0.0


def test_non_off_preset_gives_nonzero_smoothness(panel):
    """Any preset other than 'Off' yields a positive smoothness value."""
    for i in range(1, panel._smooth_combo.count()):
        panel._smooth_combo.setCurrentIndex(i)
        assert panel.get_override()["strain_smoothness"] > 0.0


def test_all_presets_are_distinct(panel):
    """Each preset maps to a distinct smoothness value."""
    seen = set()
    for i in range(panel._smooth_combo.count()):
        panel._smooth_combo.setCurrentIndex(i)
        seen.add(panel.get_override()["strain_smoothness"])
    assert len(seen) == len(_SMOOTH_PRESETS), "All presets should be distinct"


def test_smooth_combo_always_enabled(panel):
    """Smoothing combo has no gating checkbox; always interactive."""
    assert panel._smooth_combo.isEnabled() is True


def test_initially_clean(panel):
    assert panel.is_dirty() is False


def test_dirty_after_param_change(panel):
    panel._vsg_spin.setValue(51)
    assert panel.is_dirty() is True


def test_mark_clean_resets_dirty(panel):
    panel._vsg_spin.setValue(51)
    assert panel.is_dirty() is True
    panel.mark_clean()
    assert panel.is_dirty() is False


def test_params_dirty_signal(panel):
    received: list[bool] = []
    panel.params_dirty.connect(lambda: received.append(True))
    panel._vsg_spin.setValue(61)
    assert len(received) >= 1
