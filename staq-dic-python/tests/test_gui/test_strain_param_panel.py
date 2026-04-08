"""Tests for StrainParamPanel -- strain-only parameter editor."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from staq_dic.gui.controllers.strain_controller import ALLOWED_OVERRIDES
from staq_dic.gui.widgets.strain_param_panel import StrainParamPanel, _SMOOTH_PRESETS


@pytest.fixture
def panel():
    return StrainParamPanel()


def test_default_override(panel):
    """Default: plane fitting (method 2), vsg=41 -> rad=20, no pre-smooth,
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


def test_presmooth_unchecked_gives_zero_smoothness(panel):
    assert panel._presmooth_check.isChecked() is False
    assert panel.get_override()["strain_smoothness"] == 0.0


def test_presmooth_checked_gives_nonzero_smoothness(panel):
    panel._presmooth_check.setChecked(True)
    s = panel.get_override()["strain_smoothness"]
    assert s > 0.0


def test_presmooth_strength_combo_updates_smoothness(panel):
    """Each preset in the dropdown maps to a distinct smoothness value."""
    panel._presmooth_check.setChecked(True)
    seen = set()
    for i in range(panel._smooth_combo.count()):
        panel._smooth_combo.setCurrentIndex(i)
        val = panel.get_override()["strain_smoothness"]
        assert val > 0.0
        seen.add(val)
    assert len(seen) == len(_SMOOTH_PRESETS), "All presets should be distinct"


def test_presmooth_combo_hidden_when_unchecked(panel):
    assert panel._smooth_combo.isEnabled() is False
    panel._presmooth_check.setChecked(True)
    assert panel._smooth_combo.isEnabled() is True
    panel._presmooth_check.setChecked(False)
    assert panel._smooth_combo.isEnabled() is False


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
