"""Tests for StrainParamPanel -- strain-only parameter editor."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from staq_dic.gui.controllers.strain_controller import ALLOWED_OVERRIDES
from staq_dic.gui.widgets.strain_param_panel import StrainParamPanel


@pytest.fixture
def panel():
    return StrainParamPanel()


def test_default_override(panel):
    o = panel.get_override()
    assert o == {
        "method_to_compute_strain": 2,
        "strain_plane_fit_rad": 20.0,
        "strain_smoothness": 1e-5,
        "strain_type": 0,
    }


def test_override_keys_match_whitelist(panel):
    o = panel.get_override()
    assert set(o.keys()) == ALLOWED_OVERRIDES


def test_changing_rad_updates_override(panel):
    panel._rad_spin.setValue(10.0)
    assert panel.get_override()["strain_plane_fit_rad"] == 10.0


def test_changing_method_updates_override(panel):
    # Method combobox: index 0 -> 2 (plane fit), index 1 -> 3 (FEM)
    panel._method_combo.setCurrentIndex(1)
    assert panel.get_override()["method_to_compute_strain"] == 3


def test_changing_strain_type_updates_override(panel):
    # 0 = infinitesimal, 1 = Eulerian, 2 = Green-Lagrangian
    panel._type_combo.setCurrentIndex(2)
    assert panel.get_override()["strain_type"] == 2


def test_changing_smoothness_updates_override(panel):
    panel._smooth_spin.setValue(1e-3)
    assert panel.get_override()["strain_smoothness"] == pytest.approx(1e-3)


def test_initially_clean(panel):
    assert panel.is_dirty() is False


def test_dirty_after_param_change(panel):
    panel._rad_spin.setValue(10.0)
    assert panel.is_dirty() is True


def test_mark_clean_resets_dirty(panel):
    panel._rad_spin.setValue(10.0)
    assert panel.is_dirty() is True
    panel.mark_clean()
    assert panel.is_dirty() is False


def test_params_dirty_signal(panel):
    received: list[bool] = []
    panel.params_dirty.connect(lambda: received.append(True))
    panel._rad_spin.setValue(15.0)
    assert len(received) >= 1


def test_no_dirty_signal_after_mark_clean(panel):
    panel._rad_spin.setValue(15.0)
    panel.mark_clean()
    received: list[bool] = []
    panel.params_dirty.connect(lambda: received.append(True))
    panel._rad_spin.setValue(15.0)  # same value -> no signal
    assert received == []
