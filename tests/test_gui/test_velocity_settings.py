"""Tests for VelocitySettingsWidget — physical unit conversion."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.widgets.velocity_settings import VelocitySettingsWidget


@pytest.fixture
def widget():
    return VelocitySettingsWidget()


def test_default_state_no_physical(widget):
    """Default: physical units disabled."""
    cfg = widget.get_config()
    assert cfg["use_physical"] is False
    assert cfg["result_unit"] == "px/frame"


def test_get_config_keys(widget):
    expected = {"use_physical", "pixel_size", "unit", "fps", "result_unit"}
    assert set(widget.get_config().keys()) == expected


def test_set_visible_for_velocity_field(widget):
    widget.set_visible_for_field("velocity")
    assert widget.isVisible()


def test_set_visible_for_non_velocity_field(widget):
    widget.set_visible_for_field("strain_exx")
    assert not widget.isVisible()


def test_apply_conversion_disabled(widget):
    """When disabled, return input unchanged."""
    vel = np.array([1.0, 2.0, 3.0])
    result, unit = widget.apply_conversion(vel)
    np.testing.assert_array_equal(result, vel)
    assert unit == "px/frame"


def test_apply_conversion_enabled(widget):
    """When enabled, vel_physical = vel * pixel_size * fps."""
    widget._use_phy.setChecked(True)
    widget._px_size_spin.setValue(0.5)
    widget._fps_spin.setValue(10.0)

    vel = np.array([2.0, 4.0])
    result, unit = widget.apply_conversion(vel)
    expected = vel * 0.5 * 10.0
    np.testing.assert_allclose(result, expected)
    assert "/s" in unit


def test_toggle_enables_spin_boxes(widget):
    """Checking 'use physical' should enable the spin boxes."""
    assert not widget._px_size_spin.isEnabled()
    assert not widget._fps_spin.isEnabled()

    widget._use_phy.setChecked(True)
    assert widget._px_size_spin.isEnabled()
    assert widget._fps_spin.isEnabled()

    widget._use_phy.setChecked(False)
    assert not widget._px_size_spin.isEnabled()


def test_settings_changed_signal(widget):
    """Signal should fire on toggle."""
    fired = []
    widget.settings_changed.connect(lambda: fired.append(True))
    widget._use_phy.setChecked(True)
    assert len(fired) >= 1
