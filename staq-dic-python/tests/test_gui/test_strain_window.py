"""Tests for StrainWindow -- the post-processing strain window assembly.

Verifies the maximum-decoupling design contracts:
* StrainWindow has its own _strain_current_frame, never touches state.current_frame
* Default field is disp_u (matches selector default)
* Compute Strain populates state.results.result_strain
* Param change marks the window stale; Recompute clears the stale label
* Field change re-renders without recomputing
* StrainWindow never writes to state.colormap / state.color_min / state.display_field
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from staq_dic.core.data_structures import StrainResult
from staq_dic.gui.app_state import AppState
from staq_dic.gui.strain_window import StrainWindow

from ._helpers import make_synthetic_pipeline_result


@pytest.fixture
def state_with_results():
    """Fresh AppState pre-populated with a synthetic pure-shear result."""
    state = AppState()
    result, mask = make_synthetic_pipeline_result(
        n_frames=3, shear=0.01, img_shape=(128, 128), step=16,
    )
    state.results = result
    state.per_frame_rois[0] = mask.astype(bool)
    state.current_frame = 0
    state.colormap = "jet"
    state.color_min = 0.0
    state.color_max = 1.0
    state.display_field = "disp_u"
    return state


@pytest.fixture
def window(state_with_results):
    return StrainWindow(state_with_results)


# ----------------------------------------------------------------------
# Independent timeline
# ----------------------------------------------------------------------

def test_window_has_independent_frame_state(window, state_with_results):
    """Setting the strain frame must NOT mutate AppState.current_frame."""
    assert state_with_results.current_frame == 0
    window.set_strain_frame(1)
    assert window.strain_current_frame() == 1
    assert state_with_results.current_frame == 0  # untouched


def test_default_strain_frame_is_zero(window):
    assert window.strain_current_frame() == 0


# ----------------------------------------------------------------------
# Field selector wiring
# ----------------------------------------------------------------------

def test_default_field_is_disp_u(window):
    """Selector defaults to disp_u (the displacement view that mirrors
    what the user just computed)."""
    assert window.current_field() == "disp_u"


# ----------------------------------------------------------------------
# Compute / recompute behaviour
# ----------------------------------------------------------------------

def test_compute_button_populates_results(window, state_with_results):
    window.trigger_compute()
    result_strain = state_with_results.results.result_strain
    assert len(result_strain) == len(state_with_results.results.result_disp)
    assert isinstance(result_strain[0], StrainResult)


def test_compute_clears_stale_label(window):
    """Touching a parameter sets the stale hint, and Compute clears it."""
    window.param_panel()._rad_spin.setValue(15.0)  # mark dirty
    assert window.is_stale() is True
    window.trigger_compute()
    assert window.is_stale() is False


def test_param_change_marks_stale(window):
    assert window.is_stale() is False
    window.param_panel()._rad_spin.setValue(15.0)
    assert window.is_stale() is True


# ----------------------------------------------------------------------
# Decoupling guarantee: StrainWindow does NOT touch shared display state
# ----------------------------------------------------------------------

def test_no_writes_to_main_state_colormap(window, state_with_results):
    """Changing strain viz must NEVER mutate state.colormap / color_min /
    color_max / display_field. Those belong to the displacement view."""
    state_with_results.colormap = "jet"
    state_with_results.display_field = "disp_u"
    state_with_results.color_min = 0.123
    state_with_results.color_max = 0.456

    window.trigger_compute()
    window.set_current_field("strain_exx")
    # Force a re-render with a different colormap
    window.viz_panel()._cmap_combo.setCurrentText("seismic")

    assert state_with_results.colormap == "jet"
    assert state_with_results.display_field == "disp_u"
    assert state_with_results.color_min == 0.123
    assert state_with_results.color_max == 0.456


def test_field_change_does_not_recompute(window, state_with_results):
    """Changing the field after compute should re-render only, not call
    StrainController again. We assert by counting results_changed emissions."""
    received: list[bool] = []
    state_with_results.results_changed.connect(lambda: received.append(True))
    window.trigger_compute()
    n_after_compute = len(received)
    window.set_current_field("strain_exx")
    assert len(received) == n_after_compute  # no extra emit


# ----------------------------------------------------------------------
# New fields: displacement before strain, derived fields
# ----------------------------------------------------------------------

def test_disp_u_available_before_compute(window, state_with_results):
    """disp_u should render from result_disp without running Compute Strain.
    Verifies fix for item 2: displacement fields bypass result_strain."""
    result = state_with_results.results
    assert not result.result_strain   # strain NOT computed yet
    # _get_field_values should return the u-component from result_disp
    vals = window._get_field_values("disp_u", 0, result)
    assert vals is not None
    assert len(vals) == result.result_fe_mesh_each_frame[0].coordinates_fem.shape[0]


def test_velocity_available_before_compute(window, state_with_results):
    """velocity field is derived from result_disp increments, no strain needed."""
    result = state_with_results.results
    assert not result.result_strain
    vals = window._get_field_values("velocity", 0, result)
    assert vals is not None
    assert (vals >= 0).all()  # velocity magnitude is non-negative


def test_disp_magnitude_available_before_compute(window, state_with_results):
    result = state_with_results.results
    vals = window._get_field_values("disp_magnitude", 0, result)
    assert vals is not None
    assert (vals >= 0).all()


def test_strain_rotation_requires_compute(window, state_with_results):
    """strain_rotation returns None until Compute Strain is run."""
    result = state_with_results.results
    assert window._get_field_values("strain_rotation", 0, result) is None
    window.trigger_compute()
    result_after = state_with_results.results
    vals = window._get_field_values("strain_rotation", 0, result_after)
    assert vals is not None


def test_auto_range_disabled_populates_spinboxes(window, state_with_results):
    """Disabling auto range triggers set_range() with field's data range.
    For a uniform shear field all nodes share one value so vmin == vmax."""
    window.trigger_compute()
    window.set_current_field("strain_exy")
    panel = window.viz_panel()
    # Start with known far-off defaults
    panel.set_range(-999.0, 999.0)
    # Disabling auto should repopulate with the actual field values
    panel._auto_check.setChecked(False)
    s = panel.get_state()
    # Values must be finite and within [-1, 1] for a small-shear field
    assert abs(s["vmin"]) < 1.0
    assert abs(s["vmax"]) < 1.0
    # vmin <= vmax (equality is valid for a uniform-value field)
    assert s["vmin"] <= s["vmax"]
