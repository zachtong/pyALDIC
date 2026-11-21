"""Tests for WorkflowTypePanel and the dynamic ROIHint.

WorkflowTypePanel owns Tracking Mode + Solver + Reference Update; it
lives above the Region of Interest section so users see the high-level
choice before they start drawing. ROIHint is a live label under the
Region of Interest header that tells the user which frames need a
region, based on the current workflow choice.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState
from al_dic.gui.widgets.roi_hint import ROIHint
from al_dic.gui.widgets.workflow_type_panel import WorkflowTypePanel


@pytest.fixture(autouse=True)
def reset_state():
    """Each test starts from a fresh AppState singleton."""
    AppState._instance = None
    yield
    AppState._instance = None


# ---------------------------------------------------------------------
# WorkflowTypePanel
# ---------------------------------------------------------------------

def test_workflow_panel_exposes_core_controls():
    panel = WorkflowTypePanel()
    assert hasattr(panel, "_tracking_mode")
    assert hasattr(panel, "_solver")
    assert hasattr(panel, "_ref_mode")
    assert hasattr(panel, "_interval_spin")
    assert hasattr(panel, "_custom_edit")


def test_switching_tracking_mode_updates_state():
    panel = WorkflowTypePanel()
    state = AppState.instance()

    panel._tracking_mode.setCurrentText("Accumulative")
    assert state.tracking_mode == "accumulative"
    panel._tracking_mode.setCurrentText("Incremental")
    assert state.tracking_mode == "incremental"


def test_incremental_sub_panel_visibility_follows_mode():
    panel = WorkflowTypePanel()
    # Qt's isVisible() requires the parent window to be shown; use
    # isHidden() to test explicit setVisible() calls on an unshown widget.
    panel._tracking_mode.setCurrentText("Accumulative")
    assert panel._inc_panel.isHidden() is True
    panel._tracking_mode.setCurrentText("Incremental")
    assert panel._inc_panel.isHidden() is False


def test_new_project_defaults_to_local_dic():
    """A fresh project starts on Local DIC; AL-DIC is an explicit choice."""
    state = AppState.instance()
    panel = WorkflowTypePanel()

    assert state.use_admm is False
    assert panel._solver.currentData() == "local"


def test_shape_function_defaults_to_first_order():
    state = AppState.instance()
    panel = WorkflowTypePanel()

    assert state.shape_function_order == "first"
    assert panel._shape_order.currentData() == "first"
    assert panel._shape_order.isEnabled() is True     # Local DIC is the default


def test_choosing_second_order_updates_state():
    state = AppState.instance()
    panel = WorkflowTypePanel()

    panel._shape_order.setCurrentIndex(panel._shape_order.findData("second"))
    assert state.shape_function_order == "second"
    panel._shape_order.setCurrentIndex(panel._shape_order.findData("first"))
    assert state.shape_function_order == "first"


def test_auto_shape_function_can_be_chosen():
    state = AppState.instance()
    panel = WorkflowTypePanel()

    index = panel._shape_order.findData("auto")
    assert index >= 0
    panel._shape_order.setCurrentIndex(index)
    assert state.shape_function_order == "auto"


def test_shape_function_is_available_for_local_dic_only():
    """Second order exists for Local DIC only: with AL-DIC the control is
    disabled. The user's choice is kept, so switching back restores it."""
    state = AppState.instance()
    panel = WorkflowTypePanel()
    panel._shape_order.setCurrentIndex(panel._shape_order.findData("second"))

    panel._solver.setCurrentText("AL-DIC")
    assert panel._shape_order.isEnabled() is False
    assert state.shape_function_order == "second"

    panel._solver.setCurrentText("Local DIC")
    assert panel._shape_order.isEnabled() is True
    assert panel._shape_order.currentData() == "second"


def test_panel_starts_from_the_saved_shape_function():
    state = AppState.instance()
    state.shape_function_order = "second"
    state.use_admm = True
    panel = WorkflowTypePanel()

    assert panel._shape_order.currentData() == "second"
    assert panel._shape_order.isEnabled() is False


def test_solver_switch_updates_state_use_admm():
    """ADMM iteration control was moved from WorkflowTypePanel to
    AdvancedTuningWidget (ADVANCED section). The panel now only
    flips state.use_admm — the downstream widget watches that state
    to enable/disable its ADMM spinbox.
    """
    from al_dic.gui.app_state import AppState
    from al_dic.gui.widgets.advanced_tuning_widget import AdvancedTuningWidget

    state = AppState.instance()
    panel = WorkflowTypePanel()
    adv = AdvancedTuningWidget()

    panel._solver.setCurrentText("Local DIC")
    assert state.use_admm is False
    # AdvancedTuningWidget syncs on params_changed; emit manually since
    # the panel's set_param path doesn't emit in this synthetic test.
    state.params_changed.emit()
    assert adv._admm_iter_spin.isEnabled() is False

    panel._solver.setCurrentText("AL-DIC")
    assert state.use_admm is True
    state.params_changed.emit()
    assert adv._admm_iter_spin.isEnabled() is True


# ---------------------------------------------------------------------
# ROIHint — content follows workflow choice
# ---------------------------------------------------------------------

def test_hint_without_images_prompts_to_load():
    hint = ROIHint()
    assert "load images" in hint.text().lower()


def test_hint_accumulative_says_frame_1_only():
    state = AppState.instance()
    state.image_files = ["a.tif", "b.tif", "c.tif"]
    state.set_param("tracking_mode", "accumulative")
    state.images_changed.emit()
    hint = ROIHint()
    txt = hint.text().lower()
    assert "accumulative" in txt
    assert "frame 1" in txt
    assert "only" in txt


def test_hint_incremental_every_frame_states_the_frame_1_fallback():
    """Nothing warps the frame-1 region forward (the hint once said so).

    A reference frame without its own Region of Interest reuses frame 1's
    unchanged after the Run-time confirmation, so the hint points users at
    per-frame regions (Batch Import) and says what happens without them.
    """
    state = AppState.instance()
    state.image_files = ["a.tif", "b.tif", "c.tif", "d.tif"]
    state.set_param("tracking_mode", "incremental")
    state.set_param("inc_ref_mode", "every_frame")
    state.images_changed.emit()
    hint = ROIHint()
    txt = hint.text().lower()
    assert "warp" not in txt
    assert "batch import" in txt
    assert "reuse frame 1's region" in txt
    assert "confirm" in txt


def test_hint_every_n_lists_ref_frame_indices():
    state = AppState.instance()
    state.image_files = [f"f{i}.tif" for i in range(10)]   # 10 frames
    state.set_param("tracking_mode", "incremental")
    state.set_param("inc_ref_mode", "every_n")
    state.inc_ref_interval = 3
    state.images_changed.emit()
    state.params_changed.emit()
    hint = ROIHint()
    txt = hint.text()
    # Ref frames at 0, 3, 6, 9 -> 1-based 1, 4, 7, 10
    for n in ("1", "4", "7", "10"):
        assert n in txt


def test_hint_custom_frames_list():
    state = AppState.instance()
    state.image_files = [f"f{i}.tif" for i in range(20)]
    state.set_param("tracking_mode", "incremental")
    state.set_param("inc_ref_mode", "custom")
    state.inc_custom_refs = [4, 9, 14]   # 0-based -> 1-based 5, 10, 15
    state.images_changed.emit()
    state.params_changed.emit()
    hint = ROIHint()
    txt = hint.text()
    # Frame 1 is always included, then user-selected
    assert "1" in txt
    for n in ("5", "10", "15"):
        assert n in txt


def test_hint_updates_live_when_mode_changes():
    state = AppState.instance()
    state.image_files = ["a.tif", "b.tif", "c.tif"]
    state.images_changed.emit()

    hint = ROIHint()
    state.set_param("tracking_mode", "accumulative")
    state.params_changed.emit()
    acc_txt = hint.text().lower()
    assert "accumulative" in acc_txt

    state.set_param("tracking_mode", "incremental")
    state.set_param("inc_ref_mode", "every_frame")
    state.params_changed.emit()
    inc_txt = hint.text().lower()
    assert "incremental" in inc_txt
    assert acc_txt != inc_txt
