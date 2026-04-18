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


def test_hint_incremental_every_frame_mentions_warping():
    state = AppState.instance()
    state.image_files = ["a.tif", "b.tif", "c.tif", "d.tif"]
    state.set_param("tracking_mode", "incremental")
    state.set_param("inc_ref_mode", "every_frame")
    state.images_changed.emit()
    hint = ROIHint()
    txt = hint.text().lower()
    assert "warped" in txt or "warp" in txt


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
