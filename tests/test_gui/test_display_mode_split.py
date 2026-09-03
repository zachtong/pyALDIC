"""Geometry and background are chosen separately, on every surface.

One checkbox used to answer two questions: which image sits behind the field,
and where the field's nodes are drawn. Splitting them is only worth anything
if all four combinations actually survive the trip from widget to exporter --
which is what these check.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult,
)
from al_dic.gui.app_state import AppState
from al_dic.gui.dialogs.export_dialog import (
    ExportDialog, VizExportHint, _bg_mode_for,
)

app = QApplication.instance() or QApplication([])


def _result(img=128):
    xs, ys = np.meshgrid(np.linspace(8, img - 8, 6), np.linspace(8, img - 8, 6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    n = coords.shape[0]
    mesh = DICMesh(coordinates_fem=coords,
                   elements_fem=np.zeros((0, 8), np.int64))
    u = coords[:, 0] / img
    fr = FrameResult(U=np.repeat(u, 2), U_accum=np.repeat(u, 2))
    sr = StrainResult(
        disp_u=u, disp_v=np.zeros(n), strain_exx=np.full(n, 0.01),
        strain_eyy=np.zeros(n), strain_exy=np.zeros(n),
        strain_principal_max=np.full(n, 0.01), strain_principal_min=np.zeros(n),
        strain_maxshear=np.full(n, 0.005), strain_von_mises=np.full(n, 0.01),
        strain_rotation=np.zeros(n),
    )
    return PipelineResult(
        dic_para=dicpara_default(img_size=(img, img)), dic_mesh=mesh,
        result_disp=[fr, fr], result_def_grad=[fr, fr], result_strain=[sr, sr],
        result_fe_mesh_each_frame=[mesh, mesh],
        frame_schedule=FrameSchedule.from_mode("accumulative", 3))


@pytest.fixture
def dialog():
    return ExportDialog(_result(), None, VizExportHint(), image_files=[])


# --- the rule itself ------------------------------------------------------

@pytest.mark.parametrize("deformed,background,expected", [
    (True,  True,  "current_frame"),
    (False, True,  "ref_frame"),
    (True,  False, "none"),
    (False, False, "none"),
])
def test_the_background_rule(deformed, background, expected):
    """Geometry picks WHICH frame; the checkbox decides whether to show one.

    The two "none" rows are the point of the split: with no image behind it,
    the field can still be drawn on either geometry.
    """
    assert _bg_mode_for(deformed, background) == expected


# --- state ----------------------------------------------------------------

def test_state_defaults_to_showing_the_background():
    """A session saved before this existed must render exactly as it did."""
    state = AppState()
    assert state.show_background is True
    assert state.show_deformed is True
    assert state.hidden_bg_color == "white"


def test_the_hint_carries_both_settings():
    h = VizExportHint()
    assert h.show_background is True
    assert h.hidden_bg_color == "white"


# --- export dialog wiring -------------------------------------------------

def test_hiding_the_background_reaches_both_exporters(dialog):
    dialog._img_bg_check.setChecked(False)
    dialog._anim_bg_check.setChecked(False)
    cfg = dialog.get_config()
    assert cfg.bg_mode == "none"
    assert cfg.anim_bg_mode == "none"


def test_geometry_survives_hiding_the_background(dialog):
    """The combination the old radio pair could not express."""
    dialog._img_geom_combo.setCurrentIndex(0)      # deformed
    dialog._img_bg_check.setChecked(False)
    cfg = dialog.get_config()
    assert cfg.show_deformed is True, "geometry was lost with the image"
    assert cfg.bg_mode == "none"

    dialog._img_geom_combo.setCurrentIndex(1)      # reference
    dialog._img_bg_check.setChecked(False)
    cfg = dialog.get_config()
    assert cfg.show_deformed is False
    assert cfg.bg_mode == "none"


def test_images_and_animation_are_set_independently(dialog):
    """Two tabs, two exports; one must not silently follow the other."""
    dialog._img_bg_check.setChecked(False)
    dialog._anim_bg_check.setChecked(True)
    dialog._img_geom_combo.setCurrentIndex(1)
    dialog._anim_geom_combo.setCurrentIndex(0)

    cfg = dialog.get_config()
    assert cfg.bg_mode == "none"
    assert cfg.anim_bg_mode == "current_frame"
    assert cfg.show_deformed is False
    assert cfg.anim_show_deformed is True


def test_showing_the_background_still_picks_the_frame_the_old_way(dialog):
    """Back-compat: with the box ticked, behaviour is what it always was."""
    dialog._img_bg_check.setChecked(True)
    dialog._img_geom_combo.setCurrentIndex(0)
    assert dialog.get_config().bg_mode == "current_frame"
    dialog._img_geom_combo.setCurrentIndex(1)
    assert dialog.get_config().bg_mode == "ref_frame"


def test_the_fill_flows_from_the_preview_tab(dialog):
    """The colour lives with the other page-styling settings, not per tab."""
    idx = dialog._pv_hidden_bg_combo.findData("transparent")
    assert idx >= 0, "transparent should be offered"
    dialog._pv_hidden_bg_combo.setCurrentIndex(idx)
    assert dialog.get_config().hidden_bg_color == "transparent"


def test_the_preview_renders_with_no_background(dialog):
    """The preview path resolves bg_mode separately -- it can drift."""
    dialog._img_bg_check.setChecked(False)
    idx = dialog._pv_hidden_bg_combo.findData("transparent")
    dialog._pv_hidden_bg_combo.setCurrentIndex(idx)
    dialog._render_preview()          # must not raise on a 4-channel image


# --- the fill has to be reachable, not just readable ----------------------

@pytest.fixture
def sidebar():
    from al_dic.gui.app_state import AppState as _AS
    from al_dic.gui.controllers.image_controller import ImageController
    from al_dic.gui.controllers.pipeline_controller import PipelineController
    from al_dic.gui.panels.right_sidebar import RightSidebar

    _AS._instance = None
    state = _AS.instance()
    ctrl = PipelineController(state, ImageController(state))
    panel = RightSidebar(ctrl)
    yield panel, state
    _AS._instance = None


def test_the_sidebar_can_actually_set_the_fill(sidebar):
    """Every render path read state.hidden_bg_color and nothing wrote it, so
    the on-screen fill was permanently white whatever was configured."""
    panel, state = sidebar
    panel._background_cb.setChecked(False)
    panel._hidden_bg_combo.setCurrentIndex(
        panel._hidden_bg_combo.findData("black"))
    assert state.hidden_bg_color == "black"


def test_the_fill_is_disabled_while_the_image_is_shown(sidebar):
    panel, _ = sidebar
    panel._background_cb.setChecked(True)
    assert panel._hidden_bg_combo.isEnabled() is False
    panel._background_cb.setChecked(False)
    assert panel._hidden_bg_combo.isEnabled() is True


def test_the_export_dialog_opens_on_the_stored_fill():
    """The combo used to always start at white, so an export silently ignored
    what the window was already showing."""
    hint = VizExportHint(hidden_bg_color="transparent", show_background=False)
    dlg = ExportDialog(_result(), None, hint, image_files=[])
    assert dlg._pv_hidden_bg_combo.currentData() == "transparent"
    assert dlg.get_config().hidden_bg_color == "transparent"


# --- the fill must not leak between windows -------------------------------

def test_the_strain_window_starts_from_the_main_windows_fill():
    """Opening it should not silently change what you already chose."""
    from al_dic.gui.strain_window import StrainWindow

    state = AppState()
    state.hidden_bg_color = "black"
    win = StrainWindow(state)
    assert win._viz_panel.get_state()["hidden_bg_color"] == "black"


def test_the_strain_window_never_writes_the_shared_fill():
    """It used to write its panel's value into AppState on every refresh,
    while never reading it -- so its own default (white) clobbered the main
    window's choice on every frame change and every viz edit."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / ".." / "src" / "al_dic"
           / "gui" / "strain_window.py").resolve()
    tree = ast.parse(src.read_text(encoding="utf-8"), str(src))
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and t.attr == "hidden_bg_color"
    ]
    assert not writes, (
        f"strain_window.py assigns hidden_bg_color at line(s) {writes}; "
        "its fill is panel-local, like its colormap and opacity"
    )
