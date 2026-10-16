""""Hide untracked regions" reaches every surface: state, sidebar, canvas,
strain window, session, export dialog (images, animation, preview)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult,
)
from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.viz_controller import VizController
from al_dic.gui.dialogs.export_dialog import ExportDialog, VizExportHint

app = QApplication.instance() or QApplication([])


def _result(img=128, dead=False):
    xs, ys = np.meshgrid(np.linspace(8, img - 8, 6), np.linspace(8, img - 8, 6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    n = coords.shape[0]
    mesh = DICMesh(coordinates_fem=coords, elements_fem=np.zeros((0, 8), np.int64))
    u = coords[:, 0] / img
    U = np.repeat(u, 2)
    if dead:
        U[0:2] = np.nan                         # node 0 lost
    fr = FrameResult(U=U.copy(), U_accum=U.copy())
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


class _SpyViz(VizController):
    def __init__(self):
        super().__init__()
        self.calls = []

    def render_field(self, *args, **kwargs):
        self.calls.append(kwargs)
        return super().render_field(*args, **kwargs)


@pytest.fixture
def fresh_state():
    AppState._instance = None
    yield AppState.instance()
    AppState._instance = None


# --- state + session ---------------------------------------------------------

def test_state_defaults_to_hiding(fresh_state):
    assert fresh_state.hide_untracked is True


def test_session_saves_and_restores_the_toggle(fresh_state):
    from al_dic.gui.session import _VIEW_SCALAR_KEYS, _apply_view_state, _build_view_state

    assert "hide_untracked" in _VIEW_SCALAR_KEYS
    fresh_state.hide_untracked = False
    vs = _build_view_state(fresh_state)
    assert vs["hide_untracked"] is False
    AppState._instance = None
    other = AppState.instance()
    _apply_view_state(vs, other)
    assert other.hide_untracked is False


def test_a_session_without_the_key_keeps_the_default(fresh_state):
    from al_dic.gui.session import _apply_view_state

    _apply_view_state({"show_mesh": True}, fresh_state)
    assert fresh_state.hide_untracked is True


# --- main window -----------------------------------------------------------------

@pytest.fixture
def sidebar(fresh_state):
    from al_dic.gui.controllers.image_controller import ImageController
    from al_dic.gui.controllers.pipeline_controller import PipelineController
    from al_dic.gui.panels.right_sidebar import RightSidebar

    ctrl = PipelineController(fresh_state, ImageController(fresh_state))
    return RightSidebar(ctrl), fresh_state


def test_sidebar_checkbox_drives_the_state(sidebar):
    panel, state = sidebar
    fired = []
    state.display_changed.connect(lambda: fired.append(1))
    assert panel._hide_untracked_cb.isChecked() is True
    panel._hide_untracked_cb.setChecked(False)
    assert state.hide_untracked is False and fired


def test_sidebar_follows_an_external_change(sidebar):
    panel, state = sidebar
    state.hide_untracked = False
    panel._sync_display_controls()
    assert panel._hide_untracked_cb.isChecked() is False


def test_main_export_hint_carries_the_toggle(sidebar, monkeypatch):
    panel, state = sidebar
    state.results = _result()
    state.hide_untracked = False
    seen = {}

    class _Dlg:
        def __init__(self, results, folder, hint, **kw):
            seen["hint"] = hint

        def exec(self):
            return 0

        def deleteLater(self):  # noqa: N802 -- Qt name
            pass

    monkeypatch.setattr("al_dic.gui.dialogs.export_dialog.ExportDialog", _Dlg)
    panel._on_export()
    assert seen["hint"].hide_untracked is False


@pytest.mark.parametrize("hide", [True, False])
def test_canvas_passes_flag_and_tracked_nodes(fresh_state, hide):
    from al_dic.gui.controllers.image_controller import ImageController
    from al_dic.gui.panels.canvas_area import CanvasArea

    spy = _SpyViz()
    area = CanvasArea(ImageController(fresh_state), viz_ctrl=spy)
    fresh_state.results = _result(dead=True)
    fresh_state.set_roi_mask(np.ones((128, 128), bool))
    fresh_state.hide_untracked = hide
    fresh_state.current_frame = 1
    area._refresh_overlay()
    assert spy.calls, "canvas did not render"
    kw = spy.calls[-1]
    assert kw["hide_untracked"] is hide
    assert kw["tracked"].dtype == bool
    assert not kw["tracked"][0] and kw["tracked"][1:].all()


# --- strain window ------------------------------------------------------------------

def test_strain_panel_toggle_default_and_state():
    from al_dic.gui.widgets.strain_viz_panel import StrainVizPanel

    panel = StrainVizPanel()
    assert panel.get_state()["hide_untracked"] is True
    fired = []
    panel.viz_changed.connect(lambda: fired.append(1))
    panel._hide_untracked_check.setChecked(False)
    assert panel.get_state()["hide_untracked"] is False and fired
    fired.clear()
    panel.set_hide_untracked(True)
    assert panel.get_state()["hide_untracked"] is True and not fired


def test_strain_window_starts_from_the_main_window_choice(fresh_state):
    from al_dic.gui.strain_window import StrainWindow

    fresh_state.hide_untracked = False
    win = StrainWindow(fresh_state)
    assert win._viz_panel.get_state()["hide_untracked"] is False


def test_strain_window_renders_with_flag_and_tracked(fresh_state):
    from al_dic.gui.strain_window import StrainWindow

    fresh_state.results = _result(dead=True)
    win = StrainWindow(fresh_state)
    spy = _SpyViz()
    win._viz_ctrl = spy
    win._field_image("disp_u", 1, fresh_state.results, deformed=False, cmap="jet",
                     alpha=0.7, fill_trimmed=False, hide_untracked=True,
                     range_of=lambda v: (0.0, 1.0))
    kw = spy.calls[-1]
    assert kw["hide_untracked"] is True
    assert not kw["tracked"][0] and kw["tracked"][1:].all()


# --- export dialog ---------------------------------------------------------------------

@pytest.mark.parametrize("hide", [True, False])
def test_dialog_checkboxes_start_from_the_hint(hide):
    dlg = ExportDialog(_result(), None, VizExportHint(hide_untracked=hide), image_files=[])
    assert dlg._img_hide_check.isChecked() is hide
    assert dlg._anim_hide_check.isChecked() is hide
    cfg = dlg.get_config()
    assert cfg.hide_untracked is hide and cfg.anim_hide_untracked is hide


def test_dialog_images_and_animation_are_independent():
    dlg = ExportDialog(_result(), None, VizExportHint(), image_files=[])
    dlg._img_hide_check.setChecked(False)
    cfg = dlg.get_config()
    assert cfg.hide_untracked is False and cfg.anim_hide_untracked is True


def test_preview_renders_with_the_images_toggle(monkeypatch):
    import al_dic.export.export_png as ep

    seen = []
    real = ep.render_field_frame

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    # The preview imports render_field_frame from export_png at call time.
    monkeypatch.setattr(ep, "render_field_frame", spy)
    dlg = ExportDialog(_result(dead=True), None, VizExportHint(hide_untracked=True),
                       image_files=[])
    dlg._tabs.setCurrentIndex(dlg._preview_tab_index)  # populates the field list
    dlg._render_preview()
    assert seen and seen[-1]["hide_untracked"] is True
    assert not seen[-1]["tracked"][0]
    dlg._img_hide_check.setChecked(False)
    dlg._render_preview()
    assert seen[-1]["hide_untracked"] is False


def test_workers_pass_the_flags(monkeypatch, tmp_path):
    import al_dic.export.export_animation as ea
    import al_dic.export.export_png as ep
    from al_dic.gui.dialogs.export_dialog import (
        ExportAnimationWorker, ExportConfig, ExportImagesWorker,
    )

    got = {}
    monkeypatch.setattr(ep, "export_png", lambda **kw: got.setdefault("img", kw) and [])
    monkeypatch.setattr(ea, "export_animation", lambda **kw: got.setdefault("anim", kw) and [])
    cfg = ExportConfig(dest_dir=tmp_path, prefix="p", timestamp="t",
                       hide_untracked=False, anim_hide_untracked=True)
    ExportImagesWorker(cfg, _result(), [], None, None).run()
    ExportAnimationWorker(cfg, _result(), [], None, None).run()
    assert got["img"]["hide_untracked"] is False
    assert got["anim"]["hide_untracked"] is True
