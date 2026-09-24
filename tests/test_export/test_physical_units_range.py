"""Colour ranges and physical units.

A fixed range is stored in the units the window was displaying, not in pixels:
``resolve_color_range`` hands back ``state.color_min/max`` untouched while the
values they bound have already been multiplied by the pixel size, and
``AppState.set_physical_units`` does not convert them when the unit label
changes. Exporting used to multiply them a second time, so a range typed as
+-200 um came out of the exporter as +-60000 um at 300 um/px.

An auto range is the opposite case: it has to be computed from the scaled
values, or it would bound micrometres with a pixel-sized window.
"""
from __future__ import annotations

import numpy as np
import pytest

import al_dic.export.export_animation as ea
import al_dic.export.export_png as ep
from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult,
)
from al_dic.gui.dialogs.export_dialog import FieldImageConfig

IMG = 96
PIXEL_SIZE = 300.0          # um/px
DISP_PX = 0.5               # +-0.5 px  ->  +-150 um


@pytest.fixture
def result():
    xs, ys = np.meshgrid(np.linspace(8, IMG - 8, 5), np.linspace(8, IMG - 8, 5))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    n = coords.shape[0]
    mesh = DICMesh(coordinates_fem=coords,
                   elements_fem=np.zeros((0, 8), np.int64))
    u = np.linspace(-DISP_PX, DISP_PX, n)
    fr = FrameResult(U=np.column_stack([u, u]).ravel(),
                     U_accum=np.column_stack([u, u]).ravel())
    sr = StrainResult(
        disp_u=u, disp_v=u, strain_exx=u, strain_eyy=u, strain_exy=u,
        strain_principal_max=u, strain_principal_min=u, strain_maxshear=u,
        strain_von_mises=u, strain_rotation=u)
    return PipelineResult(
        dic_para=dicpara_default(img_size=(IMG, IMG)), dic_mesh=mesh,
        result_disp=[fr], result_def_grad=[fr], result_strain=[sr],
        result_fe_mesh_each_frame=[mesh],
        frame_schedule=FrameSchedule.from_mode("accumulative", 2))


def _cfg(auto: bool, vmin: float = -200.0, vmax: float = 200.0):
    return FieldImageConfig(
        field_name="disp_u", enabled=True, colormap="jet",
        auto_range=auto, vmin=vmin, vmax=vmax, bg_alpha=1.0)


def _range_used(monkeypatch, run) -> tuple[float, float]:
    """The (vmin, vmax) the renderer is actually handed.

    Patched on export_png even for the animation path: that module imports
    render_field_frame inside the function, so the name it resolves is this
    one, not a copy of its own.
    """
    seen: list[tuple[float, float]] = []
    real = ep.render_field_frame

    def spy(*a, **kw):
        cfg = kw["field_cfg"] if "field_cfg" in kw else a[4]
        seen.append((cfg.vmin, cfg.vmax))
        return real(*a, **kw)

    monkeypatch.setattr(ep, "render_field_frame", spy)
    run()
    assert seen, "nothing was rendered"
    return seen[0]


# --- images ---------------------------------------------------------------

def test_a_fixed_range_is_not_rescaled_on_export(monkeypatch, result, tmp_path):
    """The reported bug: +-200 um exported as +-60000 um at 300 um/px."""
    got = _range_used(monkeypatch, lambda: ep.export_png(
        dest_dir=tmp_path, prefix="p", timestamp="t", results=result,
        configs=[_cfg(auto=False)], image_files=[], bg_mode="none",
        roi_mask=None, dpi=72, show_deformed=False,
        frame_start=0, frame_end=0, include_colorbar=True,
        use_physical_units=True, pixel_size=PIXEL_SIZE, pixel_unit="um"))
    assert got == (-200.0, 200.0)


def test_an_auto_range_still_follows_the_scaled_values(
    monkeypatch, result, tmp_path,
):
    """The other half: auto must bound micrometres, not pixels."""
    vmin, vmax = _range_used(monkeypatch, lambda: ep.export_png(
        dest_dir=tmp_path, prefix="p", timestamp="t", results=result,
        configs=[_cfg(auto=True)], image_files=[], bg_mode="none",
        roi_mask=None, dpi=72, show_deformed=False,
        frame_start=0, frame_end=0, include_colorbar=True,
        use_physical_units=True, pixel_size=PIXEL_SIZE, pixel_unit="um"))
    assert vmin == pytest.approx(-DISP_PX * PIXEL_SIZE)
    assert vmax == pytest.approx(DISP_PX * PIXEL_SIZE)


def test_a_fixed_range_is_untouched_without_physical_units(
    monkeypatch, result, tmp_path,
):
    got = _range_used(monkeypatch, lambda: ep.export_png(
        dest_dir=tmp_path, prefix="p", timestamp="t", results=result,
        configs=[_cfg(auto=False, vmin=-0.5, vmax=0.5)], image_files=[],
        bg_mode="none", roi_mask=None, dpi=72, show_deformed=False,
        frame_start=0, frame_end=0, include_colorbar=True,
        use_physical_units=False, pixel_size=PIXEL_SIZE, pixel_unit="um"))
    assert got == (-0.5, 0.5)


# --- animation ------------------------------------------------------------

def test_the_animation_exporter_agrees_with_the_image_one(
    monkeypatch, result, tmp_path,
):
    """Two copies of the same range logic; they drifted apart once already."""
    got = _range_used(monkeypatch, lambda: ea.export_animation(
        dest_dir=tmp_path, prefix="p", timestamp="t", results=result,
        configs=[_cfg(auto=False)], image_files=[], bg_mode="none",
        roi_mask=None, fmt="gif", fps=5, show_deformed=False,
        frame_start=0, frame_end=0, include_colorbar=True,
        use_physical_units=True, pixel_size=PIXEL_SIZE, pixel_unit="um"))
    assert got == (-200.0, 200.0)
