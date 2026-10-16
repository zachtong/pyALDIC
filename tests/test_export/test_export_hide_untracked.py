"""Image export with "Hide untracked regions" (render_field_frame + entry points)."""
from __future__ import annotations

import numpy as np
import pytest

import al_dic.export.export_png as ep
from al_dic.export.export_png import export_png, render_field_frame
from al_dic.gui.dialogs.export_dialog import FieldImageConfig

H = W = 96
STEP = 8


def _cfg():
    return FieldImageConfig("disp_u", True, "jet", False, -2.0, 2.0, 1.0)


def _mesh(mask=None):
    xs = np.arange(STEP, W - STEP + 1, STEP, float)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    if mask is not None:
        pts = pts[mask[pts[:, 1].astype(int), pts[:, 0].astype(int)]]
    return pts


def _alpha(coords, values, *, hide, tracked=None, roi=None, deformed=None, dmask=None,
           blank=True):
    img = render_field_frame(
        coords=coords, values=values, image_shape=(H, W), bg_image=None, field_cfg=_cfg(),
        roi_mask=np.ones((H, W), bool) if roi is None else roi, deformed_coords=deformed,
        deformed_mask=dmask, blank_invalid_nodes=blank, hidden_bg_color="transparent",
        hide_untracked=hide, tracked=tracked, mesh_step=STEP)
    return img[..., 3] > 0


def test_off_is_the_default_and_unchanged():
    coords = _mesh()
    values = coords[:, 0] / 50.0
    a = render_field_frame(coords=coords, values=values, image_shape=(H, W), bg_image=None,
                           field_cfg=_cfg(), roi_mask=np.ones((H, W), bool),
                           hidden_bg_color="transparent")
    b = _alpha(coords, values, hide=False)
    np.testing.assert_array_equal(a[..., 3] > 0, b)


def test_dead_region_hidden_reference_view():
    coords = _mesh()
    values = np.ones(len(coords))
    dead = coords[:, 0] >= 56
    values[dead] = np.nan
    a = _alpha(coords, values, hide=True, tracked=~dead)
    assert a[50, 30] and not a[50, 85]
    # Displacement is never trim-blanked: within reach of tracked nodes the
    # field continues past the live/dead mid-line (consistent with screen).
    assert a[50, 55]


def test_dead_region_hidden_deformed_view():
    coords = _mesh()
    values = np.ones(len(coords))
    dead = coords[:, 1] >= 56
    pos = coords.copy()
    pos[dead] = np.nan
    values[dead] = np.nan
    a = _alpha(coords, values, hide=True, tracked=~dead, deformed=pos,
               dmask=np.ones((H, W), bool))
    assert a[20, 40] and not a[85, 40]


def test_slot_edges_are_painted_to_the_mask_edge():
    mask = np.ones((H, W), bool)
    mask[44:52, 10:96] = False                       # a slot like the lattice gaps
    coords = _mesh(mask)
    values = coords[:, 1] / 50.0
    off = _alpha(coords, values, hide=False, roi=mask)
    on = _alpha(coords, values, hide=True, tracked=np.ones(len(coords), bool), roi=mask)
    assert on.sum() > off.sum()
    assert on[42, 60] and on[53, 60]                 # both slot faces reach the mask edge
    assert not on[47, 60]                            # the slot itself stays empty
    assert on[mask].mean() > 0.99


def test_strain_trim_band_stays_blank():
    coords = _mesh()
    strain = np.ones(len(coords))
    trimmed = coords[:, 0] == STEP
    strain[trimmed] = np.nan
    a = _alpha(coords, strain, hide=True, tracked=np.ones(len(coords), bool), blank=True)
    assert not a[48, STEP] and a[48, 48]


def test_export_png_defaults_on_and_passes_tracked(tmp_path, minimal_result, monkeypatch):
    seen = []
    real = ep.render_field_frame

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(ep, "render_field_frame", spy)
    export_png(tmp_path, "t", "20260101000000", minimal_result, [_cfg()], [], "none", None,
               72, False, 0, 0)
    assert seen and seen[0]["hide_untracked"] is True
    assert seen[0]["mesh_step"] == minimal_result.dic_para.winstepsize
    assert seen[0]["tracked"].dtype == bool and seen[0]["tracked"].all()

    seen.clear()
    export_png(tmp_path, "t", "20260101000001", minimal_result, [_cfg()], [], "none", None,
               72, False, 0, 0, hide_untracked=False)
    assert seen[0]["hide_untracked"] is False


@pytest.mark.parametrize("hide", [True, False])
def test_export_animation_passes_flag(tmp_path, minimal_result, monkeypatch, hide):
    import al_dic.export.export_animation as ea
    seen = []
    real = ep.render_field_frame

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    # export_animation imports render_field_frame from export_png at call time.
    monkeypatch.setattr(ep, "render_field_frame", spy)
    try:
        ea.export_animation(tmp_path, "t", "20260101000000", minimal_result, [_cfg()], [],
                            "none", None, "gif", 2, False, 0, 1, hide_untracked=hide)
    except Exception:
        pass  # encoder availability is not under test; the render call is
    assert seen and all(k["hide_untracked"] is hide for k in seen)


def test_without_a_mask_nothing_is_painted_beyond_the_node_hull():
    # No ROI = no edge to fill toward: hide-untracked must not grow a halo.
    coords = _mesh()
    values = coords[:, 0] / 50.0
    img = render_field_frame(
        coords=coords, values=values, image_shape=(H, W), bg_image=None, field_cfg=_cfg(),
        roi_mask=None, hidden_bg_color="transparent", hide_untracked=True,
        tracked=np.ones(len(coords), bool), mesh_step=STEP)
    a = img[..., 3] > 0
    assert a[48, 48]
    assert not a[2, 2] and not a[H - 2, W - 2]      # beyond the hull (nodes start at 8)
