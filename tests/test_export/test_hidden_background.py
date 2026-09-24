"""Rendering a field with no background image behind it.

The field used to be inseparable from a frame: `show_deformed` chose both the
backing image and the node geometry. These tests cover the third case -- no
image at all -- and the fill that replaces it.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from al_dic.export.colorbar import ColorbarStyle, add_margin, attach_colorbar
from al_dic.export.export_png import (
    _load_frame_image,
    export_png,
    render_field_frame,
)
from al_dic.gui.dialogs.export_dialog import FieldImageConfig


@pytest.fixture
def field_cfg():
    return FieldImageConfig(
        field_name="disp_u", enabled=True, colormap="jet",
        auto_range=True, vmin=0.0, vmax=1.0, bg_alpha=1.0,
    )


# --- choosing no background ----------------------------------------------

def test_none_mode_loads_nothing_even_when_images_exist(tmp_path):
    """"none" is a deliberate choice, not the accident of an empty list.

    _load_frame_image already returned None for a missing file, which is why
    the render layer needs no new no-image path -- but it must return None
    here for a reason the caller can rely on.
    """
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), np.zeros((8, 8), dtype=np.uint8))
    files = [str(img)]

    assert _load_frame_image(files, 0, "none") is None
    assert _load_frame_image(files, 0, "ref_frame") is not None


# --- the fill that replaces the image ------------------------------------

def test_the_default_fill_is_still_black(minimal_result, field_cfg):
    """Existing callers pass no fill and must keep getting what they got.

    Roughly forty call sites in tests/test_export render with bg_image=None
    today; a changed default would silently rewrite all of their expectations.
    """
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0])
    img = render_field_frame(
        coords=coords, values=values, image_shape=(64, 64),
        bg_image=None, field_cfg=field_cfg,
    )
    assert img.shape == (64, 64, 3)
    assert img[0, 0].tolist() == [0, 0, 0]        # corner is outside the hull


@pytest.mark.parametrize("colour,expected", [
    ("white", [255, 255, 255]),
    ("black", [0, 0, 0]),
])
def test_a_solid_fill_paints_everything_outside_the_field(
    minimal_result, field_cfg, colour, expected,
):
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0])
    img = render_field_frame(
        coords=coords, values=values, image_shape=(64, 64),
        bg_image=None, field_cfg=field_cfg, hidden_bg_color=colour,
    )
    assert img.shape == (64, 64, 3)
    assert img[0, 0].tolist() == expected


def test_transparent_returns_an_alpha_channel(minimal_result, field_cfg):
    """Outside the field is transparent, inside is opaque."""
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0])
    img = render_field_frame(
        coords=coords, values=values, image_shape=(64, 64),
        bg_image=None, field_cfg=field_cfg, hidden_bg_color="transparent",
    )
    assert img.shape == (64, 64, 4)
    alpha = img[:, :, 3]
    assert alpha[0, 0] == 0, "the corner sits outside the hull"
    assert alpha.max() == 255, "somewhere inside the field must be opaque"


def test_opacity_becomes_real_alpha_when_there_is_nothing_to_blend_with(
    minimal_result,
):
    """A 50% field over black is a darker field, which is not what was asked.

    With no image behind it there is nothing to blend toward, so the opacity
    slider has to mean transparency instead -- otherwise 'field only' at 70%
    silently returns a muddied colormap.
    """
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0])
    half = FieldImageConfig(
        field_name="disp_u", enabled=True, colormap="jet",
        auto_range=True, vmin=0.0, vmax=1.0, bg_alpha=0.5,
    )
    img = render_field_frame(
        coords=coords, values=values, image_shape=(64, 64),
        bg_image=None, field_cfg=half, hidden_bg_color="transparent",
    )
    inside = img[:, :, 3] > 0
    assert inside.any()
    # 0.5 opacity -> half-transparent, and the colour itself untouched
    assert 120 <= int(img[:, :, 3][inside].max()) <= 135


def test_a_real_background_is_never_made_transparent(minimal_result, field_cfg):
    """The fill only applies when the background is hidden."""
    bg = np.full((64, 64), 128, dtype=np.uint8)
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0])
    img = render_field_frame(
        coords=coords, values=values, image_shape=(64, 64),
        bg_image=bg, field_cfg=field_cfg, hidden_bg_color="transparent",
    )
    assert img.shape == (64, 64, 3)


# --- the rest of the post-render chain -----------------------------------

def test_margin_keeps_the_alpha_channel(field_cfg):
    """add_margin pads with a solid colour; on a transparent image the pad
    has to be transparent too, or the figure gains an opaque frame."""
    img = np.zeros((32, 32, 4), dtype=np.uint8)
    img[..., 3] = 255
    out = add_margin(img, 0.25, "white")
    assert out.shape[2] == 4
    assert out[0, 0, 3] == 0, "the margin must not be opaque"
    assert out[out.shape[0] // 2, out.shape[1] // 2, 3] == 255


def test_the_colorbar_can_be_attached_to_a_transparent_image():
    """The strip is rendered BGR; stacking it onto BGRA needs a channel."""
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    img[..., 3] = 255
    out = attach_colorbar(img, ColorbarStyle(), "jet", 0.0, 1.0, "u")
    assert out.shape[2] == 4, "attach_colorbar dropped the alpha channel"
    assert out.shape[:2] != (64, 64), "nothing was actually attached"


# --- end to end ----------------------------------------------------------

def test_export_writes_png_with_alpha(minimal_result, tmp_path):
    out = export_png(
        dest_dir=tmp_path, prefix="exp", timestamp="ts",
        results=minimal_result,
        configs=[FieldImageConfig(
            field_name="disp_u", enabled=True, colormap="jet",
            auto_range=True, vmin=0.0, vmax=1.0, bg_alpha=1.0,
        )],
        image_files=[], bg_mode="none",
        roi_mask=None, dpi=72, show_deformed=False,
        frame_start=0, frame_end=0,
        image_format="png", hidden_bg_color="transparent",
    )
    assert out, "no file written"
    data = cv2.imdecode(np.fromfile(str(out[0]), dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED)
    assert data.shape[2] == 4, "PNG was written without an alpha channel"


def test_jpeg_falls_back_instead_of_failing(minimal_result, tmp_path):
    """JPEG has no alpha. Refusing the export would be worse than filling."""
    out = export_png(
        dest_dir=tmp_path, prefix="exp", timestamp="ts",
        results=minimal_result,
        configs=[FieldImageConfig(
            field_name="disp_u", enabled=True, colormap="jet",
            auto_range=True, vmin=0.0, vmax=1.0, bg_alpha=1.0,
        )],
        image_files=[], bg_mode="none",
        roi_mask=None, dpi=72, show_deformed=False,
        frame_start=0, frame_end=0,
        image_format="jpeg", hidden_bg_color="transparent",
    )
    assert out
    data = cv2.imdecode(np.fromfile(str(out[0]), dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED)
    assert data.ndim == 3 and data.shape[2] == 3
    assert data[0, 0].min() > 200, "should have fallen back to a white fill"
