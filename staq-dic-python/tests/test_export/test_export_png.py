"""Tests for PNG batch image export backend."""

import threading

import cv2
import numpy as np
import pytest

from staq_dic.export.export_png import export_png, render_field_frame
from staq_dic.gui.dialogs.export_dialog import FieldImageConfig


@pytest.fixture
def field_cfg():
    return FieldImageConfig(
        field_name="disp_u",
        enabled=True,
        colormap="jet",
        auto_range=True,
        vmin=0.0,
        vmax=1.0,
        bg_alpha=0.7,
    )


def test_render_field_frame_returns_bgr(minimal_result, field_cfg):
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0], dtype=np.float64)
    img = render_field_frame(
        coords=coords,
        values=values,
        image_shape=(64, 64),
        ref_image=None,
        field_cfg=field_cfg,
    )
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3   # BGR
    assert img.dtype == np.uint8


def test_render_with_background(minimal_result, field_cfg):
    coords = minimal_result.dic_mesh.coordinates_fem
    values = np.ones(coords.shape[0], dtype=np.float64)
    ref = np.zeros((64, 64), dtype=np.uint8)
    img = render_field_frame(
        coords=coords,
        values=values,
        image_shape=(64, 64),
        ref_image=ref,
        field_cfg=field_cfg,
    )
    assert img is not None
    assert img.shape == (64, 64, 3)


def test_export_png_creates_files(tmp_path, minimal_result, field_cfg):
    paths = export_png(
        dest_dir=tmp_path,
        prefix="exp",
        timestamp="ts",
        results=minimal_result,
        configs=[field_cfg],
        ref_image=None,
        dpi=72,
        show_deformed=False,
        show_background=False,
        frame_start=0,
        frame_end=1,
    )
    assert len(paths) >= 1
    for p in paths:
        img = cv2.imread(str(p))
        assert img is not None


def test_export_png_only_enabled_fields(tmp_path, minimal_result):
    enabled_cfg = FieldImageConfig("disp_u", True, "jet", True, 0, 1, 0.7)
    disabled_cfg = FieldImageConfig("disp_v", False, "jet", True, 0, 1, 0.7)
    paths = export_png(
        dest_dir=tmp_path,
        prefix="exp",
        timestamp="ts",
        results=minimal_result,
        configs=[enabled_cfg, disabled_cfg],
        ref_image=None,
        dpi=72,
        show_deformed=False,
        show_background=False,
        frame_start=0,
        frame_end=0,
    )
    # Only disp_u should produce files
    field_dirs = [p.parent.name for p in paths]
    assert "disp_u" in field_dirs
    assert "disp_v" not in field_dirs


def test_export_png_stop_event(tmp_path, minimal_result, field_cfg):
    stop = threading.Event()
    stop.set()   # stop immediately
    paths = export_png(
        dest_dir=tmp_path,
        prefix="exp",
        timestamp="ts",
        results=minimal_result,
        configs=[field_cfg],
        ref_image=None,
        dpi=72,
        show_deformed=False,
        show_background=False,
        frame_start=0,
        frame_end=1,
        stop_event=stop,
    )
    # With stop set before start, no frames should be written
    assert len(paths) == 0


def test_export_png_dir_structure(tmp_path, minimal_result, field_cfg):
    paths = export_png(
        dest_dir=tmp_path,
        prefix="exp",
        timestamp="ts",
        results=minimal_result,
        configs=[field_cfg],
        ref_image=None,
        dpi=72,
        show_deformed=False,
        show_background=False,
        frame_start=0,
        frame_end=0,
    )
    # Expect: tmp_path/exp_images_ts/disp_u/frame_1.png
    assert len(paths) >= 1
    assert paths[0].parent.name == "disp_u"
    assert paths[0].parent.parent.name == "exp_images_ts"
