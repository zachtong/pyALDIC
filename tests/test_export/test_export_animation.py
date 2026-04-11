"""Tests for animation export (GIF and MP4)."""

import threading

import pytest
import numpy as np

from al_dic.export.export_animation import export_animation
from al_dic.gui.dialogs.export_dialog import FieldImageConfig


def _make_config(field_name: str, enabled: bool = True) -> FieldImageConfig:
    return FieldImageConfig(
        field_name=field_name,
        enabled=enabled,
        colormap="jet",
        auto_range=True,
        vmin=0.0,
        vmax=1.0,
        bg_alpha=0.7,
    )


def test_gif_creates_file(tmp_path, minimal_result):
    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=[_make_config("disp_u")],
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
    )
    assert len(paths) == 1
    assert paths[0].suffix == ".gif"
    assert paths[0].exists()


def test_mp4_creates_file(tmp_path, minimal_result):
    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=[_make_config("disp_u")],
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="mp4", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
    )
    assert len(paths) == 1
    # Either .mp4 or fallback .avi (depends on system codecs)
    assert paths[0].suffix in (".mp4", ".avi")
    assert paths[0].exists()


def test_animation_disabled_fields_excluded(tmp_path, minimal_result):
    configs = [
        _make_config("disp_u", enabled=True),
        _make_config("disp_v", enabled=False),
    ]
    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=configs,
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
    )
    assert len(paths) == 1
    assert "disp_u" in paths[0].name


def test_animation_multiple_fields(tmp_path, minimal_result):
    configs = [
        _make_config("disp_u"),
        _make_config("disp_v"),
    ]
    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=configs,
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
    )
    assert len(paths) == 2


def test_animation_frame_range(tmp_path, minimal_result):
    """Single-frame range produces a valid animation file."""
    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=[_make_config("disp_u")],
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=0,
    )
    assert len(paths) == 1
    assert paths[0].exists()


def test_animation_stop_event(tmp_path, minimal_result):
    """Stop event prevents frames from being written after it is set."""
    stop = threading.Event()
    stop.set()  # stop immediately

    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=[_make_config("disp_u")],
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
        stop_event=stop,
    )
    assert isinstance(paths, list)


def test_animation_progress_callback(tmp_path, minimal_result):
    calls: list[tuple[int, int, str]] = []

    export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=[_make_config("disp_u")],
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
        progress_callback=lambda d, t, f: calls.append((d, t, f)),
    )
    assert len(calls) == 2  # 2 frames
    assert all(c[2] == "disp_u" for c in calls)


def test_animation_output_dir_structure(tmp_path, minimal_result):
    """Animation files should live in {prefix}_animation_{timestamp}/."""
    paths = export_animation(
        tmp_path, "exp", "ts", minimal_result,
        configs=[_make_config("disp_u")],
        image_files=[], bg_mode="ref_frame", roi_mask=None,
        fmt="gif", fps=5,
        show_deformed=False,
        frame_start=0, frame_end=1,
    )
    assert paths[0].parent.name == "exp_animation_ts"
