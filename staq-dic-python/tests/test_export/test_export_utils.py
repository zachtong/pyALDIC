"""Tests for export utility functions."""

from pathlib import Path

from staq_dic.export.export_utils import frame_tag, make_prefix, make_timestamp


def test_frame_tag_1based_zero_padded_small():
    assert frame_tag(0, 10) == "frame_01"
    assert frame_tag(9, 10) == "frame_10"


def test_frame_tag_1based_zero_padded_large():
    assert frame_tag(0, 100) == "frame_001"
    assert frame_tag(99, 100) == "frame_100"


def test_frame_tag_single_frame():
    assert frame_tag(0, 1) == "frame_1"


def test_make_prefix_uses_folder_name():
    assert make_prefix(Path("/data/experiment_A")) == "experiment_A"


def test_make_prefix_none_returns_dic():
    assert make_prefix(None) == "dic"


def test_make_prefix_sanitizes_spaces():
    result = make_prefix(Path("/data/my experiment"))
    assert " " not in result
    assert "my" in result


def test_make_prefix_sanitizes_special_chars():
    result = make_prefix(Path("/data/test:result"))
    assert ":" not in result


def test_make_timestamp_format():
    ts = make_timestamp()
    assert len(ts) == 14  # YYYYMMDDHHMMSS
    assert ts.isdigit()
