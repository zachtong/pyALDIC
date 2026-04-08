"""Tests for per-frame CSV export."""

import csv
from pathlib import Path

import pytest

from staq_dic.export.export_csv import export_csv


def test_csv_per_frame_count(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=True)
    assert len(paths) == 2


def test_csv_files_exist(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=True)
    for p in paths:
        assert p.exists()


def test_csv_has_disp_columns(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=False)
    with open(paths[0], encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert "node_id" in row
    assert "x_px" in row
    assert "y_px" in row
    assert "disp_u" in row
    assert "disp_v" in row
    assert "disp_magnitude" in row


def test_csv_has_strain_columns(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=True)
    with open(paths[0], encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert "strain_exx" in row
    assert "strain_eyy" in row


def test_csv_no_strain_columns_if_excluded(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=False)
    with open(paths[0], encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    assert "strain_exx" not in fieldnames


def test_csv_row_count_matches_nodes(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=False)
    with open(paths[0], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12  # 3x4 grid = 12 nodes in minimal_result


def test_csv_utf8_bom(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=False)
    with open(paths[0], "rb") as f:
        header = f.read(3)
    assert header == b"\xef\xbb\xbf"  # UTF-8 BOM


def test_csv_in_subdir(tmp_path, minimal_result):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=False)
    # All CSV files should be inside a dedicated subdirectory
    assert paths[0].parent.name.startswith("exp_csv_")


def test_csv_no_strain_result_skips_strain_cols(tmp_path, minimal_result_no_strain):
    paths = export_csv(tmp_path, "exp", "ts", minimal_result_no_strain,
                       include_disp=True, include_strain=True)
    with open(paths[0], encoding="utf-8-sig") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert "strain_exx" not in fieldnames
