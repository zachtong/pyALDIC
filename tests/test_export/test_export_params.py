"""Tests for parameters JSON export."""

import json

import numpy as np
import pytest

from al_dic.export.export_params import export_params


def test_params_file_always_written(tmp_path, minimal_result):
    p = export_params(tmp_path, "exp", "20260408000000", minimal_result)
    assert p.exists()
    assert p.name == "exp_parameters_20260408000000.json"


def test_params_contains_required_fields(tmp_path, minimal_result):
    p = export_params(tmp_path, "exp", "20260408000000", minimal_result)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["export_timestamp"] == "20260408000000"
    assert data["n_frames"] == len(minimal_result.result_disp)
    assert data["n_nodes"] == minimal_result.dic_mesh.coordinates_fem.shape[0]
    assert "has_strain" in data


def test_params_contains_dic_para_fields(tmp_path, minimal_result):
    p = export_params(tmp_path, "exp", "20260408000000", minimal_result)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "winsize" in data
    assert "winstepsize" in data
    assert "reference_mode" in data


def test_params_creates_dest_dir(tmp_path, minimal_result):
    subdir = tmp_path / "nested" / "output"
    export_params(subdir, "exp", "ts", minimal_result)
    assert subdir.exists()
