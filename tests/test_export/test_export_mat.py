"""Tests for MATLAB .mat export."""

import numpy as np
import pytest
import scipy.io

from al_dic.export.export_mat import export_mat

_ALL_DISP = ["disp_u", "disp_v", "disp_magnitude"]
_ALL_STRAIN = [
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
]


def test_mat_keys_present(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   fields=_ALL_DISP + _ALL_STRAIN)
    mat = scipy.io.loadmat(str(p))
    assert "CoordinatesFEM" in mat
    assert "disp_U" in mat
    assert "disp_V" in mat
    assert "strain_exx" in mat


def test_mat_shapes(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   fields=_ALL_DISP + _ALL_STRAIN)
    mat = scipy.io.loadmat(str(p))
    assert mat["CoordinatesFEM"].shape == (12, 2)  # 3x4 grid fixture
    assert mat["disp_U"].shape == (12, 2)   # N x T
    assert mat["strain_exx"].shape == (12, 2)


def test_mat_no_strain_if_excluded(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result, fields=_ALL_DISP)
    mat = scipy.io.loadmat(str(p))
    assert "strain_exx" not in mat
    assert "disp_U" in mat


def test_mat_no_disp_if_excluded(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result, fields=_ALL_STRAIN)
    mat = scipy.io.loadmat(str(p))
    assert "disp_U" not in mat
    assert "strain_exx" in mat


def test_mat_disp_uses_accum_when_available(tmp_path, minimal_result):
    """disp_U/disp_V should contain accumulated displacement (U_accum)."""
    p = export_mat(tmp_path, "exp", "ts", minimal_result, fields=_ALL_DISP)
    mat = scipy.io.loadmat(str(p))
    assert "disp_U" in mat
    # No separate accum fields — U/V *are* the accumulated displacement
    assert "disp_U_accum" not in mat
    assert "disp_V_accum" not in mat


def test_mat_filename(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "20260408", minimal_result, fields=_ALL_DISP)
    assert p.name == "exp_results_20260408.mat"


def test_mat_selective_fields(tmp_path, minimal_result):
    """Only requested fields are exported."""
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   fields=["disp_u", "strain_exx"])
    mat = scipy.io.loadmat(str(p))
    assert "disp_U" in mat
    assert "strain_exx" in mat
    assert "disp_V" not in mat
    assert "strain_eyy" not in mat


def test_mat_coordinates_always_present(tmp_path, minimal_result):
    """CoordinatesFEM always exported regardless of fields list."""
    p = export_mat(tmp_path, "exp", "ts", minimal_result, fields=[])
    mat = scipy.io.loadmat(str(p))
    assert "CoordinatesFEM" in mat
