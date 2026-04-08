"""Tests for MATLAB .mat export."""

import numpy as np
import pytest
import scipy.io

from staq_dic.export.export_mat import export_mat


def test_mat_keys_present(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=True)
    mat = scipy.io.loadmat(str(p))
    assert "CoordinatesFEM" in mat
    assert "disp_U" in mat
    assert "disp_V" in mat
    assert "strain_exx" in mat


def test_mat_shapes(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=True)
    mat = scipy.io.loadmat(str(p))
    assert mat["CoordinatesFEM"].shape == (5, 2)
    assert mat["disp_U"].shape == (5, 2)   # N x T
    assert mat["strain_exx"].shape == (5, 2)


def test_mat_no_strain_if_excluded(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=False)
    mat = scipy.io.loadmat(str(p))
    assert "strain_exx" not in mat
    assert "disp_U" in mat


def test_mat_no_disp_if_excluded(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   include_disp=False, include_strain=True)
    mat = scipy.io.loadmat(str(p))
    assert "disp_U" not in mat
    assert "strain_exx" in mat


def test_mat_has_accum_disp(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=False)
    mat = scipy.io.loadmat(str(p))
    assert "disp_U_accum" in mat


def test_mat_filename(tmp_path, minimal_result):
    p = export_mat(tmp_path, "exp", "20260408", minimal_result,
                   include_disp=True, include_strain=False)
    assert p.name == "exp_results_20260408.mat"
