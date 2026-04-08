"""Tests for NPZ export."""

import numpy as np
import pytest

from staq_dic.export.export_npz import export_npz


def test_npz_single_file_shapes(tmp_path, minimal_result):
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=True, per_frame=False)
    assert p.name.endswith(".npz")
    data = np.load(str(p))
    N, T = 12, 2  # 3x4 grid fixture
    assert data["coordinates"].shape == (N, 2)
    assert data["disp_u"].shape == (N, T)
    assert data["disp_v"].shape == (N, T)
    assert data["strain_exx"].shape == (N, T)
    assert data["strain_eyy"].shape == (N, T)


def test_npz_no_strain_if_excluded(tmp_path, minimal_result):
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=False, per_frame=False)
    data = np.load(str(p))
    assert "strain_exx" not in data
    assert "disp_u" in data


def test_npz_no_disp_if_excluded(tmp_path, minimal_result):
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   include_disp=False, include_strain=True, per_frame=False)
    data = np.load(str(p))
    assert "disp_u" not in data
    assert "strain_exx" in data


def test_npz_per_frame_returns_list(tmp_path, minimal_result):
    paths = export_npz(tmp_path, "exp", "ts", minimal_result,
                       include_disp=True, include_strain=True, per_frame=True)
    assert isinstance(paths, list)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        data = np.load(str(p))
        assert "disp_u" in data
        assert data["disp_u"].shape == (12,)  # single frame: (N,) not (N,T), N=12


def test_npz_single_file_returns_path(tmp_path, minimal_result):
    result = export_npz(tmp_path, "exp", "ts", minimal_result,
                        include_disp=True, include_strain=False, per_frame=False)
    from pathlib import Path
    assert isinstance(result, Path)


def test_npz_has_accum_disp(tmp_path, minimal_result):
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   include_disp=True, include_strain=False, per_frame=False)
    data = np.load(str(p))
    # minimal_result has U_accum set
    assert "disp_u_accum" in data


def test_npz_no_strain_result_skips_strain(tmp_path, minimal_result_no_strain):
    p = export_npz(tmp_path, "exp", "ts", minimal_result_no_strain,
                   include_disp=True, include_strain=True, per_frame=False)
    data = np.load(str(p))
    assert "strain_exx" not in data
