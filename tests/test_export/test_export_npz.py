"""Tests for NPZ export."""

import numpy as np
import pytest
from pathlib import Path

from al_dic.export.export_npz import export_npz

# Canonical field groups for tests
_ALL_DISP = ["disp_u", "disp_v", "disp_magnitude"]
_ALL_STRAIN = [
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
]


def test_npz_single_file_shapes(tmp_path, minimal_result):
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   fields=_ALL_DISP + _ALL_STRAIN, per_frame=False)
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
                   fields=_ALL_DISP, per_frame=False)
    data = np.load(str(p))
    assert "strain_exx" not in data
    assert "disp_u" in data


def test_npz_no_disp_if_excluded(tmp_path, minimal_result):
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   fields=_ALL_STRAIN, per_frame=False)
    data = np.load(str(p))
    assert "disp_u" not in data
    assert "strain_exx" in data


def test_npz_per_frame_returns_list(tmp_path, minimal_result):
    paths = export_npz(tmp_path, "exp", "ts", minimal_result,
                       fields=_ALL_DISP + _ALL_STRAIN, per_frame=True)
    assert isinstance(paths, list)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        data = np.load(str(p))
        assert "disp_u" in data
        assert data["disp_u"].shape == (12,)  # single frame: (N,) not (N,T), N=12


def test_npz_single_file_returns_path(tmp_path, minimal_result):
    result = export_npz(tmp_path, "exp", "ts", minimal_result,
                        fields=_ALL_DISP, per_frame=False)
    assert isinstance(result, Path)


def test_npz_disp_uses_accum_when_available(tmp_path, minimal_result):
    """disp_u/disp_v should contain accumulated displacement (U_accum)."""
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   fields=_ALL_DISP, per_frame=False)
    data = np.load(str(p))
    assert "disp_u" in data
    # No separate accum fields — U/V *are* the accumulated displacement
    assert "disp_u_accum" not in data
    assert "disp_v_accum" not in data


def test_npz_no_strain_result_skips_strain(tmp_path, minimal_result_no_strain):
    p = export_npz(tmp_path, "exp", "ts", minimal_result_no_strain,
                   fields=_ALL_DISP + _ALL_STRAIN, per_frame=False)
    data = np.load(str(p))
    assert "strain_exx" not in data


def test_npz_coordinates_always_present(tmp_path, minimal_result):
    """coordinates array is always exported regardless of fields list."""
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   fields=[], per_frame=False)
    data = np.load(str(p))
    assert "coordinates" in data


def test_npz_selective_fields(tmp_path, minimal_result):
    """Only requested fields are included (not the entire group)."""
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   fields=["disp_u", "strain_exx"], per_frame=False)
    data = np.load(str(p))
    assert "disp_u" in data
    assert "strain_exx" in data
    assert "disp_v" not in data
    assert "strain_eyy" not in data


def test_npz_no_accum_fields_exist(tmp_path, minimal_result):
    """No separate accum fields — disp_u/v ARE the accumulated displacement."""
    p = export_npz(tmp_path, "exp", "ts", minimal_result,
                   fields=_ALL_DISP, per_frame=False)
    data = np.load(str(p))
    assert "disp_u_accum" not in data
    assert "disp_v_accum" not in data
