"""Export pipeline results to MATLAB .mat format.

Variable naming follows MATLAB AL-DIC conventions:
    CoordinatesFEM  (N, 2)
    disp_U          (N, T)   -- x-displacement (accumulated, all frames)
    disp_V          (N, T)
    strain_exx      (N, T)   -- strain components
    ...
    DICpara         struct   -- scalar parameters (arrays excluded)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir

# Canonical strain field names (match StrainResult attribute names)
_STRAIN_FIELDS: frozenset[str] = frozenset([
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
])

# Canonical displacement field names
_DISP_FIELDS: frozenset[str] = frozenset([
    "disp_u", "disp_v", "disp_magnitude",
])

# Mapping from canonical field name -> MATLAB variable name
_MAT_NAMES: dict[str, str] = {
    "disp_u":         "disp_U",
    "disp_v":         "disp_V",
    "disp_magnitude": "disp_magnitude",
}


def _dicpara_struct(results: PipelineResult) -> dict[str, Any]:
    """Build a MATLAB-compatible struct dict from DICPara (scalars only)."""
    p = results.dic_para
    struct: dict[str, Any] = {}
    for f in dataclasses.fields(p):
        v = getattr(p, f.name)
        if isinstance(v, (bool, int, float, str)):
            struct[f.name] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (int, float)) for x in v
        ):
            struct[f.name] = np.array(v, dtype=np.float64)
    return struct


def export_mat(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    fields: list[str],
) -> Path:
    """Write selected fields to a single MATLAB .mat file (N×T matrices).

    Args:
        dest_dir: Output directory (created if absent).
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        fields: List of canonical field names to export.
                'CoordinatesFEM' and 'DICpara' are always included.
                Unavailable fields are silently skipped.

    Returns:
        Path to the written .mat file.
    """
    ensure_dir(dest_dir)
    n_nodes = results.dic_mesh.coordinates_fem.shape[0]
    n_frames = len(results.result_disp)

    mat_dict: dict[str, Any] = {
        "CoordinatesFEM": results.dic_mesh.coordinates_fem,
        "DICpara": _dicpara_struct(results),
        "n_frames": n_frames,
        "n_nodes": n_nodes,
    }

    fields_set = set(fields)
    requested_disp = fields_set & _DISP_FIELDS
    requested_strain = fields_set & _STRAIN_FIELDS

    # --- Displacement fields ---
    # Always export accumulated displacement when available.
    if requested_disp and n_frames > 0:
        u_mat = np.full((n_nodes, n_frames), np.nan)
        v_mat = np.full((n_nodes, n_frames), np.nan)

        for t, fr in enumerate(results.result_disp):
            U = fr.U_accum if fr.U_accum is not None else fr.U
            u, v = split_uv(U)
            u_mat[:, t] = u
            v_mat[:, t] = v

        if "disp_u" in requested_disp:
            mat_dict["disp_U"] = u_mat
        if "disp_v" in requested_disp:
            mat_dict["disp_V"] = v_mat
        if "disp_magnitude" in requested_disp:
            mat_dict["disp_magnitude"] = np.sqrt(u_mat ** 2 + v_mat ** 2)

    # --- Strain fields ---
    if requested_strain and results.result_strain:
        n_strain_frames = len(results.result_strain)
        for field_name in requested_strain:
            mat = np.full((n_nodes, n_strain_frames), np.nan)
            for t, sr in enumerate(results.result_strain):
                val = getattr(sr, field_name, None)
                if val is not None:
                    mat[:, t] = val
            mat_dict[field_name] = mat

    out = dest_dir / f"{prefix}_results_{timestamp}.mat"
    scipy.io.savemat(str(out), mat_dict)
    return out
