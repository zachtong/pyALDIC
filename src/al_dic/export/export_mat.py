"""Export pipeline results to MATLAB .mat format (scipy v5 MAT-file).

Variable names match the other data exports (NPZ / CSV) so a downstream script
reads the same keys regardless of format:
    coordinates     (N, 2)   -- reference node positions, pixels
    disp_u          (N, T)   -- x-displacement (accumulated, world coords)
    disp_v          (N, T)   -- y-displacement (world y-up, matches strain)
    disp_magnitude  (N, T)
    strain_exx      (N, T)   -- strain components (world coords)
    ...
    DICpara         struct   -- parameters (scalars, numeric tuples, and
                                nested structs; large arrays excluded)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from al_dic.core.data_structures import PipelineResult
from al_dic.export.export_utils import ensure_dir, shape_order_matrix, world_disp

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

def _mat_value(v: Any) -> Any:
    """Convert a DICPara field to a savemat-compatible value, or None to skip.

    Mirrors the JSON parameter export so the .mat ``DICpara`` struct carries
    the same fields (scalars, numeric tuples, and nested dataclasses such as
    the ROI range) rather than scalars only.  ``None`` and large arrays are
    skipped -- savemat cannot store ``None`` and arrays are summarised via
    ``n_nodes`` / the coordinate array.
    """
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)) and v and all(
        isinstance(x, (int, float)) for x in v
    ):
        return np.array(v, dtype=np.float64)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        sub: dict[str, Any] = {}
        for f in dataclasses.fields(v):
            mv = _mat_value(getattr(v, f.name))
            if mv is not None:
                sub[f.name] = mv
        return sub or None
    return None  # ndarray / None / unsupported -> skip


def _dicpara_struct(results: PipelineResult) -> dict[str, Any]:
    """Build a MATLAB struct dict from DICPara (scalars, numeric tuples, and
    nested dataclasses; large arrays excluded)."""
    p = results.dic_para
    struct: dict[str, Any] = {}
    for f in dataclasses.fields(p):
        mv = _mat_value(getattr(p, f.name))
        if mv is not None:
            struct[f.name] = mv
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
        "coordinates": results.dic_mesh.coordinates_fem,
        "DICpara": _dicpara_struct(results),
        "n_frames": n_frames,
        "n_nodes": n_nodes,
    }

    # Which shape-function order each node ended with; present only when
    # the second-order stage ran (see ``shape_order_matrix``).
    order_mat = shape_order_matrix(results)
    if order_mat is not None:
        mat_dict["shape_order"] = order_mat

    fields_set = set(fields)
    requested_disp = fields_set & _DISP_FIELDS
    requested_strain = fields_set & _STRAIN_FIELDS

    # --- Displacement fields (world convention, matches strain) ---
    # Always export accumulated displacement when available.
    if requested_disp and n_frames > 0:
        um2px = results.dic_para.um2px
        u_mat = np.full((n_nodes, n_frames), np.nan)
        v_mat = np.full((n_nodes, n_frames), np.nan)

        for t, fr in enumerate(results.result_disp):
            U = fr.U_accum if fr.U_accum is not None else fr.U
            u, v = world_disp(U, um2px)
            u_mat[:, t] = u
            v_mat[:, t] = v

        if "disp_u" in requested_disp:
            mat_dict["disp_u"] = u_mat
        if "disp_v" in requested_disp:
            mat_dict["disp_v"] = v_mat
        if "disp_magnitude" in requested_disp:
            mat_dict["disp_magnitude"] = np.sqrt(u_mat ** 2 + v_mat ** 2)

    # --- Strain fields (columns aligned to the displacement frame axis) ---
    if requested_strain and results.result_strain:
        for field_name in requested_strain:
            mat = np.full((n_nodes, n_frames), np.nan)
            for t, sr in enumerate(results.result_strain):
                if t >= n_frames:
                    break
                val = sr.trimmed_field(field_name)  # edge-trim applied
                if val is not None:
                    mat[:, t] = val
            mat_dict[field_name] = mat

    out = dest_dir / f"{prefix}_results_{timestamp}.mat"
    scipy.io.savemat(str(out), mat_dict)
    return out
