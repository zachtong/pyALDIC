"""Export pipeline results to NumPy archive (.npz) format.

Single-file mode (default):
    All frames merged into N×T matrices.  Arrays keyed by field name.

Per-frame mode (``per_frame=True``):
    One .npz per frame; arrays are 1-D (N,).  Filenames include frame tag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir, frame_tag

# Canonical displacement field names (excluding coordinates)
_DISP_FIELDS: frozenset[str] = frozenset([
    "disp_u", "disp_v", "disp_magnitude",
])

# Canonical strain field names
_STRAIN_FIELDS: frozenset[str] = frozenset([
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
])


def _collect_all_arrays(
    results: PipelineResult,
    fields: list[str],
) -> dict[str, NDArray]:
    """Collect requested fields from all frames into N×T matrices.

    Args:
        results: Full pipeline results.
        fields: List of canonical field names to include.

    Returns:
        Dict mapping field name to (N, T) array. Always includes 'coordinates'.
    """
    n_frames = len(results.result_disp)
    n_nodes = results.dic_mesh.coordinates_fem.shape[0]
    arrays: dict[str, NDArray] = {
        "coordinates": results.dic_mesh.coordinates_fem,
    }

    if n_frames == 0:
        return arrays

    fields_set = set(fields)
    requested_disp = fields_set & _DISP_FIELDS
    requested_strain = fields_set & _STRAIN_FIELDS

    # --- Displacement fields ---
    # Always export accumulated displacement (U_accum) when available,
    # falling back to incremental U otherwise.  Users see U/V as total
    # displacement — the incremental vs. accumulated distinction is hidden.
    if requested_disp:
        u_mat = np.full((n_nodes, n_frames), np.nan)
        v_mat = np.full((n_nodes, n_frames), np.nan)

        for t, fr in enumerate(results.result_disp):
            U = fr.U_accum if fr.U_accum is not None else fr.U
            u, v = split_uv(U)
            u_mat[:, t] = u
            v_mat[:, t] = v

        if "disp_u" in requested_disp:
            arrays["disp_u"] = u_mat
        if "disp_v" in requested_disp:
            arrays["disp_v"] = v_mat
        if "disp_magnitude" in requested_disp:
            arrays["disp_magnitude"] = np.sqrt(u_mat ** 2 + v_mat ** 2)

    # --- Strain fields ---
    if requested_strain and results.result_strain:
        n_strain_frames = len(results.result_strain)
        for field_name in requested_strain:
            mat = np.full((n_nodes, n_strain_frames), np.nan)
            for t, sr in enumerate(results.result_strain):
                val = getattr(sr, field_name, None)
                if val is not None:
                    mat[:, t] = val
            arrays[field_name] = mat

    return arrays


def export_npz(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    fields: list[str],
    per_frame: bool = False,
) -> "Path | list[Path]":
    """Write selected fields to .npz archive(s).

    Args:
        dest_dir: Output directory (created if absent).
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        fields: List of canonical field names to export (e.g. ["disp_u", "strain_exx"]).
                'coordinates' is always exported regardless of this list.
                Unavailable fields (e.g. strain when not computed) are silently skipped.
        per_frame: If True, write one file per frame; if False, single N×T file.

    Returns:
        Path to single archive, or list of Paths for per-frame mode.
    """
    ensure_dir(dest_dir)
    all_arrays = _collect_all_arrays(results, fields)

    if per_frame:
        return _write_per_frame(dest_dir, prefix, timestamp, results, all_arrays)

    out = dest_dir / f"{prefix}_results_{timestamp}.npz"
    np.savez_compressed(str(out), **all_arrays)
    return out


def _write_per_frame(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    all_arrays: dict[str, NDArray],
) -> list[Path]:
    """Write one .npz per frame, extracting the t-th column from each matrix."""
    n_frames = len(results.result_disp)
    paths: list[Path] = []

    for t in range(n_frames):
        tag = frame_tag(t, n_frames)
        frame_data: dict[str, NDArray] = {}

        for key, mat in all_arrays.items():
            if key == "coordinates":
                frame_data[key] = mat
            else:
                frame_data[key] = mat[:, t] if mat.ndim == 2 else mat

        out = dest_dir / f"{prefix}_results_{timestamp}_{tag}.npz"
        np.savez_compressed(str(out), **frame_data)
        paths.append(out)

    return paths
