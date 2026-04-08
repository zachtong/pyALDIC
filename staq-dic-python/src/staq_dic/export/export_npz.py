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

from staq_dic.core.data_structures import PipelineResult, split_uv
from staq_dic.export.export_utils import ensure_dir, frame_tag

# Strain fields to export (in order)
_STRAIN_FIELDS = (
    "strain_exx",
    "strain_eyy",
    "strain_exy",
    "strain_principal_max",
    "strain_principal_min",
    "strain_maxshear",
    "strain_von_mises",
    "strain_rotation",
)


def _collect_disp_arrays(
    results: PipelineResult,
) -> dict[str, NDArray]:
    """Collect displacement fields from all frames into N×T matrices."""
    n_frames = len(results.result_disp)
    if n_frames == 0:
        return {}

    n_nodes = results.dic_mesh.coordinates_fem.shape[0]
    u_mat = np.full((n_nodes, n_frames), np.nan)
    v_mat = np.full((n_nodes, n_frames), np.nan)
    u_accum_mat = np.full((n_nodes, n_frames), np.nan)
    v_accum_mat = np.full((n_nodes, n_frames), np.nan)
    has_accum = False

    for t, fr in enumerate(results.result_disp):
        u, v = split_uv(fr.U)
        u_mat[:, t] = u
        v_mat[:, t] = v
        if fr.U_accum is not None:
            ua, va = split_uv(fr.U_accum)
            u_accum_mat[:, t] = ua
            v_accum_mat[:, t] = va
            has_accum = True

    arrays: dict[str, NDArray] = {
        "coordinates": results.dic_mesh.coordinates_fem,
        "disp_u": u_mat,
        "disp_v": v_mat,
        "disp_magnitude": np.sqrt(u_mat ** 2 + v_mat ** 2),
    }
    if has_accum:
        arrays["disp_u_accum"] = u_accum_mat
        arrays["disp_v_accum"] = v_accum_mat
    return arrays


def _collect_strain_arrays(
    results: PipelineResult,
) -> dict[str, NDArray]:
    """Collect strain fields from all frames into N×T matrices."""
    if not results.result_strain:
        return {}

    n_frames = len(results.result_strain)
    n_nodes = results.dic_mesh.coordinates_fem.shape[0]
    arrays: dict[str, NDArray] = {}

    for field_name in _STRAIN_FIELDS:
        mat = np.full((n_nodes, n_frames), np.nan)
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
    include_disp: bool,
    include_strain: bool,
    per_frame: bool,
) -> Path | list[Path]:
    """Write displacement and/or strain results to .npz archive(s).

    Args:
        dest_dir: Output directory (created if absent).
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        include_disp: Whether to export displacement fields.
        include_strain: Whether to export strain fields (skipped if no strain).
        per_frame: If True, write one file per frame; if False, single N×T file.

    Returns:
        Path to single archive, or list of Paths for per-frame mode.
    """
    ensure_dir(dest_dir)

    disp_arrays = _collect_disp_arrays(results) if include_disp else {}
    strain_arrays = (
        _collect_strain_arrays(results)
        if (include_strain and results.result_strain)
        else {}
    )

    if per_frame:
        return _write_per_frame(
            dest_dir, prefix, timestamp, results, disp_arrays, strain_arrays
        )

    # Single file: N×T matrices
    all_arrays = {}
    all_arrays.update(disp_arrays)
    all_arrays.update(strain_arrays)

    out = dest_dir / f"{prefix}_results_{timestamp}.npz"
    np.savez_compressed(str(out), **all_arrays)
    return out


def _write_per_frame(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    disp_arrays: dict[str, NDArray],
    strain_arrays: dict[str, NDArray],
) -> list[Path]:
    """Write one .npz per frame, extracting the t-th column from each matrix."""
    n_frames = len(results.result_disp)
    paths: list[Path] = []

    for t in range(n_frames):
        tag = frame_tag(t, n_frames)
        frame_data: dict[str, NDArray] = {}

        for key, mat in disp_arrays.items():
            if key == "coordinates":
                frame_data[key] = mat
            else:
                frame_data[key] = mat[:, t] if mat.ndim == 2 else mat

        for key, mat in strain_arrays.items():
            frame_data[key] = mat[:, t] if mat.ndim == 2 else mat

        out = dest_dir / f"{prefix}_results_{timestamp}_{tag}.npz"
        np.savez_compressed(str(out), **frame_data)
        paths.append(out)

    return paths
