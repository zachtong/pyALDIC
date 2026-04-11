"""Export pipeline results to per-frame CSV files.

Each frame produces one CSV file inside a dedicated subdirectory.
Files use UTF-8 with BOM (``utf-8-sig``) for Excel compatibility.

Column layout:
    node_id, x_px, y_px
    [disp_u, disp_v, disp_magnitude]       (if requested; always accumulated)
    [strain_exx, strain_eyy, strain_exy, ...]  (if requested & available)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir, frame_tag

# Canonical field name sets
_DISP_FIELDS: frozenset[str] = frozenset([
    "disp_u", "disp_v", "disp_magnitude",
])

_STRAIN_FIELDS: frozenset[str] = frozenset([
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
])

# Ordered column sequences for deterministic output
_DISP_ORDER = ["disp_u", "disp_v", "disp_magnitude"]
_STRAIN_ORDER = [
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
]


def export_csv(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    fields: list[str],
) -> list[Path]:
    """Write one CSV file per frame into a dedicated subdirectory.

    Args:
        dest_dir: Parent output directory.
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        fields: List of canonical field names to export as columns.
                'node_id', 'x_px', 'y_px' are always present.
                Unavailable strain fields are silently skipped.

    Returns:
        Sorted list of Paths to the written CSV files.
    """
    csv_dir = dest_dir / f"{prefix}_csv_{timestamp}"
    ensure_dir(csv_dir)

    coords = results.dic_mesh.coordinates_fem   # (N, 2) — x, y
    n_frames = len(results.result_disp)

    fields_set = set(fields)
    requested_disp = [f for f in _DISP_ORDER if f in fields_set]
    requested_strain = [f for f in _STRAIN_ORDER if f in fields_set]

    paths: list[Path] = []
    for t in range(n_frames):
        tag = frame_tag(t, n_frames)
        out = csv_dir / f"{tag}.csv"

        fr = results.result_disp[t]
        # Always use accumulated displacement when available.
        U = fr.U_accum if fr.U_accum is not None else fr.U
        u, v = split_uv(U)
        mag = np.sqrt(u ** 2 + v ** 2)

        # Strain result for this frame (if available and requested)
        sr = None
        if requested_strain and results.result_strain and t < len(results.result_strain):
            sr = results.result_strain[t]

        # Build header — include only requested + available columns
        header = ["node_id", "x_px", "y_px"]
        disp_cols = list(requested_disp)
        for fname in disp_cols:
            header.append(fname)

        strain_cols = []
        if sr is not None:
            for fname in requested_strain:
                if getattr(sr, fname, None) is not None:
                    strain_cols.append(fname)
                    header.append(fname)

        # Pre-build displacement value lookup for speed
        disp_values: dict[str, "np.ndarray"] = {
            "disp_u": u, "disp_v": v, "disp_magnitude": mag,
        }

        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            n_nodes = coords.shape[0]
            for i in range(n_nodes):
                row: dict = {
                    "node_id": i,
                    "x_px": coords[i, 0],
                    "y_px": coords[i, 1],
                }
                for fname in disp_cols:
                    row[fname] = disp_values[fname][i]
                for fname in strain_cols:
                    row[fname] = getattr(sr, fname)[i]  # type: ignore[index]
                writer.writerow(row)

        paths.append(out)

    return paths
