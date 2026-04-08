"""Export pipeline results to per-frame CSV files.

Each frame produces one CSV file inside a dedicated subdirectory.
Files use UTF-8 with BOM (``utf-8-sig``) for Excel compatibility.

Column layout:
    node_id, x_px, y_px
    [disp_u, disp_v, disp_magnitude, disp_u_accum, disp_v_accum]  if include_disp
    [strain_exx, strain_eyy, strain_exy, ...]                      if include_strain and available
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from staq_dic.core.data_structures import PipelineResult, split_uv
from staq_dic.export.export_utils import ensure_dir, frame_tag

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


def export_csv(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    include_disp: bool,
    include_strain: bool,
) -> list[Path]:
    """Write one CSV file per frame into a dedicated subdirectory.

    Args:
        dest_dir: Parent output directory.
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        include_disp: Include displacement columns.
        include_strain: Include strain columns (skipped if no strain computed).

    Returns:
        Sorted list of Paths to the written CSV files.
    """
    csv_dir = dest_dir / f"{prefix}_csv_{timestamp}"
    ensure_dir(csv_dir)

    coords = results.dic_mesh.coordinates_fem   # (N, 2) — x, y
    n_frames = len(results.result_disp)
    has_strain = include_strain and bool(results.result_strain)

    paths: list[Path] = []
    for t in range(n_frames):
        tag = frame_tag(t, n_frames)
        out = csv_dir / f"{tag}.csv"

        fr = results.result_disp[t]
        u, v = split_uv(fr.U)
        mag = np.sqrt(u ** 2 + v ** 2)
        has_accum = fr.U_accum is not None
        if has_accum:
            ua, va = split_uv(fr.U_accum)  # type: ignore[arg-type]

        sr = results.result_strain[t] if has_strain and t < len(results.result_strain) else None

        # Build header
        fieldnames = ["node_id", "x_px", "y_px"]
        if include_disp:
            fieldnames += ["disp_u", "disp_v", "disp_magnitude"]
            if has_accum:
                fieldnames += ["disp_u_accum", "disp_v_accum"]
        if sr is not None:
            for fname in _STRAIN_FIELDS:
                val = getattr(sr, fname, None)
                if val is not None:
                    fieldnames.append(fname)

        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            n_nodes = coords.shape[0]
            for i in range(n_nodes):
                row: dict = {
                    "node_id": i,
                    "x_px": coords[i, 0],
                    "y_px": coords[i, 1],
                }
                if include_disp:
                    row["disp_u"] = u[i]
                    row["disp_v"] = v[i]
                    row["disp_magnitude"] = mag[i]
                    if has_accum:
                        row["disp_u_accum"] = ua[i]
                        row["disp_v_accum"] = va[i]
                if sr is not None:
                    for fname in _STRAIN_FIELDS:
                        val = getattr(sr, fname, None)
                        if val is not None:
                            row[fname] = val[i]
                writer.writerow(row)

        paths.append(out)

    return paths
