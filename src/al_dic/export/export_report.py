"""Export pipeline results as a self-contained HTML report.

The HTML file includes:
  1. Header  — experiment metadata (prefix, timestamp, n_frames, n_nodes).
  2. DIC Parameters table — DICPara fields (scalars only).
  3. Per-field statistics table — min/max/mean/std per frame.
  4. Sample field images — base64-encoded PNGs, one per (field × sampled frame).

No external CSS/JS dependencies.  Viewable in any browser.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir

# Canonical field availability sets
_DISP_FIELDS: frozenset[str] = frozenset([
    "disp_u", "disp_v", "disp_magnitude", "disp_u_accum", "disp_v_accum",
])
_STRAIN_FIELDS: frozenset[str] = frozenset([
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
])

_CSS = """
body { font-family: Arial, sans-serif; background: #f5f5f5; color: #222; margin: 0; padding: 16px; }
h1   { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 8px; }
h2   { color: #2c5f8a; margin-top: 32px; }
h3   { color: #2c5f8a; margin-top: 24px; margin-bottom: 8px; }
table { border-collapse: collapse; margin-bottom: 24px; font-size: 13px; }
th   { background: #1a3a5c; color: #fff; padding: 6px 12px; text-align: left; }
td   { padding: 5px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #eef2f7; }
.stat-table td:first-child { font-weight: bold; }
img  { max-width: 200px; border: 1px solid #ccc; margin: 4px; vertical-align: top; }
.field-row { margin-bottom: 12px; }
.meta { font-size: 13px; color: #555; margin-bottom: 8px; }
"""


def _extract_field_values_all_frames(
    field_name: str,
    results: PipelineResult,
) -> list[NDArray | None]:
    """Return list of (N,) arrays for field_name, one per frame (None if unavailable)."""
    n_frames = len(results.result_disp)
    out: list[NDArray | None] = []

    for t, fr in enumerate(results.result_disp):
        U = fr.U_accum if fr.U_accum is not None else fr.U
        if U is None:
            out.append(None)
            continue

        u, v = split_uv(U)

        if field_name == "disp_u":
            out.append(u)
        elif field_name == "disp_v":
            out.append(v)
        elif field_name == "disp_magnitude":
            out.append(np.sqrt(u ** 2 + v ** 2))
        elif field_name == "disp_u_accum":
            if fr.U_accum is not None:
                ua, _ = split_uv(fr.U_accum)
                out.append(ua)
            else:
                out.append(None)
        elif field_name == "disp_v_accum":
            if fr.U_accum is not None:
                _, va = split_uv(fr.U_accum)
                out.append(va)
            else:
                out.append(None)
        elif field_name in _STRAIN_FIELDS:
            if results.result_strain and t < len(results.result_strain):
                val = getattr(results.result_strain[t], field_name, None)
                out.append(val)
            else:
                out.append(None)
        else:
            out.append(None)

    return out


def _render_field_image_b64(
    coords: NDArray,
    values: NDArray,
    img_shape: tuple[int, int],
    ref_image: NDArray | None,
) -> str:
    """Render field to a small PNG and return as base64 string."""
    from al_dic.export.export_png import render_field_frame
    from al_dic.gui.dialogs.export_dialog import FieldImageConfig

    cfg = FieldImageConfig(
        field_name="",
        enabled=True,
        colormap="jet",
        auto_range=True,
        vmin=0.0,
        vmax=1.0,
        bg_alpha=0.7,
    )

    # Downsample for report thumbnails (max 256px on long side)
    H, W = img_shape
    scale = min(256 / max(H, W, 1), 1.0)
    thumb_h = max(int(H * scale), 1)
    thumb_w = max(int(W * scale), 1)

    img = render_field_frame(
        coords=coords,
        values=values,
        image_shape=(thumb_h, thumb_w),
        bg_image=ref_image,
        field_cfg=cfg,
        roi_mask=None,
    )

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _build_stats_table_html(
    field_name: str,
    frame_values: list[NDArray | None],
) -> str:
    """Build an HTML <table> with per-frame stats for one field."""
    rows = ["<table>", "<tr><th>Frame</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr>"]
    for t, vals in enumerate(frame_values):
        if vals is None or not np.any(np.isfinite(vals)):
            rows.append(f"<tr><td>{t + 1}</td><td colspan='4'>N/A</td></tr>")
            continue
        finite = vals[np.isfinite(vals)]
        rows.append(
            f"<tr>"
            f"<td>{t + 1}</td>"
            f"<td>{finite.min():.4g}</td>"
            f"<td>{finite.max():.4g}</td>"
            f"<td>{finite.mean():.4g}</td>"
            f"<td>{finite.std():.4g}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _build_params_table_html(results: PipelineResult) -> str:
    """Build an HTML <table> from DICPara (scalars only)."""
    p = results.dic_para
    rows = ["<table>", "<tr><th>Parameter</th><th>Value</th></tr>"]
    for f in dataclasses.fields(p):
        v = getattr(p, f.name)
        if isinstance(v, (bool, int, float, str)):
            rows.append(f"<tr><td>{f.name}</td><td>{v}</td></tr>")
        elif isinstance(v, (list, tuple)) and len(v) <= 6:
            rows.append(f"<tr><td>{f.name}</td><td>{list(v)}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def export_html_report(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    fields: list[str],
    image_configs: list,
    sample_every: int = 5,
    ref_image: NDArray | None = None,
) -> Path:
    """Generate a self-contained HTML report.

    Args:
        dest_dir: Output directory.
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        fields: Canonical field names to include in statistics + sample images.
        image_configs: List of FieldImageConfig for sample image rendering.
                       Can be empty (uses auto-range defaults).
        sample_every: Include a sample image every N frames.
        ref_image: Optional reference image for background blending in thumbnails.

    Returns:
        Path to the written .html file.
    """
    ensure_dir(dest_dir)

    n_frames = len(results.result_disp)
    n_nodes = results.dic_mesh.coordinates_fem.shape[0]
    coords = results.dic_mesh.coordinates_fem
    img_shape = results.dic_para.img_size
    if img_shape == (0, 0):
        img_shape = (64, 64)

    # Sampled frame indices for thumbnails
    sampled_frames = list(range(0, n_frames, max(sample_every, 1)))
    if sampled_frames and sampled_frames[-1] != n_frames - 1:
        sampled_frames.append(n_frames - 1)

    # Build HTML sections
    sections: list[str] = []

    # --- Header ---
    sections.append(f"""
<h1>AL-DIC Export Report</h1>
<div class="meta">
  <b>Prefix:</b> {prefix} &nbsp;|&nbsp;
  <b>Timestamp:</b> {timestamp} &nbsp;|&nbsp;
  <b>Frames:</b> {n_frames} &nbsp;|&nbsp;
  <b>Nodes:</b> {n_nodes}
</div>
""")

    # --- DIC Parameters ---
    sections.append("<h2>DIC Parameters</h2>")
    sections.append(_build_params_table_html(results))

    # --- Per-field statistics + sample images ---
    if fields:
        sections.append("<h2>Field Statistics</h2>")
        for field_name in fields:
            frame_values = _extract_field_values_all_frames(field_name, results)
            any_valid = any(v is not None for v in frame_values)
            if not any_valid:
                continue

            sections.append(f"<h3>{field_name}</h3>")
            sections.append(_build_stats_table_html(field_name, frame_values))

            # Sample images
            if sampled_frames:
                sections.append("<div class='field-row'>")
                for t in sampled_frames:
                    vals = frame_values[t] if t < len(frame_values) else None
                    if vals is None or not np.any(np.isfinite(vals)):
                        continue
                    b64 = _render_field_image_b64(coords, vals, img_shape, ref_image)
                    if b64:
                        sections.append(
                            f'<img src="data:image/png;base64,{b64}" '
                            f'title="Frame {t + 1}" alt="Frame {t + 1}">'
                        )
                sections.append("</div>")

    # Assemble full page
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AL-DIC Report — {prefix}</title>
<style>{_CSS}</style>
</head>
<body>
{"".join(sections)}
</body>
</html>
"""

    out = dest_dir / f"{prefix}_report_{timestamp}.html"
    out.write_text(html, encoding="utf-8")
    return out
