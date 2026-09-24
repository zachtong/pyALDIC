"""Write a field along one line to CSV: a row per sample, a column per frame.

The same table holds every profile (one column each) and the kymograph (all
of them). Like the probe export it describes itself in a comment header --
the line, the field and its unit, the axis convention, how frame numbers map
onto the node export's files -- and writes world axes for ``disp_v``, as the
node export does. A cell with no valid measurement is left empty, never 0.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from al_dic import __version__
from al_dic.analysis.engine import Kymograph
from al_dic.analysis.probes import Probe
from al_dic.core.fields import is_strain_field
from al_dic.export.export_probes import _clean, _geometry_text


def _header_lines(
    probe: Probe,
    kymograph: Kymograph,
    field: str,
    axes: str,
    frame_rate: float | None,
    parameters: Mapping[str, str] | None,
) -> list[str]:
    unit = kymograph.unit or "dimensionless"
    measured = f"field {field}, unit {unit}"
    if field == "disp_v":
        up = "positive up" if axes == "world" else "positive down"
        measured += f", {axes} axes ({up})"
    elif is_strain_field(field):
        measured += ", as stored (world axes, y up)"
    lines = [
        f"pyALDIC {__version__} line export",
        f"{_clean(probe.label)}: line probe id {probe.id}, {_geometry_text(probe)}",
        f"    {measured}",
        "Probe coordinates are reference-frame (frame 1) image pixels, "
        "origin top-left, x = column, y = row.",
        f"distance_{kymograph.distance_unit} is reference arc length from the "
        "first endpoint; x_px, y_px locate each sample.",
        "frame is 1-based: frame_1 is the reference image; frame_N matches "
        "the node export's file frame_(N-1).",
        "An empty cell means no valid measurement at that sample and frame: "
        "off the material, consumed by a crack, or edge-trimmed strain.",
    ]
    if frame_rate:
        lines.append(f"frame_N is at time (N - 1) / {frame_rate:g} s")
    if parameters:
        lines.append("")
        lines.append("Run parameters (iDICs Good Practices Guide, ch. 6):")
        for key, value in parameters.items():
            lines.append(f"    {_clean(key)}: {_clean(value)}")
    return lines


def export_line_csv(
    path: str | Path,
    probe: Probe,
    kymograph: Kymograph,
    xy: NDArray[np.float64],
    *,
    field: str,
    axes: str,
    frame_rate: float | None = None,
    parameters: Mapping[str, str] | None = None,
) -> Path:
    """Write *kymograph* -- *field* along *probe* -- and return the path.

    *xy* holds each sample's reference position in image pixels, in the order
    of ``kymograph.distance``; *axes* is the convention the values were read
    in (``"world"`` from the application, matching the node export).
    """
    out = Path(path)
    values = np.asarray(kymograph.values, dtype=np.float64)
    positions = np.asarray(xy, dtype=np.float64)
    n_samples = values.shape[1]
    if positions.shape != (n_samples, 2) or len(kymograph.distance) != n_samples:
        raise ValueError(
            f"Sample positions {positions.shape} do not match the "
            f"{n_samples} samples along the line."
        )

    header = [f"distance_{kymograph.distance_unit}", "x_px", "y_px"]
    header += [f"frame_{int(f) + 1}" for f in kymograph.frames]

    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig like the other exports: Excel needs the BOM for CJK labels.
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        for line in _header_lines(probe, kymograph, field, axes,
                                  frame_rate, parameters):
            fh.write(f"# {line}\n" if line else "#\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for s in range(n_samples):
            row: list[object] = [
                f"{kymograph.distance[s]:.6g}",
                f"{positions[s, 0]:.3f}",
                f"{positions[s, 1]:.3f}",
            ]
            row += ["" if not np.isfinite(v) else f"{v:.9g}" for v in values[:, s]]
            writer.writerow(row)
    return out


__all__ = ["export_line_csv"]
