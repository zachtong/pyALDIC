"""Write probe time series to CSV.

One row per frame. That is a different table from ``export_csv``, which writes
one row per mesh node, so the two do not interact.

The file is self-describing: a comment header records each probe's geometry,
what every column holds and in what unit, the axis convention, how its frame
numbers map onto the node export's files, the run parameters the iDICs Good
Practices Guide asks a report to state, and the pyALDIC version. Quality
columns travel with the data by default: a value alone cannot distinguish a
mean over two hundred valid points from a mean over three.

Axis convention. The screen shows displacement in image axes (v positive
down). The node export writes world axes (v positive up) so that displacement
and strain agree inside one file -- strain components are world-axis
quantities, and mixing the two gave v and the shear components opposite signs.
Each series carries the convention it was computed in (``TimeSeries.axes``) and
the header states it for every ``disp_v`` column; nothing is flipped here,
because negating a reduced value is wrong for a maximum, a minimum or a
standard deviation. The application exports in world axes, to match the node
file for the same point and frame. Crack sliding is defined without reference
to either axis (positive when the second endpoint moves to the right of the
gauge direction, as seen on screen), so it reads the same everywhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from al_dic import __version__
from al_dic.analysis.probes import (
    GAUGE_QUANTITIES,
    AreaGeom,
    LineGeom,
    PointGeom,
    Probe,
)
from al_dic.analysis.series import TimeSeries

#: Written for a frame that holds no value, so a reader is never asked to guess
#: whether an empty cell means zero.
_EMPTY = ""


@dataclass(frozen=True)
class ProbeSeries:
    """One curve, with enough context to name and describe its columns.

    ``field`` is None for a gauge quantity, which does not read a field.
    """

    probe: Probe
    field: str | None
    reduction: str
    series: TimeSeries


def run_parameters(result) -> dict[str, str]:
    """The DIC parameters a report should state, where the run records them.

    Only what ``result.dic_para`` holds reliably: subset and step. Strain
    settings are chosen in the strain window and are not written back to the
    result, so reading them from here would report stale numbers; a caller
    that knows the settings of the last Compute Strain adds them itself.
    """
    para = getattr(result, "dic_para", None)
    out = {"software": f"pyALDIC {__version__}"}
    winsize = getattr(para, "winsize", None)
    step = getattr(para, "winstepsize", None)
    if winsize:
        # The interface shows the odd subset (31) for the even API size (32).
        out["subset"] = f"{int(winsize) - 1} px (winsize {int(winsize)})"
    if step:
        out["step"] = f"{int(step)} px"
    return out


def _clean(text: str) -> str:
    """One line, no control characters: a newline in a label broke the header."""
    return " ".join(str(text).split())


def _geometry_text(probe: Probe) -> str:
    g = probe.geometry
    if isinstance(g, PointGeom):
        return f"point at ({g.x:.3f}, {g.y:.3f}) px"
    if isinstance(g, LineGeom):
        return (
            f"line ({g.x0:.3f}, {g.y0:.3f}) -> ({g.x1:.3f}, {g.y1:.3f}) px, "
            f"length {g.length():.3f} px"
        )
    if isinstance(g, AreaGeom):
        if g.shape == "rect":
            x0, y0, x1, y1 = g.data  # type: ignore[misc]
            return f"rect ({x0:.3f}, {y0:.3f}) -> ({x1:.3f}, {y1:.3f}) px"
        if g.shape == "circle":
            cx, cy, r = g.data  # type: ignore[misc]
            return f"circle centre ({cx:.3f}, {cy:.3f}) px, radius {r:.3f} px"
        pts = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in g.data)  # type: ignore[misc]
        return f"polygon [{pts}] px"
    return "unknown geometry"


def _stem(entry: ProbeSeries) -> str:
    label = _clean(entry.probe.label)
    if entry.reduction in GAUGE_QUANTITIES or entry.field is None:
        # A gauge reads no field; naming one ("P1_strain_eyy_strain") was a lie.
        return f"{label}_{entry.reduction}"
    return f"{label}_{entry.field}_{entry.reduction}"


def _column_stems(entries: Sequence[ProbeSeries]) -> list[str]:
    """Column prefixes, disambiguated when two probes share a label."""
    stems = [_stem(e) for e in entries]
    seen: dict[str, int] = {}
    for stem in stems:
        seen[stem] = seen.get(stem, 0) + 1
    return [
        f"{stem}_id{entry.probe.id}" if seen[stem] > 1 else stem
        for entry, stem in zip(entries, stems)
    ]


def _header_lines(
    entries: Sequence[ProbeSeries],
    stems: Sequence[str],
    frame_rate: float | None,
    parameters: Mapping[str, str] | None,
    notes: Sequence[str] = (),
) -> list[str]:
    lines = [
        f"pyALDIC {__version__} probe export",
        "Probe coordinates are reference-frame (frame 1) image pixels, "
        "origin top-left, x = column, y = row.",
        "frame is 1-based: frame 1 is the reference image; frame N matches "
        "the node export's file frame_(N-1).",
        "Strain is written as stored (world axes, y up). Each disp_v column "
        "states its axes below: world = positive up, as in the node export; "
        "image = positive down, as on screen. Crack sliding is positive when "
        "the second endpoint moves to the right of the gauge direction as "
        "seen on screen.",
        "An empty cell means no valid measurement; the matching _flag "
        "column says why.",
    ]
    if frame_rate:
        lines.append(f"time_s = (frame - 1) / {frame_rate:g}")
    lines.extend(_clean(note) if not note.startswith(" ") else note.rstrip()
                 for note in notes)
    if parameters:
        lines.append("")
        lines.append("Run parameters (iDICs Good Practices Guide, ch. 6):")
        for key, value in parameters.items():
            lines.append(f"    {_clean(key)}: {_clean(value)}")
    lines.append("")
    for entry, stem in zip(entries, stems):
        lines.append(
            f"{stem}: {entry.probe.kind} probe id {entry.probe.id} "
            f"'{_clean(entry.probe.label)}', {_geometry_text(entry.probe)}"
        )
        unit = entry.series.unit or "dimensionless"
        what = (f"gauge {entry.reduction}" if entry.field is None
                else f"field {entry.field}, reduction {entry.reduction}")
        axes = ""
        if entry.field == "disp_v" and entry.reduction != "valid_fraction":
            up = "positive up" if entry.series.axes == "world" else "positive down"
            axes = f", {entry.series.axes} axes ({up})"
        lines.append(f"    {what}, unit {unit}{axes}")
    return lines


def export_probe_csv(
    path: str | Path,
    entries: Sequence[ProbeSeries],
    *,
    frame_rate: float | None = None,
    include_quality: bool = True,
    parameters: Mapping[str, str] | None = None,
    extra_columns: Mapping[str, np.ndarray] | None = None,
    notes: Sequence[str] = (),
) -> Path:
    """Write *entries* as one table and return the path written.

    Parameters
    ----------
    frame_rate:
        Frames per second. When given, a ``time_s`` column is added.
    include_quality:
        Add ``_valid_fraction`` and ``_flag`` columns beside every value.
    parameters:
        Run parameters for the header, e.g. from ``run_parameters(result)``
        plus the strain settings of the last Compute Strain.
    extra_columns:
        Per-frame values written after ``frame`` (and ``time_s``), indexed by
        frame -- a machine's load and stress. Empty where a frame has none.
    notes:
        Header lines describing the extra columns.
    """
    out = Path(path)
    if not entries:
        raise ValueError("Nothing to export: no probe series were given.")

    lengths = {len(e.series.frames) for e in entries}
    if len(lengths) > 1:
        raise ValueError(
            f"All series must cover the same frames, got lengths {sorted(lengths)}."
        )

    stems = _column_stems(entries)
    frames = entries[0].series.frames

    extra = dict(extra_columns or {})
    header = ["frame"]
    if frame_rate:
        header.append("time_s")
    header.extend(extra)
    for stem in stems:
        header.append(stem)
        if include_quality:
            header.append(f"{stem}_valid_fraction")
            header.append(f"{stem}_flag")

    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig like the node export: without the BOM, Excel reads the file
    # in the system code page and CJK probe labels arrive garbled.
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        for line in _header_lines(entries, stems, frame_rate, parameters, notes):
            fh.write(f"# {line}\n" if line else "#\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, frame in enumerate(frames):
            row: list[object] = [int(frame) + 1]      # 1-based in the file
            if frame_rate:
                row.append(f"{int(frame) / frame_rate:.6g}")
            for values in extra.values():
                value = values[int(frame)] if int(frame) < len(values) else np.nan
                row.append(_EMPTY if not np.isfinite(value) else f"{value:.9g}")
            for entry in entries:
                value = entry.series.values[i]
                row.append(_EMPTY if not np.isfinite(value) else f"{value:.9g}")
                if include_quality:
                    row.append(f"{entry.series.valid_fraction[i]:.4g}")
                    row.append(entry.series.status[i].value)
            writer.writerow(row)
    return out


__all__ = ["ProbeSeries", "export_probe_csv", "run_parameters"]
