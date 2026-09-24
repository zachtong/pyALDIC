"""A field along one line, as CSV: a row per sample, a column per frame."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from al_dic.analysis.engine import Kymograph
from al_dic.analysis.probes import LineGeom, Probe
from al_dic.export.export_line import export_line_csv


def _probe(label="E1") -> Probe:
    return Probe(id=3, kind="line", geometry=LineGeom(10.0, 20.0, 40.0, 20.0),
                 label=label, color="#00FF00")


def _kymo(values, unit="px", distance_unit="px") -> Kymograph:
    v = np.asarray(values, dtype=float)                 # [frames, samples]
    n_frames, n = v.shape
    return Kymograph(
        frames=np.arange(n_frames, dtype=np.int64),
        distance=np.linspace(0.0, 30.0, n),
        values=v,
        status=np.zeros(v.shape, dtype=np.uint8),
        unit=unit,
        distance_unit=distance_unit,
    )


def _xy(n):
    return np.column_stack([np.linspace(10.0, 40.0, n), np.full(n, 20.0)])


def _read(path):
    with open(path, encoding="utf-8-sig") as fh:
        comments = [ln[1:].strip() for ln in fh if ln.startswith("#")]
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(ln for ln in fh if not ln.startswith("#")))
    return comments, rows[0], rows[1:]


def test_rows_are_samples_and_columns_are_frames(tmp_path):
    k = _kymo([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    p = export_line_csv(tmp_path / "l.csv", _probe(), k, _xy(3),
                        field="disp_u", axes="world")
    _, header, rows = _read(p)
    assert header == ["distance_px", "x_px", "y_px", "frame_1", "frame_2"]
    assert len(rows) == 3
    assert [float(c) for c in rows[2]] == pytest.approx([30.0, 40.0, 20.0, 0.0, 0.3])


def test_a_gap_is_an_empty_cell_not_a_zero(tmp_path):
    k = _kymo([[0.0, 0.0], [np.nan, 0.5]])
    p = export_line_csv(tmp_path / "l.csv", _probe(), k, _xy(2),
                        field="disp_u", axes="world")
    comments, _, rows = _read(p)
    assert rows[0][-1] == ""
    assert any("empty cell" in c for c in comments)


def test_the_distance_column_carries_the_length_unit(tmp_path):
    k = _kymo([[0.0, 0.0]], unit="mm", distance_unit="mm")
    p = export_line_csv(tmp_path / "l.csv", _probe(), k, _xy(2),
                        field="disp_u", axes="world")
    _, header, _ = _read(p)
    assert header[0] == "distance_mm"
    assert header[1:3] == ["x_px", "y_px"], "positions stay in image pixels"


def test_the_header_says_what_was_measured_and_where(tmp_path):
    k = _kymo([[0.0, 0.0]])
    p = export_line_csv(tmp_path / "l.csv", _probe("gauge, 1\nA"), k, _xy(2),
                        field="disp_v", axes="world", frame_rate=10.0,
                        parameters={"subset": "31 px (winsize 32)"})
    comments, _, _ = _read(p)
    text = "\n".join(comments)
    assert "gauge, 1 A" in text, "a newline in a label must not break the header"
    assert "(10.000, 20.000) -> (40.000, 20.000)" in text
    assert "disp_v" in text and "world axes" in text and "positive up" in text
    assert "frame_N matches" in text
    assert "(N - 1) / 10" in text
    assert "subset: 31 px (winsize 32)" in text


def test_strain_is_labelled_dimensionless(tmp_path):
    k = _kymo([[0.0, 0.0]], unit="")
    p = export_line_csv(tmp_path / "l.csv", _probe(), k, _xy(2),
                        field="strain_eyy", axes="world")
    comments, _, _ = _read(p)
    assert any("strain_eyy, unit dimensionless" in c for c in comments)


def test_mismatched_positions_are_refused(tmp_path):
    k = _kymo([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positions"):
        export_line_csv(tmp_path / "l.csv", _probe(), k, _xy(2),
                        field="disp_u", axes="world")
