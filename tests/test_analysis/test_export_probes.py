"""Probe time-series CSV."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from al_dic.analysis.probes import LineGeom, PointGeom, Probe
from al_dic.analysis.series import FrameStatus, TimeSeries
from al_dic.export.export_probes import (
    ProbeSeries,
    export_probe_csv,
    run_parameters,
)


def _series(values, unit="px", status_at=None, axes="image") -> TimeSeries:
    v = np.asarray(values, dtype=float)
    status = [FrameStatus.OK if np.isfinite(x) else FrameStatus.NO_DATA for x in v]
    for i, st in (status_at or {}).items():
        status[i] = st
    return TimeSeries(
        frames=np.arange(len(v), dtype=np.int64),
        values=v,
        valid_fraction=np.where(np.isfinite(v), 1.0, 0.0),
        status=status,
        unit=unit,
        axes=axes,
    )


def _entry(label="p1", field="disp_u", reduction="value", values=(0.0, 1.0, 2.0),
           probe_id=1, **kw) -> ProbeSeries:
    return ProbeSeries(
        probe=Probe(id=probe_id, kind="point", geometry=PointGeom(10.0, 20.0),
                    label=label, color="#FF0000"),
        field=field,
        reduction=reduction,
        series=_series(values, **kw),
    )


def _read(path):
    with open(path, encoding="utf-8-sig") as fh:
        comments = [ln[1:].strip() for ln in fh if ln.startswith("#")]
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(ln for ln in fh if not ln.startswith("#")))
    return comments, rows[0], rows[1:]


def test_one_row_per_frame_and_frames_are_one_based(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()])
    _, header, rows = _read(p)
    assert header[0] == "frame"
    assert [r[0] for r in rows] == ["1", "2", "3"]


def test_value_column_is_named_for_probe_field_and_reduction(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()])
    _, header, _ = _read(p)
    assert "p1_disp_u_value" in header


def test_a_gauge_column_is_not_named_after_a_field_it_does_not_read(tmp_path):
    entry = ProbeSeries(
        probe=Probe(id=2, kind="line", geometry=LineGeom(0.0, 0.0, 30.0, 40.0),
                    label="E1", color="#00FF00"),
        field=None, reduction="strain", series=_series((0.0, 0.01), unit=""),
    )
    p = export_probe_csv(tmp_path / "probes.csv", [entry])
    comments, header, _ = _read(p)
    assert "E1_strain" in header
    assert "gauge strain" in "\n".join(comments)


def test_quality_columns_travel_with_the_value(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()])
    _, header, _ = _read(p)
    assert "p1_disp_u_value_valid_fraction" in header
    assert "p1_disp_u_value_flag" in header


def test_quality_columns_can_be_turned_off(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()],
                         include_quality=False)
    _, header, _ = _read(p)
    assert header == ["frame", "p1_disp_u_value"]


def test_missing_values_are_empty_cells_not_the_word_nan(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv",
                         [_entry(values=(0.0, np.nan, 2.0))])
    _, header, rows = _read(p)
    col = header.index("p1_disp_u_value")
    assert rows[1][col] == ""
    assert all("nan" not in cell.lower() for row in rows for cell in row)


def test_flag_column_says_why(tmp_path):
    p = export_probe_csv(
        tmp_path / "probes.csv",
        [_entry(values=(0.0, np.nan, 2.0),
                status_at={1: FrameStatus.BELOW_THRESHOLD, 2: FrameStatus.CRACK})],
    )
    _, header, rows = _read(p)
    flag = header.index("p1_disp_u_value_flag")
    assert [r[flag] for r in rows] == ["ok", "below_threshold", "crack"]
    value = header.index("p1_disp_u_value")
    assert rows[2][value] == "2", "a crack frame can still carry a value"


def test_header_records_geometry_units_frames_and_version(tmp_path):
    entry = ProbeSeries(
        probe=Probe(id=3, kind="line", geometry=LineGeom(0.0, 0.0, 30.0, 40.0),
                    label="gauge", color="#00FF00"),
        field=None, reduction="strain", series=_series((0.0, 0.01), unit=""),
    )
    p = export_probe_csv(tmp_path / "probes.csv", [entry])
    blob = "\n".join(_read(p)[0])
    assert "pyALDIC" in blob
    assert "line (0.000, 0.000) -> (30.000, 40.000) px" in blob
    assert "length 50.000 px" in blob
    assert "dimensionless" in blob
    assert "frame_(N-1)" in blob, "which node file a row corresponds to"


def test_each_disp_v_column_states_its_axes(tmp_path):
    """The screen shows v positive down; the node export writes it positive up."""
    world = _entry(label="W", field="disp_v", reduction="mean", axes="world")
    image = _entry(label="I", field="disp_v", reduction="mean", axes="image",
                   probe_id=2)
    blob = "\n".join(_read(export_probe_csv(tmp_path / "p.csv",
                                            [world, image]))[0])
    assert "world axes (positive up)" in blob
    assert "image axes (positive down)" in blob


def test_run_parameters_go_in_the_header(tmp_path):
    params = run_parameters(SimpleNamespace(
        dic_para=SimpleNamespace(winsize=32, winstepsize=16)))
    assert params["subset"].startswith("31 px")
    assert params["step"] == "16 px"
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()],
                         parameters={**params, "strain window": "41 px"})
    blob = "\n".join(_read(p)[0])
    assert "Good Practices" in blob
    assert "subset: 31 px (winsize 32)" in blob
    assert "strain window: 41 px" in blob


def test_time_column_appears_only_with_a_frame_rate(tmp_path):
    p = export_probe_csv(tmp_path / "a.csv", [_entry()])
    assert "time_s" not in _read(p)[1]
    p = export_probe_csv(tmp_path / "b.csv", [_entry()], frame_rate=2.0)
    _, header, rows = _read(p)
    col = header.index("time_s")
    assert [r[col] for r in rows] == ["0", "0.5", "1"]


def test_several_probes_share_one_file(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv",
                         [_entry(label="a"), _entry(label="b", probe_id=2)])
    _, header, _ = _read(p)
    assert "a_disp_u_value" in header and "b_disp_u_value" in header


def test_duplicate_labels_are_disambiguated_by_id(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv",
                         [_entry(label="x"), _entry(label="x", probe_id=7)])
    _, header, _ = _read(p)
    assert "x_disp_u_value_id1" in header and "x_disp_u_value_id7" in header


def test_series_of_different_lengths_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="same frames"):
        export_probe_csv(tmp_path / "p.csv",
                         [_entry(values=(0.0, 1.0)), _entry(values=(0.0,))])


def test_empty_export_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Nothing to export"):
        export_probe_csv(tmp_path / "p.csv", [])


def test_a_newline_in_a_label_cannot_break_the_header(tmp_path):
    p = export_probe_csv(tmp_path / "p.csv", [_entry(label="left\nedge")])
    _, header, rows = _read(p)
    assert "left edge_disp_u_value" in header
    assert len(rows) == 3


def test_the_file_opens_in_excel_with_cjk_labels(tmp_path):
    """utf-8-sig like the node export; without the BOM Excel garbles CJK."""
    p = export_probe_csv(tmp_path / "p.csv", [_entry(label="试样中心")],
                         include_quality=False)
    with open(p, "rb") as fh:
        assert fh.read(3) == b"\xef\xbb\xbf"
    assert _read(p)[1][1].startswith("试样中心")
