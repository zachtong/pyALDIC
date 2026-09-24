"""A testing machine's record, read from CSV and matched to the frames."""

from __future__ import annotations

import numpy as np
import pytest

from al_dic.analysis.load_data import (
    LoadData,
    LoadSync,
    LoadTable,
    parse_load_table,
    read_load_table,
)

# --- reading ---------------------------------------------------------------------

BLUEHILL = '''"Results Table 1"
"Time","Extension","Load"
"(s)","(mm)","(kN)"
"0.000","0.0000","0.0012"
"0.500","0.0105","0.8500"
"1.000","0.0210","1.7000"
'''


def test_names_take_the_units_row_below_them():
    t = parse_load_table(BLUEHILL)
    assert t.names == ("Time (s)", "Extension (mm)", "Load (kN)")
    np.testing.assert_allclose(t.column("Load (kN)"), [0.0012, 0.85, 1.7])


def test_a_european_file_uses_semicolons_and_decimal_commas():
    text = "Zeit;Kraft\ns;N\n0,0;1,5\n0,5;2,5\n"
    t = parse_load_table(text)
    assert t.names == ("Zeit (s)", "Kraft (N)")
    np.testing.assert_allclose(t.column("Kraft (N)"), [1.5, 2.5])


def test_a_tab_separated_file_without_a_header_gets_column_numbers():
    t = parse_load_table("1\t10.0\n2\t20.0\n3\t30.0\n")
    assert t.names == ("Column 1", "Column 2")
    np.testing.assert_allclose(t.column("Column 2"), [10.0, 20.0, 30.0])


def test_a_cell_that_is_not_a_number_is_nan_not_a_crash():
    t = parse_load_table("frame,load\n1,5.0\n2,n/a\n3,7.0\n")
    assert np.isnan(t.column("load")[1])


def test_duplicate_and_empty_names_are_made_distinct():
    t = parse_load_table("load,,load\n1,2,3\n")
    assert t.names == ("load", "Column 2", "load (2)")


def test_a_file_with_no_numbers_is_refused():
    with pytest.raises(ValueError, match="number"):
        parse_load_table("a,b\nc,d\n")


def test_reading_a_file_with_a_bom(tmp_path):
    path = tmp_path / "machine.csv"
    path.write_text("frame,load\n1,5.0\n", encoding="utf-8-sig")
    t = read_load_table(path)
    assert t.names == ("frame", "load")
    assert t.source == "machine.csv"


# --- matching to frames ----------------------------------------------------------

def _table(**columns) -> LoadTable:
    return LoadTable(names=tuple(columns),
                     columns=tuple(np.asarray(v, dtype=float) for v in columns.values()))


def test_by_frame_number_each_frame_takes_its_row():
    data = LoadData(_table(frame=[1, 2, 3, 4], load=[0.0, 10.0, 20.0, 30.0]),
                    LoadSync(mode="frame", load_column="load", frame_column="frame"))
    np.testing.assert_allclose(data.load_n(5, frame_rate=0.0),
                               [0.0, 10.0, 20.0, 30.0, np.nan])


def test_a_zero_based_frame_column_is_shifted():
    data = LoadData(_table(idx=[0, 1, 2], load=[1.0, 2.0, 3.0]),
                    LoadSync(mode="frame", load_column="load", frame_column="idx",
                             frame_base=0))
    np.testing.assert_allclose(data.load_n(3, frame_rate=0.0), [1.0, 2.0, 3.0])


def test_repeated_rows_for_a_frame_are_averaged():
    data = LoadData(_table(frame=[1, 1, 2], load=[1.0, 3.0, 5.0]),
                    LoadSync(mode="frame", load_column="load", frame_column="frame"))
    np.testing.assert_allclose(data.load_n(2, frame_rate=0.0), [2.0, 5.0])


def test_by_time_the_load_is_interpolated_at_each_frame():
    """Camera at 2 fps started 1 s into the machine's record."""
    data = LoadData(_table(t=[0.0, 1.0, 2.0, 3.0], load=[0.0, 100.0, 200.0, 300.0]),
                    LoadSync(mode="time", load_column="load", time_column="t",
                             offset_s=1.0))
    np.testing.assert_allclose(data.load_n(4, frame_rate=2.0),
                               [100.0, 150.0, 200.0, 250.0])


def test_frames_outside_the_record_have_no_load():
    data = LoadData(_table(t=[0.0, 1.0], load=[0.0, 10.0]),
                    LoadSync(mode="time", load_column="load", time_column="t"))
    load = data.load_n(4, frame_rate=1.0)
    assert load[1] == pytest.approx(10.0)
    assert np.isnan(load[2]) and np.isnan(load[3])


def test_time_sync_needs_a_frame_rate():
    data = LoadData(_table(t=[0.0, 1.0], load=[0.0, 10.0]),
                    LoadSync(mode="time", load_column="load", time_column="t"))
    with pytest.raises(ValueError, match="frame rate"):
        data.load_n(2, frame_rate=0.0)


def test_kilonewtons_become_newtons():
    data = LoadData(_table(frame=[1], load=[1.5]),
                    LoadSync(mode="frame", load_column="load", frame_column="frame",
                             load_unit="kN"))
    assert data.load_n(1, frame_rate=0.0)[0] == pytest.approx(1500.0)


def test_engineering_stress_is_load_over_the_initial_area():
    data = LoadData(_table(frame=[1, 2], load=[0.0, 500.0]),
                    LoadSync(mode="frame", load_column="load", frame_column="frame"),
                    area_mm2=20.0)
    np.testing.assert_allclose(data.stress_mpa(2, frame_rate=0.0), [0.0, 25.0])


def test_there_is_no_stress_without_an_area():
    data = LoadData(_table(frame=[1], load=[1.0]),
                    LoadSync(mode="frame", load_column="load", frame_column="frame"))
    assert data.stress_mpa(1, frame_rate=0.0) is None


def test_a_mapping_to_a_missing_column_is_refused():
    with pytest.raises(ValueError, match="nope"):
        LoadData(_table(frame=[1], load=[1.0]),
                 LoadSync(mode="frame", load_column="nope", frame_column="frame"))


def test_a_non_positive_area_is_refused():
    with pytest.raises(ValueError, match="area"):
        LoadData(_table(frame=[1], load=[1.0]),
                 LoadSync(mode="frame", load_column="load", frame_column="frame"),
                 area_mm2=0.0)


# --- persistence ---------------------------------------------------------------

def test_a_session_keeps_the_columns_in_use_and_the_mapping():
    data = LoadData(_table(t=[0.0, 1.0], ext=[0.0, 0.1], load=[0.0, np.nan]),
                    LoadSync(mode="time", load_column="load", time_column="t",
                             offset_s=0.5, load_unit="kN"),
                    area_mm2=12.5, source="run7.csv")
    restored = LoadData.from_payload(data.to_payload())
    assert restored.sync == data.sync
    assert restored.area_mm2 == 12.5 and restored.source == "run7.csv"
    assert restored.table.names == ("t", "load"), "only what the mapping uses"
    assert np.isnan(restored.table.column("load")[1])


def test_a_malformed_payload_is_a_value_error():
    with pytest.raises(ValueError):
        LoadData.from_payload({"sync": {"mode": "sideways"}})


# --- describing it in an export header -------------------------------------------

def test_the_description_says_where_the_load_came_from_and_how_it_was_matched():
    data = LoadData(_table(t=[0.0, 1.0], F=[0.0, 1.0]),
                    LoadSync(mode="time", load_column="F", time_column="t",
                             offset_s=0.25, load_unit="kN"),
                    area_mm2=12.5, source="run7.csv")
    text = "\n".join(data.describe(frame_rate=2.0))
    assert "run7.csv" in text and "'F'" in text and "kN" in text
    assert "0.25" in text and "2 fps" in text
    assert "12.5 mm" in text


def test_the_description_of_a_frame_match_names_the_numbering():
    data = LoadData(_table(idx=[0.0], F=[1.0]),
                    LoadSync(mode="frame", load_column="F", frame_column="idx",
                             frame_base=0))
    text = "\n".join(data.describe(frame_rate=0.0))
    assert "'idx'" in text and "0" in text
    assert "stress" not in text.lower()
