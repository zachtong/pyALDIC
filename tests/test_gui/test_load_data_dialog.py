"""Importing a testing machine's record: mapping columns, matching frames."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from al_dic.analysis.load_data import LoadData, LoadSync, LoadTable, parse_load_table
from al_dic.gui.dialogs.load_data_dialog import LoadDataDialog, guess_mapping


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["pyALDIC-tests"])


BLUEHILL = '''"Time","Extension","Load"
"(s)","(mm)","(kN)"
"0.0","0.00","0.0"
"1.0","0.01","1.0"
"2.0","0.02","2.0"
"3.0","0.03","3.0"
'''


def _table(**columns) -> LoadTable:
    return LoadTable(tuple(columns),
                     tuple(np.asarray(v, dtype=float) for v in columns.values()),
                     "machine.csv")


# --- guessing the mapping --------------------------------------------------------

def test_the_obvious_columns_are_guessed():
    guess = guess_mapping(parse_load_table(BLUEHILL), frame_rate=1.0)
    assert guess.load_column == "Load (kN)"
    assert guess.load_unit == "kN"
    assert guess.mode == "time" and guess.time_column == "Time (s)"


def test_a_frame_column_is_used_when_there_is_no_frame_rate():
    table = _table(**{"Frame": [1, 2], "Time": [0.0, 1.0], "Force (N)": [0.0, 5.0]})
    guess = guess_mapping(table, frame_rate=0.0)
    assert guess.mode == "frame" and guess.frame_column == "Frame"
    assert guess.load_column == "Force (N)" and guess.load_unit == "N"


def test_without_recognisable_names_the_last_column_is_the_load():
    guess = guess_mapping(_table(a=[0, 1], b=[2, 3], c=[4, 5]), frame_rate=1.0)
    assert guess.load_column == "c"
    assert guess.key_column == "a"


# --- the dialog --------------------------------------------------------------------

def test_ok_waits_for_a_file(qapp):
    dialog = LoadDataDialog(None, n_frames=4, frame_rate=1.0)
    assert not dialog._ok_button().isEnabled()
    dialog.set_table(parse_load_table(BLUEHILL, "machine.csv"))
    assert dialog._ok_button().isEnabled()


def test_the_summary_says_how_many_frames_the_record_covers(qapp):
    dialog = LoadDataDialog(None, n_frames=4, frame_rate=1.0)
    dialog.set_table(parse_load_table(BLUEHILL, "machine.csv"))
    assert "4" in dialog._summary.text() and "of 4" in dialog._summary.text()
    dialog._offset.setValue(2.0)            # frames at t = 2, 3, 4, 5
    assert "2" in dialog._summary.text().split("of")[0]


def test_time_matching_is_unavailable_without_a_frame_rate(qapp):
    dialog = LoadDataDialog(None, n_frames=4, frame_rate=0.0)
    dialog.set_table(parse_load_table(BLUEHILL, "machine.csv"))
    assert not dialog._by_time.isEnabled()
    assert dialog._by_frame.isChecked()


def test_the_dialog_builds_the_mapping_it_shows(qapp):
    dialog = LoadDataDialog(None, n_frames=4, frame_rate=1.0)
    dialog.set_table(_table(**{"Frame": [1, 2, 3], "Load (N)": [0.0, 5.0, 10.0]}))
    dialog._by_frame.setChecked(True)
    dialog._area.setValue(2.5)
    data = dialog.load_data()
    assert data.sync.mode == "frame" and data.sync.frame_column == "Frame"
    assert data.sync.load_column == "Load (N)"
    assert data.area_mm2 == 2.5
    dialog._area.setValue(0.0)
    assert dialog.load_data().area_mm2 is None, "0 means no stress, not 0 mm2"


def test_editing_opens_on_the_current_mapping(qapp):
    current = LoadData(_table(t=[0.0, 1.0], F=[0.0, 1.0]),
                       LoadSync(mode="time", load_column="F", time_column="t",
                                offset_s=0.75, load_unit="kN"), area_mm2=3.0)
    dialog = LoadDataDialog(current, n_frames=2, frame_rate=1.0)
    assert dialog._by_time.isChecked()
    assert dialog._offset.value() == pytest.approx(0.75)
    assert dialog._unit.currentData() == "kN"
    assert dialog._area.value() == pytest.approx(3.0)
    assert dialog.load_data().sync == current.sync


def test_remove_clears_the_record(qapp):
    current = LoadData(_table(frame=[1], load=[1.0]),
                       LoadSync(mode="frame", load_column="load", frame_column="frame"))
    dialog = LoadDataDialog(current, n_frames=1, frame_rate=1.0)
    dialog._remove_btn.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.load_data() is None


def test_remove_is_only_offered_when_there_is_something_to_remove(qapp):
    fresh = LoadDataDialog(None, n_frames=1, frame_rate=1.0)
    fresh.show()
    assert fresh._remove_btn.isHidden()
    fresh.close()
    current = LoadData(_table(frame=[1], load=[1.0]),
                       LoadSync(mode="frame", load_column="load", frame_column="frame"))
    editing = LoadDataDialog(current, n_frames=1, frame_rate=1.0)
    editing.show()
    assert not editing._remove_btn.isHidden()
    editing.close()
