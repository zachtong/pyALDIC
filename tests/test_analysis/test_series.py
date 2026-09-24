"""TimeSeries: gaps stay gaps, and a status names the reason."""

from __future__ import annotations

import numpy as np

from al_dic.analysis.series import FrameStatus, TimeSeries


def _ts(values, status):
    n = len(values)
    return TimeSeries(
        frames=np.arange(n, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64),
        valid_fraction=np.ones(n),
        status=list(status),
        unit="",
    )


def test_series_never_bridges_a_gap():
    """One polyline per unbroken stretch; the reference drew through NaN."""
    ts = _ts([0.0, 1.0, np.nan, np.nan, 4.0, 5.0],
             [FrameStatus.OK] * 2 + [FrameStatus.NO_DATA] * 2
             + [FrameStatus.OK] * 2)
    assert list(ts.contiguous_runs()) == [(0, 2), (4, 6)]


def test_a_series_that_is_all_gap_has_no_runs():
    ts = _ts([np.nan, np.nan], [FrameStatus.NO_DATA] * 2)
    assert list(ts.contiguous_runs()) == []


def test_status_is_independent_of_whether_a_value_survived():
    """A crack frame can carry a value; the status says why it is marked."""
    ts = _ts([0.0, 1.0, 2.0], [FrameStatus.OK, FrameStatus.OK, FrameStatus.CRACK])
    assert list(ts.contiguous_runs()) == [(0, 3)]
    assert ts.first_frame(FrameStatus.CRACK) == 2


def test_first_frame_is_none_when_it_never_happens():
    ts = _ts([0.0, 1.0], [FrameStatus.OK] * 2)
    assert ts.first_frame(FrameStatus.CRACK) is None
