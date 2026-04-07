"""Tests for the auto-range -> manual-range handoff in the colorbar logic.

Bug being pinned:
    When the user un-checks "Auto" on the colorbar widget, the Min/Max
    spin boxes were initialised from ``state.color_min/color_max`` --
    but those fields were never updated by the auto-mode rendering path,
    so the user always saw the stale defaults (0.0 and 1.0) instead of
    the actual current-frame range.

Fix:
    The auto-range computation in ``canvas_area`` is extracted into a
    pure helper, ``resolve_color_range``, that *also* writes the
    computed (vmin, vmax) back into ``AppState`` when ``color_auto`` is
    True.  When the user later flips Auto off, the spin boxes read
    those fields and start from the values that were just on screen --
    so the displayed colormap does not jump.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from staq_dic.gui.app_state import AppState


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    state = AppState.instance()
    for sig in (
        state.images_changed,
        state.current_frame_changed,
        state.roi_changed,
        state.params_changed,
        state.run_state_changed,
        state.progress_updated,
        state.results_changed,
        state.display_changed,
        state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


class TestResolveColorRange:
    """Pin: auto mode writes the current-frame range back into AppState."""

    def test_auto_writes_percentile_range_to_state(self) -> None:
        """In auto mode, the helper must write the 2-98 percentile of
        ``values`` to ``state.color_min/color_max`` so the ColorRange
        widget reads the live frame range when Auto is toggled off.
        """
        from staq_dic.gui.panels.canvas_area import resolve_color_range

        state = AppState.instance()
        state.color_auto = True
        # Sentinel "stale defaults" -- these MUST be overwritten.
        state.color_min = 0.0
        state.color_max = 1.0

        # Linear ramp; np.percentile([0,1,...,100], [2,98]) -> (2.0, 98.0)
        values = np.arange(101, dtype=np.float64)
        vmin, vmax = resolve_color_range(state, values)

        assert vmin == pytest.approx(2.0)
        assert vmax == pytest.approx(98.0)
        # The bug fix: state must reflect the actual current-frame range.
        assert state.color_min == pytest.approx(2.0)
        assert state.color_max == pytest.approx(98.0)

    def test_auto_with_nans_ignores_nans(self) -> None:
        """NaNs (masked / unsolved subsets) must be excluded before
        the percentile is taken; otherwise vmin/vmax come back NaN and
        the colorbar collapses to nothing.
        """
        from staq_dic.gui.panels.canvas_area import resolve_color_range

        state = AppState.instance()
        state.color_auto = True

        values = np.array(
            [np.nan, 10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
            dtype=np.float64,
        )
        vmin, vmax = resolve_color_range(state, values)

        assert np.isfinite(vmin)
        assert np.isfinite(vmax)
        assert vmin >= 10.0
        assert vmax <= 50.0
        assert state.color_min == vmin
        assert state.color_max == vmax

    def test_auto_with_all_nan_falls_back_to_unit_range(self) -> None:
        """All-NaN frame: the helper degrades gracefully to (0, 1)
        instead of returning NaNs (which would crash matplotlib's
        normaliser).
        """
        from staq_dic.gui.panels.canvas_area import resolve_color_range

        state = AppState.instance()
        state.color_auto = True

        values = np.full(8, np.nan, dtype=np.float64)
        vmin, vmax = resolve_color_range(state, values)

        assert vmin == 0.0
        assert vmax == 1.0
        assert state.color_min == 0.0
        assert state.color_max == 1.0

    def test_manual_mode_returns_state_values_unchanged(self) -> None:
        """In manual mode, the helper must NOT touch state -- it just
        returns whatever the user has dialled in via the spin boxes.
        """
        from staq_dic.gui.panels.canvas_area import resolve_color_range

        state = AppState.instance()
        state.color_auto = False
        state.color_min = -3.5
        state.color_max = 7.25

        # Pass values whose percentile would be [0, 100] -- the helper
        # must IGNORE them and return the manual range instead.
        values = np.linspace(0, 100, 50, dtype=np.float64)
        vmin, vmax = resolve_color_range(state, values)

        assert vmin == pytest.approx(-3.5)
        assert vmax == pytest.approx(7.25)
        # State must be untouched.
        assert state.color_min == pytest.approx(-3.5)
        assert state.color_max == pytest.approx(7.25)


class TestColorRangeWidgetReadsLiveState:
    """Pin: when the user flips Auto off, the ColorRange widget must
    populate the spin boxes from the LATEST state.color_min/color_max
    -- which the rendering path has just written via resolve_color_range.
    """

    def test_unchecking_auto_loads_state_values_into_spinboxes(self) -> None:
        """End-to-end: simulate auto-rendering having just written
        (vmin=2.0, vmax=98.0) to state, then flip the widget's Auto off
        and assert the spin boxes show those values, not the stale
        (0.0, 1.0) defaults.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        del app  # silence linter

        from staq_dic.gui.widgets.color_range import ColorRange

        state = AppState.instance()
        # Simulate an auto-mode render that just observed a frame whose
        # 2-98 percentile is (2.0, 98.0).
        state.color_auto = True
        state.color_min = 2.0
        state.color_max = 98.0

        widget = ColorRange()
        # Sanity: widget should boot in Auto mode with disabled spinboxes.
        assert widget._auto_cb.isChecked()
        assert not widget._min_spin.isEnabled()

        # Flip Auto off -- this should pull (2.0, 98.0) from state.
        widget._auto_cb.setChecked(False)

        assert widget._min_spin.isEnabled()
        assert widget._max_spin.isEnabled()
        assert widget._min_spin.value() == pytest.approx(2.0)
        assert widget._max_spin.value() == pytest.approx(98.0)
