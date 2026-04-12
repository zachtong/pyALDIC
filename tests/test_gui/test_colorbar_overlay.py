"""Tests for colorbar overlay pure functions (_nice_ticks, _format_tick)."""

import pytest

from al_dic.gui.widgets.colorbar_overlay import _nice_ticks, _format_tick


# ---------------------------------------------------------------------------
# _nice_ticks
# ---------------------------------------------------------------------------

class TestNiceTicks:
    def test_normal_range(self):
        ticks = _nice_ticks(-8.0, 12.0, n=5)
        assert len(ticks) == 5
        assert ticks[0] == pytest.approx(-8.0)
        assert ticks[-1] == pytest.approx(12.0)
        # Evenly spaced: step = 20/4 = 5
        for i in range(4):
            assert ticks[i + 1] - ticks[i] == pytest.approx(5.0)

    def test_negative_range(self):
        ticks = _nice_ticks(-10.0, -2.0, n=5)
        assert len(ticks) == 5
        assert ticks[0] == pytest.approx(-10.0)
        assert ticks[-1] == pytest.approx(-2.0)

    def test_degenerate_equal(self):
        """When vmin == vmax, return [vmin]."""
        ticks = _nice_ticks(5.0, 5.0, n=5)
        assert ticks == [5.0]

    def test_degenerate_inverted(self):
        """When vmax < vmin, return [vmin]."""
        ticks = _nice_ticks(10.0, 5.0, n=5)
        assert ticks == [10.0]

    def test_n_equals_2(self):
        ticks = _nice_ticks(0.0, 10.0, n=2)
        assert len(ticks) == 2
        assert ticks[0] == pytest.approx(0.0)
        assert ticks[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _format_tick
# ---------------------------------------------------------------------------

class TestFormatTick:
    def test_zero(self):
        assert _format_tick(0.0) == "0"
        assert _format_tick(1e-12) == "0"

    def test_integer(self):
        assert _format_tick(5.0) == "5"
        assert _format_tick(-3.0) == "-3"

    def test_scientific_large(self):
        result = _format_tick(1500.0)
        assert "e" in result or "E" in result

    def test_scientific_small(self):
        result = _format_tick(0.005)
        assert "e" in result or "E" in result

    def test_small_float(self):
        result = _format_tick(0.123)
        assert result == "0.123"
