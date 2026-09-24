"""The colorbar panel is sized from what it has to hold.

Its left edge used to be a hard-coded 50 px from the bar, which clipped two
separate things: long field labels ("von Mises strain" arrived as "von M...")
and ticks in scientific notation. The arithmetic is tested here rather than
the painting, so a font change does not turn these red for no reason.
"""
from __future__ import annotations

import pytest

from al_dic.gui.widgets.colorbar_overlay import (
    _BAR_WIDTH, _RIGHT_MARGIN, panel_layout,
)

W = 900                                   # a realistic strain-window canvas
BAR_X = W - _RIGHT_MARGIN - _BAR_WIDTH


def test_a_long_label_stays_inside_the_widget():
    """The bar hugs the right edge, so a label centred on it overflows the
    widget before it overflows the panel -- which is what users saw as the
    right-hand panel 'covering' the text."""
    label_w = 160.0
    _, label_x = panel_layout(W, BAR_X, tick_w=40.0, label_w=label_w)
    assert label_x + label_w <= W, "label runs off the right edge"


def test_the_panel_reaches_the_label():
    label_w = 160.0
    panel_x, label_x = panel_layout(W, BAR_X, tick_w=40.0, label_w=label_w)
    assert panel_x <= label_x, "the label starts outside its own panel"


def test_the_panel_holds_the_widest_tick():
    """-1.23e-05 in Consolas 9 is wider than the old fixed 50 px."""
    tick_w = 72.0
    panel_x, _ = panel_layout(W, BAR_X, tick_w=tick_w, label_w=0.0)
    assert BAR_X - panel_x >= tick_w, "the widest tick label overflows"


def test_a_short_label_does_not_widen_the_panel():
    """Sizing to contents must not mean always-maximum."""
    narrow, _ = panel_layout(W, BAR_X, tick_w=30.0, label_w=24.0)
    wide, _ = panel_layout(W, BAR_X, tick_w=30.0, label_w=200.0)
    assert narrow > wide, "a short label should give a narrower panel"
    assert BAR_X - narrow == pytest.approx(50.0), "minimum width lost"


def test_no_label_still_gives_a_panel():
    panel_x, _ = panel_layout(W, BAR_X, tick_w=20.0, label_w=0.0)
    assert 0 < panel_x < BAR_X


def test_the_panel_never_leaves_the_widget():
    """A label wider than the canvas must not push the panel off-screen."""
    panel_x, label_x = panel_layout(200, 200 - _RIGHT_MARGIN - _BAR_WIDTH,
                                    tick_w=90.0, label_w=400.0)
    assert panel_x >= 0.0
    assert label_x >= 0.0
