"""Selecting and reshaping probes: the geometry, without the canvas."""

from __future__ import annotations

import pytest

from al_dic.analysis.geometry_edit import (
    control_points,
    hit_test,
    move_control_point,
    translate,
)
from al_dic.analysis.probes import AreaGeom, LineGeom, PointGeom, ProbeSet


def _set():
    ps = ProbeSet()
    ps.add("point", PointGeom(10.0, 10.0))                       # 1
    ps.add("line", LineGeom(20.0, 50.0, 80.0, 50.0))             # 2
    ps.add("area", AreaGeom.rect(100.0, 100.0, 140.0, 130.0))    # 3
    ps.add("area", AreaGeom.circle(200.0, 200.0, 15.0))          # 4
    ps.add("area", AreaGeom.polygon([(300, 300), (340, 300), (320, 340)]))  # 5
    return ps


# --- hit testing ---------------------------------------------------------

@pytest.mark.parametrize("x, y, expected", [
    (11.0, 9.0, 1),          # near the point
    (50.0, 52.0, 2),         # near the line's body
    (120.0, 115.0, 3),       # inside the rectangle
    (205.0, 205.0, 4),       # inside the circle
    (320.0, 310.0, 5),       # inside the polygon
    (500.0, 500.0, None),    # nothing
])
def test_a_click_finds_the_probe_under_it(x, y, expected):
    assert hit_test(_set(), x, y, tolerance=4.0) == expected


def test_a_line_is_hit_along_its_body_not_only_at_its_ends():
    assert hit_test(_set(), 50.0, 58.0, tolerance=4.0) is None     # 8 px off
    assert hit_test(_set(), 50.0, 53.0, tolerance=4.0) == 2


def test_hidden_probes_cannot_be_clicked():
    from al_dic.analysis.probes import replace

    ps = _set()
    ps.replace(replace(ps.get(1), visible=False))
    assert hit_test(ps, 11.0, 9.0, tolerance=4.0) is None


def test_the_topmost_probe_wins_where_two_overlap():
    ps = ProbeSet()
    ps.add("area", AreaGeom.rect(0.0, 0.0, 100.0, 100.0))
    ps.add("point", PointGeom(50.0, 50.0))
    assert hit_test(ps, 50.0, 50.0, tolerance=4.0) == 2


def test_a_region_drawn_later_does_not_bury_what_lies_inside_it():
    """Markers and outlines beat region interiors, whatever the order.

    Otherwise a region drawn around a point would make the point
    unselectable: every click on it would land in the region.
    """
    ps = ProbeSet()
    ps.add("point", PointGeom(50.0, 50.0))
    ps.add("line", LineGeom(20.0, 70.0, 80.0, 70.0))
    ps.add("area", AreaGeom.rect(0.0, 0.0, 100.0, 100.0))
    assert hit_test(ps, 50.0, 50.0, tolerance=4.0) == 1
    assert hit_test(ps, 50.0, 71.0, tolerance=4.0) == 2
    assert hit_test(ps, 30.0, 30.0, tolerance=4.0) == 3, "the interior still works"
    assert hit_test(ps, 100.0, 40.0, tolerance=4.0) == 3, "and so does its edge"


# --- control points ----------------------------------------------------------

def test_control_points_per_shape():
    ps = _set()
    assert control_points(ps.get(1).geometry) == [(10.0, 10.0)]
    assert control_points(ps.get(2).geometry) == [(20.0, 50.0), (80.0, 50.0)]
    assert len(control_points(ps.get(3).geometry)) == 4           # corners
    assert control_points(ps.get(4).geometry) == [(200.0, 200.0), (215.0, 200.0)]
    assert len(control_points(ps.get(5).geometry)) == 3


def test_dragging_a_line_end_moves_only_that_end():
    g = LineGeom(20.0, 50.0, 80.0, 50.0)
    moved = move_control_point(g, 1, 90.0, 60.0)
    assert moved == LineGeom(20.0, 50.0, 90.0, 60.0)


def test_dragging_a_rectangle_corner_keeps_the_opposite_one():
    g = AreaGeom.rect(100.0, 100.0, 140.0, 130.0)
    corners = control_points(g)
    far = corners.index((140.0, 130.0))
    moved = move_control_point(g, far, 160.0, 150.0)
    assert moved == AreaGeom.rect(100.0, 100.0, 160.0, 150.0)


def test_the_circle_has_a_centre_and_a_radius_handle():
    g = AreaGeom.circle(200.0, 200.0, 15.0)
    assert move_control_point(g, 1, 200.0, 230.0) == AreaGeom.circle(200.0, 200.0, 30.0)
    moved = move_control_point(g, 0, 210.0, 190.0)
    assert moved == AreaGeom.circle(210.0, 190.0, 15.0), "moving the centre keeps r"


def test_a_polygon_vertex_moves_alone():
    g = AreaGeom.polygon([(300, 300), (340, 300), (320, 340)])
    moved = move_control_point(g, 2, 320.0, 360.0)
    assert moved == AreaGeom.polygon([(300, 300), (340, 300), (320, 360)])


def test_a_degenerate_result_is_refused_not_stored():
    """Dragging a line's end onto its start would store an unmeasurable probe."""
    g = LineGeom(20.0, 50.0, 80.0, 50.0)
    assert move_control_point(g, 1, 20.0, 50.0) is None
    assert move_control_point(AreaGeom.circle(0.0, 0.0, 5.0), 1, 0.0, 0.0) is None
    rect = AreaGeom.rect(100.0, 100.0, 140.0, 130.0)
    far = control_points(rect).index((140.0, 130.0))
    assert move_control_point(rect, far, 140.0, 100.0) is None, "zero height"
    tri = AreaGeom.polygon([(0, 0), (10, 0), (5, 9)])
    assert move_control_point(tri, 2, 5.0, 0.0) is None, "collapsed onto a line"


def test_a_rectangle_corner_dragged_past_its_anchor_flips_cleanly():
    """Moves are computed from the geometry at the start of the drag, so a
    corner can cross its anchor without the handles swapping mid-drag."""
    g = AreaGeom.rect(100.0, 100.0, 140.0, 130.0)
    far = control_points(g).index((140.0, 130.0))
    assert move_control_point(g, far, 60.0, 70.0) == AreaGeom.rect(60.0, 70.0, 100.0, 100.0)


# --- moving a whole probe -------------------------------------------------------

@pytest.mark.parametrize("geom", [
    PointGeom(10.0, 10.0),
    LineGeom(20.0, 50.0, 80.0, 50.0),
    AreaGeom.rect(100.0, 100.0, 140.0, 130.0),
    AreaGeom.circle(200.0, 200.0, 15.0),
    AreaGeom.polygon([(300, 300), (340, 300), (320, 340)]),
])
def test_translation_moves_every_point_by_the_same_amount(geom):
    moved = translate(geom, 5.0, -3.0)
    before = control_points(geom)
    after = control_points(moved)
    assert [(x + 5.0, y - 3.0) for x, y in before] == pytest.approx(after)
    assert type(moved) is type(geom)
