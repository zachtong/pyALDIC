"""Selecting and reshaping probes, as pure geometry.

The canvas turns mouse positions into scene coordinates and calls these; the
rules for what a click hits and what a drag produces live here, where they can
be tested without a window.

Every edit is computed from the geometry as it was when the drag began, never
from the previous mouse event. That is what lets a rectangle's corner cross
its anchor without the handles swapping identity half-way through a drag.
"""

from __future__ import annotations

import math

from al_dic.analysis.probes import AreaGeom, Geometry, LineGeom, PointGeom, ProbeSet

#: Below this a shape measures nothing: a line or edge shorter than a pixel,
#: a region smaller than one. Refused rather than stored.
_MIN_LENGTH = 1.0
_MIN_AREA = 1.0

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Control points
# ---------------------------------------------------------------------------

def control_points(geom: Geometry) -> list[Point]:
    """Where the handles of *geom* sit, in the order ``move_control_point`` uses.

    A rectangle's corners run clockwise on screen from the top-left, so the
    corner opposite index *i* is ``(i + 2) % 4``. A circle has its centre and
    one radius handle on the right.
    """
    if isinstance(geom, PointGeom):
        return [(geom.x, geom.y)]
    if isinstance(geom, LineGeom):
        return [(geom.x0, geom.y0), (geom.x1, geom.y1)]
    if geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return [(cx, cy), (cx + r, cy)]
    return [(float(x), float(y)) for x, y in geom.data]  # type: ignore[misc]


def move_control_point(geom: Geometry, index: int, x: float, y: float) -> Geometry | None:
    """*geom* with control point *index* moved to (x, y), or ``None`` if the
    result would be degenerate -- the caller keeps the last good shape."""
    if isinstance(geom, PointGeom):
        return PointGeom(float(x), float(y))
    if isinstance(geom, LineGeom):
        ends = [(geom.x0, geom.y0), (geom.x1, geom.y1)]
        ends[index] = (float(x), float(y))
        (x0, y0), (x1, y1) = ends
        if math.hypot(x1 - x0, y1 - y0) < _MIN_LENGTH:
            return None
        return LineGeom(x0, y0, x1, y1)
    if geom.shape == "rect":
        anchor = control_points(geom)[(index + 2) % 4]
        if abs(x - anchor[0]) < _MIN_LENGTH or abs(y - anchor[1]) < _MIN_LENGTH:
            return None
        return AreaGeom.rect(anchor[0], anchor[1], x, y)
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        if index == 0:
            return AreaGeom.circle(x, y, r)
        radius = math.hypot(x - cx, y - cy)
        if radius < _MIN_LENGTH / 2:
            return None
        return AreaGeom.circle(cx, cy, radius)
    vertices = control_points(geom)
    vertices[index] = (float(x), float(y))
    if not encloses_area(vertices):
        return None
    return AreaGeom.polygon(vertices)


def encloses_area(vertices: list[Point]) -> bool:
    """True when a polygon through *vertices* covers at least a pixel.

    The floor both placing and reshaping apply: below it the region is a
    mis-click along a line, not something to measure.
    """
    return len(vertices) >= 3 and abs(_signed_area(vertices)) >= _MIN_AREA


def translate(geom: Geometry, dx: float, dy: float) -> Geometry:
    """*geom* shifted by (dx, dy); size and shape unchanged."""
    if isinstance(geom, PointGeom):
        return PointGeom(geom.x + dx, geom.y + dy)
    if isinstance(geom, LineGeom):
        return LineGeom(geom.x0 + dx, geom.y0 + dy, geom.x1 + dx, geom.y1 + dy)
    if geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        return AreaGeom.rect(x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return AreaGeom.circle(cx + dx, cy + dy, r)
    return AreaGeom.polygon([(x + dx, y + dy) for x, y in control_points(geom)])


# ---------------------------------------------------------------------------
# Hit testing
# ---------------------------------------------------------------------------

def hit_test(probes: ProbeSet, x: float, y: float, *, tolerance: float) -> int | None:
    """Id of the visible probe under (x, y), or ``None``.

    Two passes. First anything drawn as a mark -- a point, a line, a region's
    outline -- within *tolerance*; then region interiors. Within a pass the
    probe drawn last (on top) wins. The order of passes is what keeps a point
    inside a region selectable when the region was drawn after it.
    """
    visible = [p for p in reversed(list(probes)) if p.visible]
    for probe in visible:
        if _near_mark(probe.geometry, x, y, tolerance):
            return probe.id
    for probe in visible:
        if isinstance(probe.geometry, AreaGeom) and _inside(probe.geometry, x, y):
            return probe.id
    return None


def _near_mark(geom: Geometry, x: float, y: float, tol: float) -> bool:
    if isinstance(geom, PointGeom):
        return math.hypot(x - geom.x, y - geom.y) <= tol
    if isinstance(geom, LineGeom):
        return _segment_distance(x, y, geom.x0, geom.y0, geom.x1, geom.y1) <= tol
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return abs(math.hypot(x - cx, y - cy) - r) <= tol
    ring = control_points(geom)
    return any(
        _segment_distance(x, y, *ring[i], *ring[(i + 1) % len(ring)]) <= tol
        for i in range(len(ring))
    )


def _inside(geom: AreaGeom, x: float, y: float) -> bool:
    if geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        return x0 <= x <= x1 and y0 <= y <= y1
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return math.hypot(x - cx, y - cy) <= r
    return _point_in_polygon(x, y, control_points(geom))


def _segment_distance(px: float, py: float, x0: float, y0: float,
                      x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _point_in_polygon(x: float, y: float, vertices: list[Point]) -> bool:
    """Even-odd rule, which is what the engine's region mask uses too."""
    inside = False
    n = len(vertices)
    for i in range(n):
        xa, ya = vertices[i]
        xb, yb = vertices[(i + 1) % n]
        if (ya > y) != (yb > y):
            if x < xa + (y - ya) * (xb - xa) / (yb - ya):
                inside = not inside
    return inside


def _signed_area(vertices: list[Point]) -> float:
    n = len(vertices)
    return 0.5 * sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )


__all__ = [
    "control_points", "encloses_area", "hit_test", "move_control_point",
    "translate",
]
