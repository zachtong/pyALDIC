"""Where a probe's samples go, and the triangulation they are located in.

Pure geometry, no Qt: the area-weighted grid a region is sampled on, the
even-odd test for polygons, the mesh spacing when a run does not record it,
and scipy's barycentric transforms computed up front.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from al_dic.analysis.probes import AreaGeom

# A region on a fine mesh is sampled at most this densely; beyond it the grid
# coarsens evenly.
MAX_AREA_SAMPLES = 40_000


def median_spacing(nodes: NDArray[np.float64]) -> float:
    """Typical node spacing, when the run does not record its mesh step."""
    from scipy.spatial import cKDTree

    if len(nodes) < 2:
        return 1.0
    d, _ = cKDTree(nodes).query(nodes, k=2)
    spacing = float(np.median(d[:, 1]))
    return spacing if spacing > 0 else 1.0


def in_area(geom: AreaGeom, pts: NDArray[np.float64]) -> NDArray[np.bool_]:
    x, y = pts[:, 0], pts[:, 1]
    if geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if geom.shape == "polygon":
        return points_in_polygon(x, y, geom.data)  # type: ignore[arg-type]
    raise ValueError(f"Unknown area shape {geom.shape!r}.")


def area_bounds(geom: AreaGeom) -> tuple[float, float, float, float]:
    if geom.shape == "rect":
        return tuple(geom.data)  # type: ignore[return-value]
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return (cx - r, cy - r, cx + r, cy + r)
    pts = np.asarray(geom.data, dtype=np.float64)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def area_centroid(geom: AreaGeom) -> tuple[float, float]:
    if geom.shape == "circle":
        return geom.data[0], geom.data[1]  # type: ignore[return-value]
    x0, y0, x1, y1 = area_bounds(geom)
    if geom.shape == "rect":
        return (x0 + x1) / 2, (y0 + y1) / 2
    pts = np.asarray(geom.data, dtype=np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def area_grid(geom: AreaGeom, spacing: float) -> NDArray[np.float64]:
    """Cell-centred grid inside *geom* -- an area-weighted sample.

    Averaging nodes instead weights every node equally, which biases the
    result toward refined zones -- and refinement happens at cracks and edges.
    A region too small to hold one grid point is read at its centroid.
    """
    x0, y0, x1, y1 = area_bounds(geom)
    h = max(float(spacing), 1e-6)
    # Cap the count for a huge region on a fine mesh; coarsen evenly.
    area = max((x1 - x0) * (y1 - y0), 0.0)
    if area / (h * h) > MAX_AREA_SAMPLES:
        h = math.sqrt(area / MAX_AREA_SAMPLES)
    xs = np.arange(x0 + h / 2, x1, h)
    ys = np.arange(y0 + h / 2, y1, h)
    if xs.size and ys.size:
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pts = pts[in_area(geom, pts)]
        if len(pts):
            return pts
    cx, cy = area_centroid(geom)
    return np.array([[cx, cy]], dtype=np.float64)


def barycentric_transforms(points: NDArray[np.float64],
                            simplices: NDArray[np.int64]) -> NDArray[np.float64]:
    """``Delaunay.transform``, computed with numpy.

    For simplex i with corners r0, r1, r2: ``T[i, :2]`` inverts the matrix
    whose columns are r0 - r2 and r1 - r2, and ``T[i, 2]`` is r2. A simplex
    with no area gets NaN, as scipy gives it.
    """
    corner = points[simplices]                          # (S, 3, 2)
    a = corner[:, 0] - corner[:, 2]
    b = corner[:, 1] - corner[:, 2]
    det = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    scale = np.hypot(a[:, 0], a[:, 1]) * np.hypot(b[:, 0], b[:, 1])
    ok = np.abs(det) > 1e-12 * np.maximum(scale, np.finfo(np.float64).tiny)
    transform = np.full((len(simplices), 3, 2), np.nan, dtype=np.float64)
    d = det[ok]
    transform[ok, 0, 0] = b[ok, 1] / d
    transform[ok, 0, 1] = -b[ok, 0] / d
    transform[ok, 1, 0] = -a[ok, 1] / d
    transform[ok, 1, 1] = a[ok, 0] / d
    transform[:, 2, :] = corner[:, 2]
    return np.ascontiguousarray(transform)


def prime_transform(tri: Delaunay) -> None:
    """Hand scipy the barycentric transforms it would build slowly.

    scipy computes them lazily, simplex by simplex, on the first point
    location: 3 s on a 78,000-node mesh -- the whole wait before a first probe
    reads anything. The same numbers in numpy take 50 ms. Only filled in where
    scipy keeps the lazy slot this way; any other scipy builds its own, slower
    but correct.
    """
    if getattr(tri, "_transform", False) is None:
        tri._transform = barycentric_transforms(tri.points, tri.simplices)


def points_in_polygon(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    vertices: Sequence[tuple[float, float]],
) -> NDArray[np.bool_]:
    """Even-odd ray casting, vectorised over the query points."""
    verts = np.asarray(vertices, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        straddles = (yi > y) != (yj > y)
        denom = np.where(yj != yi, yj - yi, 1.0)
        crossing_x = (xj - xi) * (y - yi) / denom + xi
        inside ^= straddles & (x < crossing_x)
        j = i
    return inside


__all__ = [
    "MAX_AREA_SAMPLES", "area_bounds", "area_centroid", "area_grid",
    "barycentric_transforms", "in_area", "median_spacing", "points_in_polygon",
    "prime_transform",
]
