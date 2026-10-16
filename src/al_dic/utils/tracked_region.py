"""Hide untracked regions: where a rendered field may be painted.

Incremental tracking stops following material that left the field of view or
the Region of Interest; those nodes carry NaN displacement from then on, and
material that was never in frame 0 has no node at all.  Rendering used to
nearest-fill such areas from distant nodes, painting colours that were never
measured.  With "Hide untracked regions" on, both the on-screen view
(``gui/controllers/viz_controller``) and image / animation export
(``export/export_png``) apply one rule:

* a grid cell is painted only if a TRACKED node (finite displacement) lies
  within :func:`tracked_reach_px` -- one grid cell including its diagonal;
* inside that reach, cells the interpolation left empty (beyond the node hull,
  or blanked next to a crack) are filled from the nearest valid node on the
  same side of any crack, so the field reaches the Region of Interest edge;
* everything beyond the reach stays transparent.

"Tracked" is decided from the displacement, not from the plotted values, so a
strain node removed by edge trimming still counts as tracked.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .crack_barrier import crack_aware_nearest_index

#: Reach in grid steps: a full grid cell, diagonal included, so no corner of
#: a cell bounded by tracked nodes is left unpainted.
TRACKED_REACH_STEPS = math.sqrt(2.0)


def mask_fingerprint(mask: NDArray | None) -> tuple | None:
    """Cheap identity of a mask's content for cache keys (None for no mask)."""
    if mask is None:
        return None
    arr = np.ascontiguousarray(mask)
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).digest()
    return arr.shape, str(arr.dtype), digest


def tracked_reach_px(mesh_step: float) -> float:
    """Distance (px) from a tracked node within which cells may be painted."""
    return float(mesh_step) * TRACKED_REACH_STEPS


def untracked_cells(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    tracked_pts: NDArray[np.float64],
    reach: float,
    where: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """True for grid cells with no tracked node within *reach* px.

    With *where*, only those cells are queried (the rest are reported False):
    cells that end up masked out anyway need no distance query.
    """
    shape = np.asarray(x_grid).shape
    sel = np.ones(shape, dtype=bool) if where is None else np.asarray(where, dtype=bool)
    out = np.zeros(shape, dtype=bool)
    if len(tracked_pts) == 0:
        out[sel] = True
        return out
    query = np.column_stack([np.asarray(x_grid)[sel], np.asarray(y_grid)[sel]])
    if len(query):
        dist, _ = cKDTree(tracked_pts).query(query, k=1, distance_upper_bound=reach)
        out[sel] = ~np.isfinite(dist)
    return out


def paint_tracked_region(
    grid: NDArray[np.float64],
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    inside: NDArray[np.bool_],
    tracked_pts: NDArray[np.float64],
    fill_pts: NDArray[np.float64],
    fill_vals: NDArray[np.float64],
    barrier: NDArray | None,
    reach: float,
    far: NDArray[np.bool_] | None = None,
    cache: dict | None = None,
) -> NDArray[np.float64]:
    """Apply the hide-untracked rule to an interpolated grid (returns a copy).

    Args:
        grid:        interpolated values, NaN where interpolation gave none.
        x_grid, y_grid: cell centres in the render coordinates.
        inside:      cells inside the render mask (only these are filled).
        tracked_pts: positions of tracked nodes (finite displacement).
        fill_pts, fill_vals: nodes allowed to supply a fill value (finite
                     position and finite plotted value).
        barrier:     crack / hole mask (< 0.5 = barrier) for same-side filling,
                     or None.
        reach:       :func:`tracked_reach_px` of the mesh step.
        far:         precomputed :func:`untracked_cells` for this grid, these
                     tracked nodes and ``where=inside`` (geometry only, so a
                     caller rendering several fields of a frame shares it).
                     Cells outside *inside* are left as they are: callers mask
                     them out afterwards.
        cache:       optional dict holding the last fill-source index (keyed
                     by the cells to fill and the fill nodes), so fields of one
                     frame that share their valid nodes reuse it.
    """
    out = np.array(grid, dtype=np.float64, copy=True)
    inside = np.asarray(inside, dtype=bool)
    if far is None:
        far = untracked_cells(x_grid, y_grid, tracked_pts, reach, where=inside)
    need = np.isnan(out) & inside & ~far
    if need.any() and len(fill_pts):
        key = (need.tobytes(), np.asarray(fill_pts).tobytes(), mask_fingerprint(barrier))
        hit = None if cache is None else cache.get("_fill_index")
        if hit is not None and hit[0] == key:
            idx = hit[1]
        else:
            idx = crack_aware_nearest_index(
                np.asarray(x_grid)[need], np.asarray(y_grid)[need], fill_pts, barrier)
            if cache is not None:
                cache["_fill_index"] = (key, idx)
        out[need] = np.asarray(fill_vals, dtype=np.float64)[idx]
    out[far] = np.nan
    return out


def trimmed_cells(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    node_pos: NDArray[np.float64],
    values: NDArray[np.float64],
    tracked: NDArray[np.bool_],
) -> NDArray[np.bool_] | None:
    """Cells whose nearest TRACKED node has no value (edge-trimmed strain).

    The hide-untracked counterpart of the strain trim raster: untracked nodes
    are left to the reach rule and claim no cells here, so a displacement
    field (tracked == has a value) is never trim-blanked.  Returns None when no
    tracked node is trimmed.
    """
    pos_ok = np.isfinite(node_pos).all(axis=1)
    keep = pos_ok & np.asarray(tracked, dtype=bool)
    finite = np.isfinite(np.asarray(values, dtype=np.float64))
    if not keep.any() or bool(finite[keep].all()):
        return None
    shape = np.asarray(x_grid).shape
    query = np.column_stack([np.ravel(x_grid), np.ravel(y_grid)])
    _, idx = cKDTree(node_pos[keep]).query(query, k=1)
    return (~finite[keep][idx]).reshape(shape)
