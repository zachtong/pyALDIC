"""Hide untracked regions: the shared paint rule used by screen and export."""
from __future__ import annotations

import math

import numpy as np

from al_dic.utils.tracked_region import (
    TRACKED_REACH_STEPS,
    paint_tracked_region,
    tracked_reach_px,
    trimmed_cells,
    untracked_cells,
)


def _grid(h=100, w=120, step=1.0):
    yg, xg = np.mgrid[0:h:step, 0:w:step].astype(float)
    return xg, yg


def test_reach_is_one_grid_cell_diagonal():
    assert TRACKED_REACH_STEPS == math.sqrt(2.0)
    assert tracked_reach_px(16) == 16 * math.sqrt(2.0)


def test_untracked_cells_far_from_every_tracked_node():
    xg, yg = _grid()
    pts = np.array([[20.0, 20.0]])
    far = untracked_cells(xg, yg, pts, reach=10.0)
    assert not far[20, 20] and not far[20, 29] and not far[27, 27]
    assert far[20, 31] and far[80, 100]


def test_untracked_cells_without_tracked_nodes_is_everything():
    xg, yg = _grid(10, 10)
    assert untracked_cells(xg, yg, np.empty((0, 2)), reach=5.0).all()


def test_paint_fills_edge_band_and_blanks_dead_region():
    # Tracked nodes cover x < 60; nodes at x >= 60 died (not passed as tracked).
    xg, yg = _grid()
    gy, gx = np.mgrid[4:100:8, 4:60:8].astype(float)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vals = pts[:, 0] * 0.1
    grid = np.full(xg.shape, np.nan)
    grid[4:93, 4:53] = 1.0                       # "interpolated" interior; edges NaN
    inside = np.ones(xg.shape, bool)             # specimen mask covers everything
    reach = tracked_reach_px(8)
    out = paint_tracked_region(grid, xg, yg, inside, tracked_pts=pts, fill_pts=pts,
                               fill_vals=vals, barrier=None, reach=reach)
    assert np.all(out[4:93, 4:53] == 1.0)        # interior untouched
    assert np.isfinite(out[0, 0]) and np.isfinite(out[99, 0])   # edge band filled
    assert np.isfinite(out[50, 60])              # within reach of x=52 nodes
    assert np.isnan(out[50, 80]) and np.isnan(out[0, 119])      # dead region blank
    assert np.isnan(grid[0, 0])                  # input not mutated


def test_paint_never_fills_outside_inside_mask():
    xg, yg = _grid(40, 40)
    pts = np.array([[20.0, 20.0]])
    grid = np.full(xg.shape, np.nan)
    inside = np.zeros(xg.shape, bool)
    inside[15:25, 15:25] = True
    out = paint_tracked_region(grid, xg, yg, inside, pts, pts, np.array([3.0]), None, reach=30.0)
    assert np.all(out[inside] == 3.0)
    assert np.isnan(out[~inside]).all()


def test_paint_respects_crack_barrier():
    # Crack along y = 50; nodes on both sides with different values.
    xg, yg = _grid()
    barrier = np.ones((100, 120))
    barrier[49:52, :] = 0.0
    pts = np.array([[60.0, 45.0], [60.0, 70.0]])
    vals = np.array([-1.0, 1.0])
    grid = np.full(xg.shape, np.nan)
    inside = barrier > 0.5
    out = paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, barrier, reach=40.0)
    # The cell just below the crack is nearer the node ABOVE it (8 px vs 17 px)
    # but must take the value from its own side of the crack.
    assert out[53, 60] == 1.0
    assert out[47, 60] == -1.0


def test_paint_keeps_existing_values_inside_reach():
    xg, yg = _grid(20, 20)
    pts = np.array([[10.0, 10.0]])
    grid = np.full(xg.shape, 7.0)
    out = paint_tracked_region(grid, xg, yg, np.ones(xg.shape, bool), pts, pts,
                               np.array([1.0]), None, reach=5.0)
    assert out[10, 10] == 7.0 and out[10, 14] == 7.0
    assert np.isnan(out[0, 0])                   # beyond reach -> blank even if it had a value


def test_paint_with_no_tracked_nodes_blanks_all():
    xg, yg = _grid(10, 10)
    grid = np.ones(xg.shape)
    out = paint_tracked_region(grid, xg, yg, np.ones(xg.shape, bool), np.empty((0, 2)),
                               np.empty((0, 2)), np.empty(0), None, reach=5.0)
    assert np.isnan(out).all()


def test_paint_accepts_precomputed_far_mask():
    xg, yg = _grid(30, 30)
    pts = np.array([[5.0, 5.0]])
    far = untracked_cells(xg, yg, pts, reach=6.0)
    grid = np.full(xg.shape, np.nan)
    a = paint_tracked_region(grid, xg, yg, np.ones(xg.shape, bool), pts, pts,
                             np.array([2.0]), None, reach=6.0)
    b = paint_tracked_region(grid, xg, yg, np.ones(xg.shape, bool), pts, pts,
                             np.array([2.0]), None, reach=6.0, far=far)
    np.testing.assert_array_equal(a, b)


def test_trimmed_cells_only_for_tracked_nodes_with_nan_values():
    # x=5 tracked + valid, x=15 tracked + trimmed (NaN value), x=25 untracked.
    xg, yg = _grid(10, 30)
    pos = np.array([[5.0, 5.0], [15.0, 5.0], [25.0, 5.0]])
    vals = np.array([1.0, np.nan, np.nan])
    tracked = np.array([True, True, False])
    t = trimmed_cells(xg, yg, pos, vals, tracked)
    assert not t[5, 5] and t[5, 15]
    # The untracked node claims nothing: its cells go to the nearest tracked node.
    assert t[5, 25]                     # nearest TRACKED node is the trimmed one


def test_trimmed_cells_none_when_nothing_trimmed():
    xg, yg = _grid(10, 10)
    pos = np.array([[5.0, 5.0]])
    assert trimmed_cells(xg, yg, pos, np.array([1.0]), np.array([True])) is None
    assert trimmed_cells(xg, yg, pos, np.array([np.nan]), np.array([False])) is None


def test_untracked_cells_where_limits_the_query():
    xg, yg = _grid(30, 30)
    pts = np.array([[5.0, 5.0]])
    where = np.zeros(xg.shape, bool)
    where[:, :15] = True
    full = untracked_cells(xg, yg, pts, reach=6.0)
    part = untracked_cells(xg, yg, pts, reach=6.0, where=where)
    np.testing.assert_array_equal(part[where], full[where])
    assert not part[~where].any()               # unqueried cells are never "far"


def test_paint_result_is_independent_of_cells_outside_inside():
    xg, yg = _grid(40, 40)
    pts = np.array([[10.0, 10.0], [30.0, 10.0]])
    vals = np.array([1.0, 2.0])
    grid = np.full(xg.shape, np.nan)
    grid[5:15, 5:35] = 1.5
    inside = np.zeros(xg.shape, bool)
    inside[0:20, :] = True
    out = paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, None, reach=8.0)
    full_far = untracked_cells(xg, yg, pts, reach=8.0)
    ref = grid.copy()
    need = np.isnan(ref) & inside & ~full_far
    from al_dic.utils.crack_barrier import crack_aware_nearest_fill
    ref = np.where(need, crack_aware_nearest_fill(xg, yg, need, pts, vals, None), ref)
    ref[full_far] = np.nan
    np.testing.assert_array_equal(out[inside], ref[inside])


def test_fill_index_cache_gives_identical_results_across_fields():
    xg, yg = _grid(60, 60)
    barrier = np.ones((60, 60))
    barrier[29:31, 10:60] = 0.0
    pts = np.array([[10.0, 10.0], [40.0, 20.0], [40.0, 45.0], [15.0, 50.0]])
    grid = np.full(xg.shape, np.nan)
    inside = barrier > 0.5
    cache = {}
    for vals in (np.array([1.0, 2.0, 3.0, 4.0]), np.array([-1.0, -2.0, -3.0, -4.0])):
        a = paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, barrier, reach=30.0)
        b = paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, barrier, reach=30.0,
                                 cache=cache)
        np.testing.assert_array_equal(a, b)
    assert "_fill_index" in cache


def test_fill_index_cache_is_keyed_by_the_barrier():
    xg, yg = _grid(60, 60)
    pts = np.array([[30.0, 20.0], [30.0, 45.0]])
    vals = np.array([-1.0, 1.0])
    open_ = np.ones((60, 60))
    crack = open_.copy()
    crack[30:33, :] = 0.0
    grid = np.full(xg.shape, np.nan)
    inside = np.ones(xg.shape, bool)
    cache = {}
    paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, open_, 60.0, cache=cache)
    with_cache = paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, crack, 60.0, cache=cache)
    fresh = paint_tracked_region(grid, xg, yg, inside, pts, pts, vals, crack, 60.0)
    np.testing.assert_array_equal(with_cache, fresh)
