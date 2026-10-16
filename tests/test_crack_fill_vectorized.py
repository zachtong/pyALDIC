"""crack_aware_nearest_fill must stay bit-identical to its original per-cell loop.

The loop below is the implementation the function shipped with; the vectorised
version replaces it for speed (edge filling now runs on every render) and has to
pick exactly the same node for every cell.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from al_dic.utils.crack_barrier import crack_aware_nearest_fill, segment_crosses_barrier


def _reference_fill(x_grid, y_grid, need_fill, node_pts, node_vals, mask):
    out = np.full(np.asarray(x_grid).shape, np.nan, dtype=np.float64)
    need_fill = np.asarray(need_fill, dtype=bool)
    if not need_fill.any() or len(node_pts) == 0:
        return out
    qx = np.asarray(x_grid)[need_fill]
    qy = np.asarray(y_grid)[need_fill]
    query = np.column_stack([qx, qy])
    tree = cKDTree(node_pts)
    if mask is None:
        _, idx = tree.query(query, k=1)
        out[need_fill] = node_vals[np.asarray(idx)]
        return out
    kk = int(min(16, len(node_pts)))
    _, idx = tree.query(query, k=kk)
    idx = np.atleast_2d(idx)
    if idx.shape[0] == 1 and query.shape[0] != 1:
        idx = idx.T
    filled = np.empty(query.shape[0], dtype=np.float64)
    for j in range(query.shape[0]):
        cx, cy = float(qx[j]), float(qy[j])
        chosen = int(idx[j, 0])
        for c in range(idx.shape[1]):
            ni = int(idx[j, c])
            if not segment_crosses_barrier(
                cx, cy, float(node_pts[ni, 0]), float(node_pts[ni, 1]), mask
            ):
                chosen = ni
                break
        filled[j] = node_vals[chosen]
    out[need_fill] = filled
    return out


def _scene(seed, h=160, w=200, step=8, holes=True):
    rng = np.random.default_rng(seed)
    mask = np.ones((h, w), dtype=float)
    mask[70:76, 30:w] = 0.0                                  # crack from the right
    if holes:
        yy, xx = np.mgrid[0:h, 0:w]
        mask[(yy - 40) ** 2 + (xx - 60) ** 2 < 15 ** 2] = 0.0   # round hole
        mask[120:124, 0:120] = 0.0                            # crack from the left
    gy, gx = np.mgrid[step // 2:h:step, step // 2:w:step].astype(float)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    keep = mask[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0.5
    keep &= rng.random(len(pts)) > 0.15                      # drop some nodes
    pts = pts[keep] + rng.normal(0, 0.7, (int(keep.sum()), 2))
    vals = rng.normal(size=len(pts))
    qy, qx = np.mgrid[0:h:1.5, 0:w:1.5]
    need = rng.random(qx.shape) > 0.3
    return qx, qy, need, pts, vals, mask


@pytest.mark.parametrize("seed", range(6))
def test_matches_reference_with_cracks_and_holes(seed):
    qx, qy, need, pts, vals, mask = _scene(seed)
    want = _reference_fill(qx, qy, need, pts, vals, mask)
    got = crack_aware_nearest_fill(qx, qy, need, pts, vals, mask)
    np.testing.assert_array_equal(got, want)


def test_matches_reference_without_mask():
    qx, qy, need, pts, vals, _ = _scene(7)
    np.testing.assert_array_equal(
        crack_aware_nearest_fill(qx, qy, need, pts, vals, None),
        _reference_fill(qx, qy, need, pts, vals, None))


def test_matches_reference_when_every_candidate_crosses():
    # A node walled off completely: every candidate crosses, so the plain
    # nearest node is the fallback -- the vectorised path must keep that.
    mask = np.ones((60, 60))
    mask[:, 29:31] = 0.0
    pts = np.array([[10.0, 30.0]])
    vals = np.array([5.0])
    qy, qx = np.mgrid[0:60:2.0, 0:60:2.0]
    need = np.ones(qx.shape, bool)
    np.testing.assert_array_equal(
        crack_aware_nearest_fill(qx, qy, need, pts, vals, mask),
        _reference_fill(qx, qy, need, pts, vals, mask))


def test_matches_reference_single_query_and_few_nodes():
    mask = np.ones((40, 40))
    mask[20, :] = 0.0
    pts = np.array([[5.0, 5.0], [5.0, 35.0], [35.0, 35.0]])
    vals = np.array([1.0, 2.0, 3.0])
    qx = np.array([[5.0]])
    qy = np.array([[30.0]])
    need = np.array([[True]])
    np.testing.assert_array_equal(
        crack_aware_nearest_fill(qx, qy, need, pts, vals, mask),
        _reference_fill(qx, qy, need, pts, vals, mask))


def test_matches_reference_with_a_few_very_long_segments():
    # Sparse cloud: most queries are near a node, a few are far away. The
    # vectorised path batches by length; results must not depend on that.
    mask = np.ones((400, 400))
    mask[200:203, 50:400] = 0.0
    pts = np.array([[20.0, 20.0], [25.0, 380.0], [380.0, 30.0], [210.0, 190.0], [215.0, 215.0]])
    vals = np.arange(len(pts), dtype=float)
    qy, qx = np.mgrid[0:400:3.0, 0:400:3.0]
    need = np.ones(qx.shape, bool)
    np.testing.assert_array_equal(
        crack_aware_nearest_fill(qx, qy, need, pts, vals, mask),
        _reference_fill(qx, qy, need, pts, vals, mask))


def test_segment_batches_respect_the_sample_budget(monkeypatch):
    import al_dic.utils.crack_barrier as cb

    seen = []
    real_round = np.round

    def spy_round(a, *args, **kw):
        seen.append(np.asarray(a).size)
        return real_round(a, *args, **kw)

    monkeypatch.setattr(cb.np, "round", spy_round)
    n = 5000
    x0 = np.zeros(n)
    y0 = np.zeros(n)
    x1 = np.full(n, 3.0)
    y1 = np.zeros(n)
    x1[0] = 3000.0                                # one very long segment
    cb._segments_cross_barrier(x0, y0, x1, y1, np.ones((10, 4000)))
    assert max(seen) <= cb._SEGMENT_SAMPLE_BUDGET + 4000
