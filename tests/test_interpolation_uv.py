"""Byte-identity tests for scattered_interpolant_uv (Phase 1: shared
triangulation for the cumulative-displacement transform). The uv helper must
be byte-identical to two independent scattered_interpolant calls."""

import numpy as np

from al_dic.utils.interpolation import (
    scattered_interpolant,
    scattered_interpolant_uv,
    scattered_interpolant_uv_diag,
)


def _ref(points, u, v, query):
    return (
        scattered_interpolant(points, u, query),
        scattered_interpolant(points, v, query),
    )


def _pts(n, seed):
    return np.random.default_rng(seed).random((n, 2)) * 100.0


def test_reused_delaunay_is_byte_identical():
    """The export reuse relies on a cached triangulation being byte-identical
    to a fresh interpolator: LinearNDInterpolator(Delaunay(pts), vals) ==
    LinearNDInterpolator(pts, vals) for multiple value sets reusing one tri."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    pts = _pts(80, 11)
    gx, gy = np.meshgrid(np.linspace(0, 100, 40), np.linspace(0, 100, 40))
    tri = Delaunay(pts)
    rng = np.random.default_rng(12)
    for _ in range(3):
        vals = rng.standard_normal(80)
        fresh = LinearNDInterpolator(pts, vals, fill_value=np.nan)(gx, gy)
        reused = LinearNDInterpolator(tri, vals, fill_value=np.nan)(gx, gy)
        np.testing.assert_array_equal(reused, fresh)


def test_scattered_uv_tri_cache_byte_identical():
    """tri_cache must not change results, and reuses one Delaunay across
    calls sharing the same points + valid mask (the cumulative-transform
    cross-frame speedup)."""
    pts = _pts(70, 20)
    query = pts + np.random.default_rng(21).standard_normal(pts.shape) * 0.5
    rng = np.random.default_rng(22)
    cache: dict = {}
    for _ in range(4):
        u = rng.standard_normal(70)
        v = rng.standard_normal(70)
        eu, ev = scattered_interpolant_uv(pts, u, v, query)
        cu, cv = scattered_interpolant_uv(pts, u, v, query, tri_cache=cache)
        np.testing.assert_array_equal(cu, eu)
        np.testing.assert_array_equal(cv, ev)
    # same points + all-finite -> one shared triangulation across all 4 calls
    assert len(cache) == 1


class TestScatteredInterpolantUV:
    def test_no_nan_non_identity(self):
        pts = _pts(60, 0)
        rng = np.random.default_rng(1)
        u, v = rng.standard_normal(60), rng.standard_normal(60)
        query = _pts(40, 2) * 0.8 + 5
        eu, ev = _ref(pts, u, v, query)
        au, av = scattered_interpolant_uv(pts, u, v, query)
        np.testing.assert_array_equal(au, eu)
        np.testing.assert_array_equal(av, ev)

    def test_shared_nan_mask(self):
        pts = _pts(60, 3)
        rng = np.random.default_rng(4)
        u, v = rng.standard_normal(60), rng.standard_normal(60)
        bad = rng.random(60) < 0.15        # a failed POI loses both u and v
        u[bad] = np.nan
        v[bad] = np.nan
        query = _pts(40, 5)
        eu, ev = _ref(pts, u, v, query)
        au, av = scattered_interpolant_uv(pts, u, v, query)
        np.testing.assert_array_equal(au, eu)
        np.testing.assert_array_equal(av, ev)

    def test_identity_query_no_nan(self):
        # Accumulative-mode case: query == data points. The short-circuit
        # must be byte-identical to interpolating at the data vertices.
        pts = _pts(80, 6)
        rng = np.random.default_rng(7)
        u, v = rng.standard_normal(80), rng.standard_normal(80)
        query = pts.copy()
        eu, ev = _ref(pts, u, v, query)
        au, av = scattered_interpolant_uv(pts, u, v, query)
        np.testing.assert_array_equal(au, eu)
        np.testing.assert_array_equal(av, ev)

    def test_differing_nan_masks_fallback(self):
        pts = _pts(60, 8)
        rng = np.random.default_rng(9)
        u, v = rng.standard_normal(60), rng.standard_normal(60)
        u[rng.random(60) < 0.1] = np.nan
        v[rng.random(60) < 0.1] = np.nan   # different nodes -> fallback path
        query = _pts(40, 10)
        eu, ev = _ref(pts, u, v, query)
        au, av = scattered_interpolant_uv(pts, u, v, query)
        np.testing.assert_array_equal(au, eu)
        np.testing.assert_array_equal(av, ev)


# ---------------------------------------------------------------------------
# scattered_interpolant_uv_diag: same values, plus which queries fell outside
# the convex hull and how far their nearest-neighbor fill came from. Always
# checked against the independent _ref (two scattered_interpolant calls),
# never against the uv wrapper, which delegates to the function under test.
# ---------------------------------------------------------------------------


def _hull_truth(points, values, query):
    """Independent outside mask and nearest distance for one field."""
    from scipy.spatial import Delaunay, cKDTree

    valid = ~np.isnan(values) & np.isfinite(points).all(axis=1)
    pts = points[valid]
    outside = Delaunay(pts).find_simplex(query) < 0
    dist = np.zeros(len(query))
    if outside.any():
        dist[outside] = cKDTree(pts).query(query[outside])[0]
    return outside, dist


def _lattice_case():
    """Regular lattice (co-circular cells) queried at exact cell midpoints,
    cell edges and outside the hull: Delaunay ties everywhere."""
    xs, ys = np.meshgrid(np.arange(6) * 16.0, np.arange(5) * 16.0)
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    u = 0.3 * pts[:, 0] - 0.1 * pts[:, 1] + 2.0
    v = 0.05 * pts[:, 0] * pts[:, 1] / 16.0
    mx, my = np.meshgrid(np.arange(-1, 7) * 16.0 + 8.0, np.arange(-1, 6) * 16.0)
    query = np.vstack([
        np.column_stack([mx.ravel(), my.ravel()]),
        np.column_stack([mx.ravel(), my.ravel() + 8.0]),
    ])
    return pts, u, v, query


class TestScatteredInterpolantUVDiag:
    def _check(self, pts, u, v, query, **kw):
        eu, ev = _ref(pts, u, v, query)
        du, dv, outside, nn_dist = scattered_interpolant_uv_diag(
            pts, u, v, query, **kw,
        )
        np.testing.assert_array_equal(du, eu)
        np.testing.assert_array_equal(dv, ev)
        assert outside.dtype == bool and outside.shape == (len(query),)
        assert nn_dist.shape == (len(query),)
        return outside, nn_dist

    def test_values_outside_and_distance_random(self):
        pts = _pts(60, 30)
        rng = np.random.default_rng(31)
        u, v = rng.standard_normal(60), rng.standard_normal(60)
        query = _pts(80, 32) * 1.4 - 20.0  # many queries beyond the hull
        outside, nn_dist = self._check(pts, u, v, query)
        t_out, t_dist = _hull_truth(pts, u, query)
        assert t_out.sum() > 10, "fixture must extrapolate"
        np.testing.assert_array_equal(outside, t_out)
        np.testing.assert_array_equal(nn_dist, t_dist)
        assert np.all(nn_dist[~outside] == 0.0)

    def test_distance_is_the_true_nearest_node_distance(self):
        pts = _pts(50, 33)
        rng = np.random.default_rng(34)
        u, v = rng.standard_normal(50), rng.standard_normal(50)
        query = np.array([[-30.0, 50.0], [150.0, 150.0], [50.0, -7.5]])
        _, nn_dist = self._check(pts, u, v, query)
        brute = np.hypot(
            query[:, None, 0] - pts[None, :, 0], query[:, None, 1] - pts[None, :, 1],
        ).min(axis=1)
        np.testing.assert_allclose(nn_dist, brute, rtol=1e-12, atol=0)

    def test_exact_tie_midpoints_on_a_lattice(self):
        pts, u, v, query = _lattice_case()
        outside, nn_dist = self._check(pts, u, v, query)
        t_out, t_dist = _hull_truth(pts, u, query)
        np.testing.assert_array_equal(outside, t_out)
        np.testing.assert_array_equal(nn_dist, t_dist)
        assert 0 < outside.sum() < len(query)

    def test_shared_nan_nodes(self):
        pts = _pts(60, 35)
        rng = np.random.default_rng(36)
        u, v = rng.standard_normal(60), rng.standard_normal(60)
        bad = rng.random(60) < 0.2
        u[bad] = np.nan
        v[bad] = np.nan
        query = _pts(50, 37) * 1.2 - 10.0
        outside, nn_dist = self._check(pts, u, v, query)
        t_out, t_dist = _hull_truth(pts, u, query)
        np.testing.assert_array_equal(outside, t_out)
        np.testing.assert_array_equal(nn_dist, t_dist)

    def test_tri_cache_reuse_keeps_values_and_hull(self):
        pts = _pts(70, 38)
        query = pts * 1.1 - 5.0
        rng = np.random.default_rng(39)
        cache: dict = {}
        for _ in range(3):
            u, v = rng.standard_normal(70), rng.standard_normal(70)
            outside, nn_dist = self._check(pts, u, v, query, tri_cache=cache)
            t_out, t_dist = _hull_truth(pts, u, query)
            np.testing.assert_array_equal(outside, t_out)
            np.testing.assert_array_equal(nn_dist, t_dist)
        assert len(cache) == 1

    def test_mismatched_nan_fallback_ors_the_two_hulls(self):
        pts = _pts(40, 40)
        rng = np.random.default_rng(41)
        u, v = rng.standard_normal(40), rng.standard_normal(40)
        # Drop the extreme nodes from one field each, so the two hulls differ.
        u[np.argmin(pts[:, 0])] = np.nan
        v[np.argmax(pts[:, 0])] = np.nan
        query = _pts(60, 42) * 1.2 - 10.0
        outside, nn_dist = self._check(pts, u, v, query)
        out_u, dist_u = _hull_truth(pts, u, query)
        out_v, dist_v = _hull_truth(pts, v, query)
        assert not np.array_equal(out_u, out_v), "fixture must split the hulls"
        np.testing.assert_array_equal(outside, out_u | out_v)
        np.testing.assert_array_equal(nn_dist, np.maximum(dist_u, dist_v))

    def test_all_invalid_gives_nan_and_nothing_outside(self):
        pts = _pts(20, 43)
        u = np.full(20, np.nan)
        v = np.full(20, np.nan)
        query = _pts(10, 44)
        outside, nn_dist = self._check(pts, u, v, query)
        assert not outside.any()
        assert np.all(nn_dist == 0.0)

    def test_nan_fill_leaves_outside_queries_nan(self):
        pts = _pts(40, 45)
        rng = np.random.default_rng(46)
        u, v = rng.standard_normal(40), rng.standard_normal(40)
        query = _pts(40, 47) * 1.3 - 15.0
        du, dv, outside, nn_dist = scattered_interpolant_uv_diag(
            pts, u, v, query, fill_method="nan",
        )
        t_out, _ = _hull_truth(pts, u, query)
        np.testing.assert_array_equal(outside, t_out)
        assert np.isnan(du[outside]).all() and np.isnan(dv[outside]).all()
        assert np.isfinite(du[~outside]).all()
        assert np.all(nn_dist == 0.0)

    def test_uv_wrapper_returns_the_diag_values(self):
        pts = _pts(60, 48)
        rng = np.random.default_rng(49)
        u, v = rng.standard_normal(60), rng.standard_normal(60)
        query = _pts(40, 50) * 1.3 - 15.0
        du, dv, _, _ = scattered_interpolant_uv_diag(pts, u, v, query)
        au, av = scattered_interpolant_uv(pts, u, v, query)
        np.testing.assert_array_equal(au, du)
        np.testing.assert_array_equal(av, dv)
