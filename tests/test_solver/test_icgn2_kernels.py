"""Second-order (12-parameter) IC-GN refinement kernel.

The kernel starts from a converged first-order result and must:
recover a quadratic displacement that a first-order subset biases,
carry about twice the first-order random error, add no systematic error
on a field the first order already represents, refuse subsets it cannot
condition, and leave the input untouched whenever it refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from al_dic._numba_compat import HAS_NUMBA
from al_dic.solver.icgn2_kernels import (
    STATUS_ILL_CONDITIONED, STATUS_OK, STATUS_OUT_OF_IMAGE,
    STATUS_TOO_FEW_PIXELS, _refine_node, icgn2_refine_parallel,
)
from tests.shape_order_helpers import (
    add_noise, first_order_solution, make_speckle, node_grid, warp_lagrangian,
)

H, W = 256, 384
WINSIZE = 32
HALF = WINSIZE // 2
SHIFT = (2.3, 1.7)
TOL_PX = 1e-3
MAX_ITER = 100
MIN_FILL = 0.5
PIVOT_MIN = 1e-8


@pytest.fixture(scope="module")
def ref():
    return make_speckle(H, W, seed=5)


@pytest.fixture(scope="module")
def coords():
    return node_grid(H, W, step=16, margin=48)


def _start_params(u_2d, f_2d):
    """(N, 12) kernel parameters from a first-order result.

    Kernel layout: [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy];
    al_dic's F is [ux, vx, uy, vy].
    """
    p = np.zeros((u_2d.shape[0], 12))
    p[:, 0], p[:, 6] = u_2d[:, 0], u_2d[:, 1]
    p[:, 1], p[:, 7] = f_2d[:, 0], f_2d[:, 1]
    p[:, 2], p[:, 8] = f_2d[:, 2], f_2d[:, 3]
    return p


def _refine(ctx, p0, deformed, active=None, min_fill=MIN_FILL):
    pre = ctx.pre
    n = p0.shape[0]
    if active is None:
        active = np.ones(n, dtype=np.bool_)
    return icgn2_refine_parallel(
        np.round(ctx.coordinates_fem), p0, active,
        pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
        pre["XX_all"], pre["YY_all"], pre["meanf_all"], pre["bottomf_all"],
        deformed, TOL_PX, MAX_ITER, min_fill, PIVOT_MIN,
    )


def _solve_both(ref_img, deformed, coords, u_true, v_true):
    u0 = np.column_stack([u_true, v_true])
    ctx, u_2d, f_2d, conv, _ = first_order_solution(ref_img, deformed, coords, WINSIZE, u0)
    assert np.all(conv <= MAX_ITER)
    p0 = _start_params(u_2d, f_2d)
    p2, iters, status, znssd = _refine(ctx, p0, deformed)
    return ctx, p0, p2, iters, status, znssd


def test_quadratic_field_is_recovered(ref, coords):
    curv = 2e-4

    def u_fn(x, y):
        return SHIFT[0] + 0.5 * curv * (x - W / 2) ** 2

    def v_fn(x, y):
        return SHIFT[1] + 0.0 * x

    deformed = warp_lagrangian(ref, u_fn, v_fn)
    u_true = u_fn(coords[:, 0], coords[:, 1])
    v_true = v_fn(coords[:, 0], coords[:, 1])
    _, p0, p2, _, status, _ = _solve_both(ref, deformed, coords, u_true, v_true)

    assert np.all(status == STATUS_OK)
    # First order carries the undermatch bias M(M+1)/6 * laplacian(u).
    formula = HALF * (HALF + 1) / 6.0 * curv
    assert np.mean(p0[:, 0] - u_true) == pytest.approx(formula, rel=0.15)
    # Second order removes it and measures the curvature itself.
    assert abs(np.mean(p2[:, 0] - u_true)) < 5e-4
    assert np.sqrt(np.mean((p2[:, 0] - u_true) ** 2)) < 3e-3
    assert np.mean(p2[:, 3]) == pytest.approx(curv, rel=0.03)


def test_noise_ratio_is_about_two(ref, coords):
    def u_fn(x, y):
        return SHIFT[0] + 0.0 * x

    def v_fn(x, y):
        return SHIFT[1] + 0.0 * x

    clean = warp_lagrangian(ref, u_fn, v_fn)
    u_true = np.full(coords.shape[0], SHIFT[0])
    v_true = np.full(coords.shape[0], SHIFT[1])
    err1, err2 = [], []
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        f = add_noise(ref, 0.01, rng)
        g = add_noise(clean, 0.01, rng)
        _, p0, p2, _, status, _ = _solve_both(f, g, coords, u_true, v_true)
        ok = status == STATUS_OK
        assert ok.mean() > 0.95
        err1.append((p0[ok, 0] - SHIFT[0]))
        err2.append((p2[ok, 0] - SHIFT[0]))
    ratio = np.concatenate(err2).std() / np.concatenate(err1).std()
    assert 1.6 < ratio < 2.4


def test_no_systematic_error_on_an_affine_field(ref, coords):
    def u_fn(x, y):
        return SHIFT[0] + 0.01 * (x - W / 2) + 0.004 * (y - H / 2)

    def v_fn(x, y):
        return SHIFT[1] - 0.003 * (x - W / 2) + 0.006 * (y - H / 2)

    deformed = warp_lagrangian(ref, u_fn, v_fn)
    u_true = u_fn(coords[:, 0], coords[:, 1])
    v_true = v_fn(coords[:, 0], coords[:, 1])
    _, p0, p2, _, status, _ = _solve_both(ref, deformed, coords, u_true, v_true)

    assert np.all(status == STATUS_OK)
    assert abs(np.mean(p2[:, 0] - u_true)) < 1e-3
    assert abs(np.mean(p2[:, 6] - v_true)) < 1e-3
    rms1 = np.sqrt(np.mean((p0[:, 0] - u_true) ** 2))
    rms2 = np.sqrt(np.mean((p2[:, 0] - u_true) ** 2))
    assert rms2 < rms1 + 2e-3
    # The second-order terms of an affine field are zero.
    assert np.abs(p2[:, [3, 4, 5, 9, 10, 11]]).max() < 2e-4


def test_inactive_nodes_and_inputs_are_untouched(ref, coords):
    def u_fn(x, y):
        return SHIFT[0] + 1e-4 * (x - W / 2) ** 2

    def v_fn(x, y):
        return SHIFT[1] + 0.0 * x

    deformed = warp_lagrangian(ref, u_fn, v_fn)
    u0 = np.column_stack([u_fn(coords[:, 0], coords[:, 1]),
                          v_fn(coords[:, 0], coords[:, 1])])
    ctx, u_2d, f_2d, _, _ = first_order_solution(ref, deformed, coords, WINSIZE, u0)
    p0 = _start_params(u_2d, f_2d)
    snapshot = p0.copy()
    active = np.zeros(p0.shape[0], dtype=np.bool_)
    active[::2] = True

    p2, iters, status, _ = _refine(ctx, p0, deformed, active=active)

    np.testing.assert_array_equal(p0, snapshot)
    np.testing.assert_array_equal(p2[~active], p0[~active])
    assert np.all(status[~active] == -1)
    assert np.all(iters[~active] == 0)
    assert np.any(p2[active, 3] != 0.0)


def _one_node_arrays(ref_img, coords):
    """Context arrays of one interior node, for direct kernel calls."""
    u0 = np.zeros((coords.shape[0], 2))
    ctx, _, _, _, _ = first_order_solution(ref_img, ref_img, coords, WINSIZE, u0)
    i = coords.shape[0] // 2
    pre = ctx.pre
    arrays = [pre[k][i].copy() for k in
              ("ref_all", "gx_all", "gy_all", "mask_all", "XX_all", "YY_all")]
    return (float(coords[i, 0]), float(coords[i, 1]), arrays,
            float(pre["meanf_all"][i]), float(pre["bottomf_all"][i]))


def _call_node(x0, y0, p, arrays, meanf, bottomf, img, min_fill=MIN_FILL, fn=_refine_node):
    ref_sub, gx_sub, gy_sub, bw, xx, yy = arrays
    return fn(x0, y0, p, ref_sub, gx_sub, gy_sub, bw, xx, yy, meanf, bottomf,
              img, TOL_PX, MAX_ITER, min_fill, PIVOT_MIN)


def test_guard_too_few_connected_pixels(ref, coords):
    x0, y0, arrays, meanf, bottomf = _one_node_arrays(ref, coords)
    arrays[3][:, : int(0.4 * arrays[3].shape[1])] = 0.0     # 60 % of the window left
    p = np.zeros(12)
    before = p.copy()
    _, status, _ = _call_node(x0, y0, p, arrays, meanf, bottomf, ref, min_fill=0.75)
    assert status == STATUS_TOO_FEW_PIXELS
    np.testing.assert_array_equal(p, before)


def test_guard_ill_conditioned_strip(ref, coords):
    x0, y0, arrays, meanf, bottomf = _one_node_arrays(ref, coords)
    bw = arrays[3]
    bw[:] = 0.0
    bw[:, bw.shape[1] // 2] = 1.0        # one pixel wide: no x-dependence can be fitted
    p = np.zeros(12)
    before = p.copy()
    _, status, _ = _call_node(x0, y0, p, arrays, meanf, bottomf, ref, min_fill=0.0)
    assert status == STATUS_ILL_CONDITIONED
    np.testing.assert_array_equal(p, before)


def test_guard_warp_leaves_the_image(ref, coords):
    x0, y0, arrays, meanf, bottomf = _one_node_arrays(ref, coords)
    p = np.zeros(12)
    p[0] = 10_000.0
    before = p.copy()
    _, status, _ = _call_node(x0, y0, p, arrays, meanf, bottomf, ref)
    assert status == STATUS_OUT_OF_IMAGE
    np.testing.assert_array_equal(p, before)


@pytest.mark.skipif(not HAS_NUMBA, reason="compares the compiled kernel with its Python source")
def test_compiled_kernel_matches_plain_python(coords):
    small = 16
    ref_img = make_speckle(96, 96, seed=9)

    def u_fn(x, y):
        return 0.6 + 2e-4 * (x - 48) ** 2

    def v_fn(x, y):
        return -0.4 + 0.002 * (y - 48)

    deformed = warp_lagrangian(ref_img, u_fn, v_fn)
    nodes = node_grid(96, 96, step=8, margin=24)
    u0 = np.column_stack([u_fn(nodes[:, 0], nodes[:, 1]), v_fn(nodes[:, 0], nodes[:, 1])])
    ctx, u_2d, f_2d, _, _ = first_order_solution(ref_img, deformed, nodes, small, u0)
    p0 = _start_params(u_2d, f_2d)
    pre = ctx.pre
    for i in (0, nodes.shape[0] // 2, nodes.shape[0] - 1):
        arrays = [pre[k][i] for k in
                  ("ref_all", "gx_all", "gy_all", "mask_all", "XX_all", "YY_all")]
        meanf, bottomf = float(pre["meanf_all"][i]), float(pre["bottomf_all"][i])
        p_jit, p_py = p0[i].copy(), p0[i].copy()
        out_jit = _call_node(nodes[i, 0], nodes[i, 1], p_jit, arrays, meanf, bottomf, deformed)
        out_py = _call_node(nodes[i, 0], nodes[i, 1], p_py, arrays, meanf, bottomf, deformed,
                            fn=_refine_node.py_func)
        assert out_jit[0] == out_py[0] and out_jit[1] == out_py[1] == STATUS_OK
        np.testing.assert_allclose(p_jit, p_py, rtol=0, atol=1e-10)
