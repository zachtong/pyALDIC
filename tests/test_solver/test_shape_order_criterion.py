"""Per-node prediction of the better shape-function order."""

from __future__ import annotations

import numpy as np
import pytest

from al_dic.solver.shape_order_criterion import (
    fit_displacement_derivatives, fit_radius, predict_shape_order,
)
from tests.shape_order_helpers import (
    add_noise, first_order_solution, make_speckle, node_grid, warp_lagrangian,
)

H, W = 288, 416
WINSIZE = 32
STEP = 8


def _grid(step=STEP, margin=24):
    return node_grid(H, W, step=step, margin=margin)


def test_fit_recovers_analytic_derivatives():
    coords = _grid()
    x, y = coords[:, 0] - W / 2, coords[:, 1] - H / 2
    cu, cv = 3e-8, -2e-8
    disp = np.column_stack([
        1.0 + 2e-4 * x * x + 1e-4 * y * y + cu * x ** 4 + 5e-9 * x * x * y * y,
        -0.5 + 3e-4 * x * y + cv * y ** 4,
    ])
    usable = np.ones(coords.shape[0], dtype=bool)

    lap, quart = fit_displacement_derivatives(coords, disp, usable, radius=48.0)

    inner = (np.abs(x) < W / 2 - 80) & (np.abs(y) < H / 2 - 80)
    lap_u = 4e-4 + 2e-4 + 12 * cu * x * x + 1e-8 * (x * x + y * y)
    np.testing.assert_allclose(lap[inner, 0], lap_u[inner], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(lap[inner, 1], 12 * cv * y[inner] ** 2, atol=1e-10)
    # 9 (u_xxxx + u_yyyy) + 70 u_xxyy
    np.testing.assert_allclose(quart[inner, 0], 9 * 24 * cu + 70 * 4 * 5e-9, rtol=1e-6)
    np.testing.assert_allclose(quart[inner, 1], 9 * 24 * cv, rtol=1e-6)


def test_node_with_too_few_neighbours_is_not_fitted():
    coords = _grid()
    disp = np.zeros((coords.shape[0], 2))
    usable = np.zeros(coords.shape[0], dtype=bool)
    usable[:12] = True                              # a dozen nodes: fewer than a fit needs

    lap, quart = fit_displacement_derivatives(coords, disp, usable, radius=48.0)

    assert np.all(np.isnan(lap)) and np.all(np.isnan(quart))


def test_fit_does_not_look_across_a_masked_slit():
    coords = _grid(step=8, margin=24)
    mask = np.ones((H, W))
    slit_x = 204
    mask[:, slit_x - 1:slit_x + 2] = 0.0             # full-height slit between node columns
    usable = mask[coords[:, 1].astype(int), coords[:, 0].astype(int)] > 0.5
    # Rigid motion on each side, a jump across the slit: no curvature anywhere.
    disp = np.column_stack([np.where(coords[:, 0] < slit_x, 0.0, 3.0),
                            np.zeros(coords.shape[0])])

    blind, _ = fit_displacement_derivatives(coords, disp, usable, radius=48.0)
    aware, _ = fit_displacement_derivatives(coords, disp, usable, radius=48.0, mask=mask)

    near = usable & (np.abs(coords[:, 0] - slit_x) < 30) & (coords[:, 1] > 80) \
        & (coords[:, 1] < H - 80)
    assert np.nanmax(np.abs(blind[near, 0])) > 1e-3          # the jump reads as curvature
    # Each side is fitted on its own: where a fit exists it sees no
    # curvature, and nodes right at the slit (one-sided) are not fitted.
    fitted = usable & np.isfinite(aware[:, 0])
    assert fitted.sum() > 100
    assert np.nanmax(np.abs(aware[fitted, 0])) < 1e-8
    assert not np.any(np.isfinite(aware[near & (np.abs(coords[:, 0] - slit_x) < 8), 0]))


def test_one_sided_neighbourhood_is_not_fitted():
    """At the edge of the node set the fit would extrapolate; its noisy
    derivatives would read as bias, so those nodes are left unfitted."""
    coords = _grid()
    rng = np.random.default_rng(3)
    disp = rng.standard_normal((coords.shape[0], 2)) * 0.02     # noise only
    usable = np.ones(coords.shape[0], dtype=bool)

    lap, _ = fit_displacement_derivatives(coords, disp, usable, radius=48.0)

    edge_dist = np.minimum.reduce([coords[:, 0] - coords[:, 0].min(),
                                   coords[:, 0].max() - coords[:, 0],
                                   coords[:, 1] - coords[:, 1].min(),
                                   coords[:, 1].max() - coords[:, 1]])
    assert np.all(np.isnan(lap[edge_dist < 8, 0]))            # outermost rows
    assert np.all(np.isfinite(lap[edge_dist >= 48, 0]))       # full neighbourhood


def test_fit_radius_grows_with_subset_and_step():
    assert fit_radius(32, 8) == pytest.approx(1.0 * 33)
    assert fit_radius(64, 8) == pytest.approx(1.0 * 65)
    assert fit_radius(16, 16) == pytest.approx(4.0 * 16)


def _predict(u_fn, v_fn, noise, seed=5, winsize=WINSIZE):
    ref = make_speckle(H, W, seed=seed)
    clean = warp_lagrangian(ref, u_fn, v_fn)
    rng = np.random.default_rng(seed + 100)
    f, g = add_noise(ref, noise, rng), add_noise(clean, noise, rng)
    coords = node_grid(H, W, step=STEP, margin=winsize // 2 + 8)
    start = np.round(np.column_stack([u_fn(coords[:, 0], coords[:, 1]),
                                      v_fn(coords[:, 0], coords[:, 1])]))
    ctx, u_2d, f_2d, conv, _ = first_order_solution(f, g, coords, winsize, start)
    return coords, predict_shape_order(ctx, u_2d, f_2d, conv, g, STEP)


def test_uniform_strain_selects_no_second_order():
    coords, pred = _predict(lambda x, y: 2.3 + 0.01 * (x - W / 2),
                            lambda x, y: 1.7 - 0.004 * (y - H / 2), noise=0.01)

    assert pred.fitted.mean() > 0.6
    assert pred.select.mean() < 0.01
    assert pred.noise_sd == pytest.approx(0.01, rel=0.3)


def test_strong_curvature_is_selected_and_quiet_region_is_not():
    # Curvature in the left half only (smoothly switched on).
    def u_fn(x, y):
        weight = 0.5 * (1.0 - np.tanh((x - W / 2) / 25.0))
        return 2.3 + weight * 0.35 * np.cos(2 * np.pi * (y - H / 2) / 70.0)

    coords, pred = _predict(u_fn, lambda x, y: 1.7 + 0.0 * x, noise=0.005)

    inner = (coords[:, 1] > 70) & (coords[:, 1] < H - 70)
    left = inner & (coords[:, 0] > 70) & (coords[:, 0] < W / 2 - 60)
    right = inner & (coords[:, 0] > W / 2 + 60) & (coords[:, 0] < W - 70)
    assert pred.select[left].mean() > 0.4
    assert pred.select[right].mean() < 0.03
    # The predicted first-order bias is of the size of the true one.
    assert 0.02 < np.nanmedian(pred.bias_first[left]) < 0.3


def test_unsolved_nodes_are_never_selected():
    ref = make_speckle(H, W, seed=5)

    def u_fn(x, y):
        return 2.3 + 0.35 * np.cos(2 * np.pi * (y - H / 2) / 70.0)

    def v_fn(x, y):
        return 1.7 + 0.0 * x

    g = warp_lagrangian(ref, u_fn, v_fn)
    coords = node_grid(H, W, step=STEP, margin=24)
    start = np.round(np.column_stack([u_fn(coords[:, 0], coords[:, 1]),
                                      v_fn(coords[:, 0], coords[:, 1])]))
    ctx, u_2d, f_2d, conv, _ = first_order_solution(ref, g, coords, WINSIZE, start)
    conv = conv.copy()
    failed = np.zeros(conv.size, dtype=bool)
    failed[::4] = True
    conv[failed] = ctx.max_iter + 1

    pred = predict_shape_order(ctx, u_2d, f_2d, conv, g, STEP)

    assert not np.any(pred.select[failed])
    assert not np.any(pred.fitted[failed])
    assert np.any(pred.select[~failed])


def test_scatter_the_image_noise_does_not_explain_is_not_read_as_curvature():
    """Real images: the measured displacement carries systematic scatter
    (interpolation bias, lens distortion) far above the image-noise
    prediction. A fit through it shows curvature, but none that stands out
    from the scatter itself, so it must (almost) never select second order."""
    ref = make_speckle(H, W, seed=5)

    def u_fn(x, y):
        return 2.3 + 0.01 * (x - W / 2)

    def v_fn(x, y):
        return 1.7 - 0.004 * (y - H / 2)

    rng = np.random.default_rng(105)
    f = add_noise(ref, 0.002, rng)
    g = add_noise(warp_lagrangian(ref, u_fn, v_fn), 0.002, rng)
    coords = node_grid(H, W, step=STEP, margin=WINSIZE // 2 + 8)
    start = np.round(np.column_stack([u_fn(coords[:, 0], coords[:, 1]),
                                      v_fn(coords[:, 0], coords[:, 1])]))
    ctx, u_2d, f_2d, conv, _ = first_order_solution(f, g, coords, WINSIZE, start)
    # Spatially correlated scatter of 0.01 px (correlation length about one
    # node step), as a sub-pixel bias would leave in the result.
    from scipy.ndimage import gaussian_filter

    nx, ny = np.unique(coords[:, 0]).size, np.unique(coords[:, 1]).size
    scatter = [gaussian_filter(rng.standard_normal((ny, nx)), 1.0) for _ in range(2)]
    scatter = np.column_stack([0.01 * s.ravel() / s.std() for s in scatter])
    u_2d = u_2d + scatter

    pred = predict_shape_order(ctx, u_2d, f_2d, conv, g, STEP)

    ok = pred.fitted & np.isfinite(pred.sigma_first)
    published_rule = pred.bias_first ** 2 > pred.bias_second ** 2 + 3.0 * pred.sigma_first ** 2
    looks_curved = published_rule[ok].mean()
    assert looks_curved > 0.2                        # the scatter does look like curvature
    assert pred.select[ok].mean() < 0.06             # and is rarely taken for it
    assert pred.select[ok].mean() < 0.25 * looks_curved


def test_dense_mesh_fit_uses_a_thinned_set_and_agrees(monkeypatch):
    """Above MAX_FIT_NEIGHBORS nodes per fit the fit takes an evenly thinned
    set: the neighbour lists stay bounded and the selection barely moves."""
    from al_dic.solver import shape_order_criterion as criterion

    def u_fn(x, y):
        weight = 0.5 * (1.0 - np.tanh((x - W / 2) / 25.0))
        return 2.3 + weight * 0.35 * np.cos(2 * np.pi * (y - H / 2) / 70.0)

    coords, full = _predict(u_fn, lambda x, y: 1.7 + 0.0 * x, noise=0.005)
    assert full.spacing == STEP

    monkeypatch.setattr(criterion, "MAX_FIT_NEIGHBORS", 40)      # about 53 per fit here
    usable = full.fitted | True
    indptr, _, cell = criterion._fit_neighbors(coords, usable, full.radius)
    assert cell > STEP
    assert np.diff(indptr).max() <= 1.6 * 40
    _, thinned = _predict(u_fn, lambda x, y: 1.7 + 0.0 * x, noise=0.005)

    assert thinned.spacing == pytest.approx(cell)
    both = full.fitted & thinned.fitted
    assert both.mean() > 0.5
    assert (full.select[both] == thinned.select[both]).mean() > 0.9
