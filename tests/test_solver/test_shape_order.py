"""The second-order stage: guards, fallback and its per-node record.

Contract under test: a node either ends with a trusted second-order
result (order 2) or is exactly its first-order input (order 1), and a
node the first-order solve did not converge is never touched (order 0).
The stage never raises and never introduces a non-finite value.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from al_dic.solver import shape_order
from al_dic.solver.shape_order import ShapeOrderResult, refine_shape_order
from tests.shape_order_helpers import (
    first_order_solution, make_speckle, node_grid, warp_lagrangian,
)

H, W = 256, 384
WINSIZE = 32
CURV = 2e-4


def _u(x, y):
    return 2.3 + 0.5 * CURV * (x - W / 2) ** 2


def _v(x, y):
    return 1.7 + 0.0 * x


@pytest.fixture(scope="module")
def scene():
    ref = make_speckle(H, W, seed=5)
    deformed = warp_lagrangian(ref, _u, _v)
    coords = node_grid(H, W, step=16, margin=48)
    truth = np.column_stack([_u(coords[:, 0], coords[:, 1]), _v(coords[:, 0], coords[:, 1])])
    ctx, u_2d, f_2d, conv, tol = first_order_solution(ref, deformed, coords, WINSIZE, truth)
    return {"ctx": ctx, "U": u_2d, "F": f_2d, "conv": conv, "tol": tol,
            "g": deformed, "truth": truth}


def _run(scene, **kwargs):
    conv = kwargs.pop("conv", scene["conv"])
    u_in = kwargs.pop("U", scene["U"])
    return refine_shape_order(scene["ctx"], u_in, scene["F"], conv, scene["g"],
                              scene["tol"], **kwargs)


def test_refines_converged_nodes_and_reports_them(scene):
    out = _run(scene)

    assert isinstance(out, ShapeOrderResult)
    assert out.order.dtype == np.int8
    assert np.all(out.order == 2)
    err1 = np.abs(scene["U"][:, 0] - scene["truth"][:, 0]).mean()
    err2 = np.abs(out.U_2d[:, 0] - scene["truth"][:, 0]).mean()
    assert err2 < 0.35 * err1
    np.testing.assert_allclose(out.second[:, 0].mean(), CURV, rtol=0.03)
    assert np.all(np.isfinite(out.znssd))


def test_inputs_are_not_mutated(scene):
    u_before, f_before = scene["U"].copy(), scene["F"].copy()
    conv_before = scene["conv"].copy()
    _run(scene)
    np.testing.assert_array_equal(scene["U"], u_before)
    np.testing.assert_array_equal(scene["F"], f_before)
    np.testing.assert_array_equal(scene["conv"], conv_before)


def test_failed_first_order_nodes_stay_untouched(scene):
    conv = scene["conv"].copy()
    failed = np.zeros(conv.size, dtype=bool)
    failed[::5] = True
    conv[failed] = scene["ctx"].max_iter + 1

    out = _run(scene, conv=conv)

    assert np.all(out.order[failed] == 0)
    assert np.all(out.order[~failed] == 2)
    np.testing.assert_array_equal(out.U_2d[failed], scene["U"][failed])
    np.testing.assert_array_equal(out.F_2d[failed], scene["F"][failed])
    assert np.all(out.second[failed] == 0.0)
    assert np.all(out.status[failed] == shape_order.STATUS_NOT_RUN)


def test_nodes_that_move_too_far_equal_first_order_exactly(scene, monkeypatch):
    monkeypatch.setattr(shape_order, "MAX_SHIFT_SECOND", 0.0)
    out = _run(scene)

    assert np.all(out.order == 1)
    assert np.all(out.status == shape_order.STATUS_SHIFT_TOO_LARGE)
    np.testing.assert_array_equal(out.U_2d, scene["U"])
    np.testing.assert_array_equal(out.F_2d, scene["F"])
    assert np.all(out.second == 0.0)
    assert np.all(np.isnan(out.znssd))


def test_nodes_the_kernel_refuses_equal_first_order_exactly(scene, monkeypatch):
    monkeypatch.setattr(shape_order, "MIN_FILL_SECOND", 1.1)   # no window can pass
    out = _run(scene)

    assert np.all(out.order == 1)
    assert np.all(out.status == shape_order.STATUS_TOO_FEW_PIXELS)
    np.testing.assert_array_equal(out.U_2d, scene["U"])
    np.testing.assert_array_equal(out.F_2d, scene["F"])


def test_select_limits_the_nodes(scene):
    select = np.zeros(scene["conv"].size, dtype=bool)
    select[1::3] = True

    out = _run(scene, select=select)

    assert np.all(out.order[select] == 2)
    assert np.all(out.order[~select] == 1)
    np.testing.assert_array_equal(out.U_2d[~select], scene["U"][~select])


def test_kernel_exception_falls_back_to_first_order(scene, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(shape_order, "icgn2_refine_parallel", boom)
    with caplog.at_level(logging.WARNING, logger="al_dic.solver.shape_order"):
        out = _run(scene)

    assert np.all(out.order == 1)
    np.testing.assert_array_equal(out.U_2d, scene["U"])
    np.testing.assert_array_equal(out.F_2d, scene["F"])
    assert any("second-order" in r.getMessage() for r in caplog.records)


def test_never_introduces_a_non_finite_value(scene):
    u_in = scene["U"].copy()
    conv = scene["conv"].copy()
    bad = np.zeros(conv.size, dtype=bool)
    bad[3::7] = True
    u_in[bad] = np.nan                       # e.g. an unsolved node upstream
    conv[bad] = scene["ctx"].max_iter + 2

    out = _run(scene, U=u_in, conv=conv)

    np.testing.assert_array_equal(np.isnan(out.U_2d), np.isnan(u_in))
    assert np.all(np.isfinite(out.F_2d))
    assert np.all(np.isfinite(out.second))
    assert np.all(out.order[bad] == 0)
    assert np.all(out.order[~bad] == 2)


def test_nan_first_order_values_are_not_refined_even_if_marked_converged(scene):
    u_in = scene["U"].copy()
    u_in[0] = np.nan
    out = _run(scene, U=u_in)
    assert out.order[0] == 0
    assert np.all(np.isnan(out.U_2d[0]))


def test_windows_clipped_by_the_image_edge(monkeypatch):
    """The pipeline's own mesh keeps nodes away from the border, but an
    external mesh (pyALDIC-3D) need not."""
    ref = make_speckle(H, W, seed=5)
    deformed = warp_lagrangian(ref, _u, _v)
    coords = node_grid(H, W, step=8, margin=4)          # nodes 4 px from the border
    truth = np.column_stack([_u(coords[:, 0], coords[:, 1]), _v(coords[:, 0], coords[:, 1])])
    ctx, u_2d, f_2d, conv, tol = first_order_solution(ref, deformed, coords, WINSIZE, truth)

    half = WINSIZE // 2
    edge_dist = np.minimum.reduce([coords[:, 0], W - 1 - coords[:, 0],
                                   coords[:, 1], H - 1 - coords[:, 1]])
    clipped = edge_dist < half

    # Default guards: a clipped window is either refined or keeps its
    # first-order value; nothing becomes non-finite.
    out = refine_shape_order(ctx, u_2d, f_2d, conv, deformed, tol)
    assert np.any(clipped)
    np.testing.assert_array_equal(np.isfinite(out.U_2d), np.isfinite(u_2d))
    # (Windows that touch the border can also fail first order, when the
    # displacement carries their pixels out of the image; hence 2 * half.)
    assert np.mean(out.order[edge_dist > 2 * half] == 2) > 0.95
    kept = out.order == 1
    np.testing.assert_array_equal(out.U_2d[kept], u_2d[kept])

    # The fill guard itself: with a stricter minimum, a window the image
    # edge clips below it is not refined. Along one edge the window keeps
    # (half + 1 + d) / (2 half + 1) of its columns.
    monkeypatch.setattr(shape_order, "MIN_FILL_SECOND", 0.75)
    strict = refine_shape_order(ctx, u_2d, f_2d, conv, deformed, tol)
    too_close = edge_dist < 0.75 * (2 * half + 1) - half - 1
    assert np.any(too_close)
    assert np.all(strict.order[too_close] != 2)
