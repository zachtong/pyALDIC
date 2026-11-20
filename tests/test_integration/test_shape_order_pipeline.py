"""Second-order shape function through the full pipeline (Local DIC)."""

from __future__ import annotations

import numpy as np
import pytest

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.io.image_ops import compute_image_gradient
from al_dic.solver.local_icgn import local_icgn, local_icgn_full
from al_dic.utils.outlier_detection import detect_bad_points
from tests.shape_order_helpers import make_speckle, node_grid, warp_lagrangian

H, W = 256, 384
WINSIZE, STEP = 32, 8
CURV = 2e-4


def _u(x, y):
    return 2.3 + 0.5 * CURV * (x - W / 2) ** 2


def _v(x, y):
    return 1.7 + 0.0 * x


@pytest.fixture(scope="module")
def images():
    ref = make_speckle(H, W, seed=5)
    return ref, warp_lagrangian(ref, _u, _v)


def _para(**overrides):
    return dicpara_default(
        winsize=WINSIZE, winstepsize=STEP, use_global_step=False,
        reference_mode="accumulative",
        gridxy_roi_range=GridxyROIRange(gridx=(0, W - 1), gridy=(0, H - 1)),
        **overrides,
    )


def _run(images, **overrides):
    mask = np.ones((H, W))
    return run_aldic(_para(**overrides), list(images), [mask, mask], compute_strain=False)


def _interior(coords, margin=48):
    return ((coords[:, 0] > margin) & (coords[:, 0] < W - 1 - margin)
            & (coords[:, 1] > margin) & (coords[:, 1] < H - 1 - margin))


def test_second_order_removes_the_first_order_bias(images):
    first = _run(images)
    second = _run(images, shape_function_order="second")

    coords = first.dic_mesh.coordinates_fem
    np.testing.assert_array_equal(coords, second.dic_mesh.coordinates_fem)
    inner = _interior(coords)
    truth = _u(coords[:, 0], coords[:, 1])
    bias1 = np.mean(first.result_disp[0].U[0::2][inner] - truth[inner])
    bias2 = np.mean(second.result_disp[0].U[0::2][inner] - truth[inner])

    half = WINSIZE // 2
    assert bias1 == pytest.approx(half * (half + 1) / 6.0 * CURV, rel=0.2)
    assert abs(bias2) < 0.15 * abs(bias1)


def test_frame_result_carries_the_order_only_when_the_stage_ran(images):
    first = _run(images)
    second = _run(images, shape_function_order="second")

    assert first.result_disp[0].shape_order is None
    order = second.result_disp[0].shape_order
    n_nodes = second.dic_mesh.coordinates_fem.shape[0]
    assert order.dtype == np.int8 and order.shape == (n_nodes,)
    assert set(np.unique(order)) <= {0, 1, 2}
    inner = _interior(second.dic_mesh.coordinates_fem)
    assert np.mean(order[inner] == 2) > 0.95
    assert np.all(np.isfinite(second.result_disp[0].U))


def test_same_nodes_are_filled_with_and_without_the_stage(images):
    ref, deformed = images
    blank = ref.copy()
    blank[100:150, 150:230] = 0.5          # no texture: those subsets cannot converge
    coords = node_grid(H, W, step=STEP, margin=40)
    grads = compute_image_gradient(blank, np.ones_like(blank), img_raw=blank)
    u0 = np.column_stack([_u(coords[:, 0], coords[:, 1]),
                          _v(coords[:, 0], coords[:, 1])]).ravel()

    plain = local_icgn_full(u0, coords, grads, blank, deformed, _para(), _para().tol)
    staged = local_icgn_full(u0, coords, grads, blank, deformed,
                             _para(shape_function_order="second"), _para().tol)

    assert plain.bad_pt_num > 0
    assert staged.bad_pt_num == plain.bad_pt_num
    np.testing.assert_array_equal(staged.conv_iter, plain.conv_iter)
    # The filled set (hard failures plus slow-convergence outliers) comes from
    # the first-order iteration counts alone, so the stage cannot change it.
    filled, _ = detect_bad_points(
        plain.conv_iter, _para().icgn_max_iter, coords,
        sigma_factor=1.0, min_threshold=6,
    )
    np.testing.assert_array_equal(np.flatnonzero(staged.shape_order == 0), filled)
    assert np.all(np.isfinite(staged.U)) and np.all(np.isfinite(staged.F))
    # Away from the filled nodes the stage did refine.
    assert np.mean(staged.shape_order == 2) > 0.7


def test_local_icgn_wrapper_returns_the_legacy_tuple(images):
    ref, deformed = images
    coords = node_grid(H, W, step=16, margin=48)
    grads = compute_image_gradient(ref, np.ones_like(ref), img_raw=ref)
    u0 = np.column_stack([_u(coords[:, 0], coords[:, 1]),
                          _v(coords[:, 0], coords[:, 1])]).ravel()
    para = _para()

    legacy = local_icgn(u0, coords, grads, ref, deformed, para, para.tol)
    full = local_icgn_full(u0, coords, grads, ref, deformed, para, para.tol)

    assert len(legacy) == 6
    np.testing.assert_array_equal(legacy[0], full.U)
    np.testing.assert_array_equal(legacy[1], full.F)
    np.testing.assert_array_equal(legacy[3], full.conv_iter)
    assert legacy[4] == full.bad_pt_num
    np.testing.assert_array_equal(legacy[5], full.mark_hole_strain)
    assert full.shape_order is None
