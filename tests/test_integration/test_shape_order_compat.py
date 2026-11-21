"""The second-order stage next to every other feature that touches local IC-GN.

Each test runs the full pipeline with ``shape_function_order="second"``
(Local DIC) in one existing mode and checks that the mode still works,
that the per-node order is recorded, and that no value is non-finite.
"""

from __future__ import annotations

import numpy as np
import pytest

import al_dic.core.pipeline as pipeline_mod
from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from al_dic.mesh.refinement import RefinementPolicy
from al_dic.solver.seed_propagation import Seed, SeedSet
from al_dic.solver.shape_order import MIN_FILL_SECOND
from tests.shape_order_helpers import make_speckle, warp_lagrangian

H, W = 224, 320
WINSIZE = 32
HALF = WINSIZE // 2
CURV = 2e-4


def _u(scale=1.0):
    return lambda x, y: scale * (2.3 + 0.5 * CURV * (x - W / 2) ** 2)


def _v(scale=1.0):
    return lambda x, y: scale * 1.7 + 0.0 * x


@pytest.fixture(scope="module")
def frames():
    ref = make_speckle(H, W, seed=21)
    return [ref] + [warp_lagrangian(ref, _u(s), _v(s)) for s in (1.0, 2.0)]


def _para(step=8, **overrides):
    kwargs = dict(
        winsize=WINSIZE, winstepsize=step, winsize_min=min(8, step),
        use_global_step=False, shape_function_order="second",
        reference_mode="accumulative",
        gridxy_roi_range=GridxyROIRange(gridx=(0, W - 1), gridy=(0, H - 1)),
    )
    kwargs.update(overrides)
    return dicpara_default(**kwargs)


def _masks(n, mask=None):
    return [np.ones((H, W)) if mask is None else mask for _ in range(n)]


def _order(result, frame=0):
    order = result.result_disp[frame].shape_order
    assert order is not None and order.dtype == np.int8
    return order


def _assert_finite(result):
    """The per-pair displacement the solver returns is finite everywhere."""
    for fr in result.result_disp:
        assert np.all(np.isfinite(fr.U))


def _assert_same_untracked_nodes(result, first_order_result):
    """Composition marks nodes inside masked regions as untracked (NaN in
    ``U_accum``). The stage must not add or remove any."""
    for a, b in zip(result.result_disp, first_order_result.result_disp):
        np.testing.assert_array_equal(np.isnan(a.U_accum), np.isnan(b.U_accum))


def _interior(coords, margin=48):
    return ((coords[:, 0] > margin) & (coords[:, 0] < W - 1 - margin)
            & (coords[:, 1] > margin) & (coords[:, 1] < H - 1 - margin))


def _bias(result, scale=1.0, frame=0, select=None):
    coords = result.dic_mesh.coordinates_fem
    fr = result.result_disp[frame]
    disp = fr.U_accum if fr.U_accum is not None else fr.U
    keep = _interior(coords) if select is None else select
    return float(np.mean(disp[0::2][keep] - _u(scale)(coords[keep, 0], coords[keep, 1])))


FORMULA_BIAS = HALF * (HALF + 1) / 6.0 * CURV


def test_window_splitting_by_a_hole_and_a_slit(frames):
    mask = np.ones((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    mask[(xx - 110) ** 2 + (yy - 112) ** 2 < 26 ** 2] = 0.0       # hole
    mask[70:160, 228:231] = 0.0                                    # 3 px slit
    result = run_aldic(_para(), frames[:2], _masks(2, mask), compute_strain=False)
    first = run_aldic(_para(shape_function_order=None), frames[:2], _masks(2, mask),
                      compute_strain=False)

    _assert_finite(result)
    _assert_same_untracked_nodes(result, first)
    order = _order(result)
    coords = np.round(result.dic_mesh.coordinates_fem).astype(int)
    # Connected fraction of each node's window, measured independently here.
    from scipy.ndimage import label
    fill = np.zeros(coords.shape[0])
    for i, (x, y) in enumerate(coords):
        ys, xs = np.mgrid[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
        inside = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        patch = np.zeros(inside.shape)
        patch[inside] = mask[ys[inside], xs[inside]]
        labels, _ = label(patch > 0.5)
        centre = labels[HALF, HALF]
        fill[i] = 0.0 if centre == 0 else np.mean(labels == centre)

    # A window the mask cuts below the minimum fill is never refined.
    assert np.all(order[fill < MIN_FILL_SECOND - 1e-9] != 2)
    # Windows cut by the hole or the slit are refined too, and stay accurate.
    cut = (fill < 1.0) & (fill >= MIN_FILL_SECOND) & (order == 2)
    assert cut.sum() > 20
    truth_cut = _u()(result.dic_mesh.coordinates_fem[cut, 0],
                     result.dic_mesh.coordinates_fem[cut, 1])
    assert np.abs(result.result_disp[0].U[0::2][cut] - truth_cut).max() < 0.1
    # Whole windows away from the mask are refined and lose the bias.
    whole = (fill == 1.0) & _interior(result.dic_mesh.coordinates_fem)
    assert np.mean(order[whole] == 2) > 0.95
    assert abs(_bias(result, select=whole & (order == 2))) < 0.2 * FORMULA_BIAS


@pytest.mark.parametrize("step", [4, 8, 16])
def test_node_steps(frames, step):
    first = run_aldic(_para(step, shape_function_order=None), frames[:2], _masks(2),
                      compute_strain=False)
    second = run_aldic(_para(step), frames[:2], _masks(2), compute_strain=False)

    _assert_finite(second)
    assert _bias(first) == pytest.approx(FORMULA_BIAS, rel=0.25)
    assert abs(_bias(second)) < 0.2 * FORMULA_BIAS


def test_three_frames_with_previous_frame_initial_guess(frames):
    para = _para(init_guess_mode="previous")
    result = run_aldic(para, frames, _masks(3), compute_strain=False)

    _assert_finite(result)
    assert len(result.result_disp) == 2
    for frame, scale in enumerate((1.0, 2.0)):
        assert np.mean(_order(result, frame)[_interior(result.dic_mesh.coordinates_fem)] == 2) > 0.95
        assert abs(_bias(result, scale, frame)) < 0.2 * scale * FORMULA_BIAS


def test_incremental_mode_with_reference_update(frames):
    para = _para(reference_mode="incremental")
    result = run_aldic(para, frames, _masks(3), compute_strain=False)

    _assert_finite(result)
    assert len(result.result_disp) == 2
    for frame in range(2):
        order = _order(result, frame)
        pair_mesh = result.result_fe_mesh_each_frame[frame]
        assert order.shape == (pair_mesh.coordinates_fem.shape[0],)
        assert np.mean(order == 2) > 0.5
    # Composition reads U only; the cumulative field of frame 2 is close to
    # the true total displacement (twice the first increment).
    assert abs(_bias(result, 2.0, 1)) < 0.05


def test_seed_propagation_initial_guess(frames):
    probe = run_aldic(_para(shape_function_order=None), frames[:2], _masks(2),
                      compute_strain=False)
    coords = probe.dic_mesh.coordinates_fem
    centre = int(np.argmin(np.hypot(coords[:, 0] - W / 2, coords[:, 1] - H / 2)))
    seed_set = SeedSet(seeds=(Seed(node_idx=centre, region_id=0),), ncc_threshold=0.3)

    result = run_aldic(
        _para(init_guess_mode="seed_propagation", seed_set=seed_set),
        frames[:2], _masks(2), compute_strain=False,
    )

    _assert_finite(result)
    assert not result.stopped_early
    assert np.mean(_order(result)[_interior(coords)] == 2) > 0.95
    assert abs(_bias(result)) < 0.2 * FORMULA_BIAS


def test_mesh_refinement_with_hanging_nodes(frames):
    mask = np.ones((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    mask[(xx - 160) ** 2 + (yy - 112) ** 2 < 30 ** 2] = 0.0
    policy = RefinementPolicy(pre_solve=[MaskBoundaryCriterion(min_element_size=4)])
    para = _para(16, winsize_min=4)

    result = run_aldic(para, frames[:2], _masks(2, mask), compute_strain=False,
                       refinement_policy=policy)
    first = run_aldic(_para(16, winsize_min=4, shape_function_order=None), frames[:2],
                      _masks(2, mask), compute_strain=False, refinement_policy=policy)

    _assert_finite(result)
    _assert_same_untracked_nodes(result, first)
    mesh = result.result_fe_mesh_each_frame[0]
    assert np.any(mesh.elements_fem[:, 4:] >= 0), "refinement should create hanging nodes"
    order = _order(result)
    assert order.shape == (mesh.coordinates_fem.shape[0],)
    assert np.any(order == 2) and np.any(order != 2)


def test_external_mesh(frames):
    probe = run_aldic(_para(shape_function_order=None), frames[:2], _masks(2),
                      compute_strain=False)
    mesh = probe.dic_mesh
    u0 = np.zeros(2 * mesh.coordinates_fem.shape[0])
    u0[0::2], u0[1::2] = 2.0, 2.0

    result = run_aldic(_para(), frames[:2], _masks(2), compute_strain=False,
                       mesh=mesh, U0=u0)

    _assert_finite(result)
    assert _order(result).shape == (mesh.coordinates_fem.shape[0],)
    assert abs(_bias(result)) < 0.2 * FORMULA_BIAS


def test_strain_is_computed_from_the_refined_field(frames):
    result = run_aldic(_para(), frames[:2], _masks(2), compute_strain=True)

    _assert_finite(result)
    strain = result.result_strain[0]
    coords = result.dic_mesh.coordinates_fem
    inner = _interior(coords, margin=64)
    exx_true = CURV * (coords[inner, 0] - W / 2)
    assert np.all(np.isfinite(strain.strain_exx[inner]))
    np.testing.assert_allclose(strain.strain_exx[inner], exx_true, atol=1.5e-3)


def test_context_cache_capacity_does_not_change_results(frames, monkeypatch):
    para = _para(reference_mode="incremental")

    def run(capacity):
        monkeypatch.setattr(pipeline_mod, "_PRECOMPUTE_CACHE_SIZE", capacity)
        return run_aldic(para, frames, _masks(3), compute_strain=False)

    small, large = run(1), run(1000)
    for a, b in zip(small.result_disp, large.result_disp):
        np.testing.assert_array_equal(a.U, b.U)
        np.testing.assert_array_equal(a.shape_order, b.shape_order)


@pytest.mark.parametrize("scenario", ["masked", "incremental", "seed_propagation", "refinement"])
def test_auto_mode_runs_in_every_scenario(frames, scenario):
    """Auto goes through the same paths as a forced second order."""
    mask = None
    kwargs = {}
    para_kwargs = {"shape_function_order": "auto"}
    n_frames = 2
    if scenario == "masked":
        mask = np.ones((H, W))
        yy, xx = np.mgrid[0:H, 0:W]
        mask[(xx - 110) ** 2 + (yy - 112) ** 2 < 26 ** 2] = 0.0
        mask[70:160, 228:231] = 0.0
    elif scenario == "incremental":
        para_kwargs["reference_mode"] = "incremental"
        n_frames = 3
    elif scenario == "seed_propagation":
        probe = run_aldic(_para(shape_function_order=None), frames[:2], _masks(2),
                          compute_strain=False)
        coords = probe.dic_mesh.coordinates_fem
        centre = int(np.argmin(np.hypot(coords[:, 0] - W / 2, coords[:, 1] - H / 2)))
        para_kwargs["init_guess_mode"] = "seed_propagation"
        para_kwargs["seed_set"] = SeedSet(
            seeds=(Seed(node_idx=centre, region_id=0),), ncc_threshold=0.3)
    else:
        mask = np.ones((H, W))
        yy, xx = np.mgrid[0:H, 0:W]
        mask[(xx - 160) ** 2 + (yy - 112) ** 2 < 30 ** 2] = 0.0
        kwargs["refinement_policy"] = RefinementPolicy(
            pre_solve=[MaskBoundaryCriterion(min_element_size=4)])
        para_kwargs["winsize_min"] = 4

    step = 16 if scenario == "refinement" else 8
    result = run_aldic(_para(step, **para_kwargs), frames[:n_frames],
                       _masks(n_frames, mask), compute_strain=False, **kwargs)

    _assert_finite(result)
    assert len(result.result_disp) == n_frames - 1
    for frame in range(n_frames - 1):
        order = _order(result, frame)
        assert set(np.unique(order)) <= {0, 1, 2}

