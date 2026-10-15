"""Each incremental reference frame is solved on a grid that covers its ROI.

Plan step 2.1 (layer 1). On the drift sequence (``_grid_drift.py``) the
specimen leaves the frame-0 grid range, so with one frame-0 grid the
cumulative displacement drifts (about 1 px by frame 3, 28 px by frame 5).
``run_aldic`` now plans a phase-aligned grid per reference frame whose mask
leaves that range (``core/_grid_extent.py``) and switches lattices between
pairs. These tests pin:

- accuracy at every frame, and K-pair meshes that cover their mask tightly;
- byte-identity of ref-0 pairs, the canonical mesh and the reported range;
- byte-identity of every run where no reference needs a larger grid;
- that an external mesh is never replaced, and that the mesh is trimmed
  again after every lattice switch;
- that the switch works with seed propagation, refinement, every-N
  schedules and the pyramid search.

"Identity" runs replace the planner, at its import site in
``al_dic.core.pipeline``, with one that keeps the frame-0 grid for every
reference frame (the behavior before this change). Each configuration runs
once per module.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import numpy as np
import pytest

import al_dic.core.pipeline as pipeline_mod
from al_dic.core._grid_extent import PAD_STEPS, ReferenceGrids, plan_reference_grids
from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICPara,
    FrameSchedule,
    GridxyROIRange,
    PipelineResult,
)
from al_dic.core.pipeline import run_aldic
from al_dic.io.image_ops import compute_clamped_roi
from al_dic.mesh.mark_bridging import trimmed_keep_indices
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.mesh.refinement import build_refinement_policy
from al_dic.solver.seed_prop_pipeline import build_grid_for_roi
from al_dic.solver.seed_propagation import PartialResultError, Seed, SeedSet
from al_dic.utils.region_analysis import precompute_node_regions

from tests.conftest import apply_displacement_lagrangian, generate_speckle
from tests.test_integration._grid_drift import (
    CX,
    CY,
    H,
    HW,
    INTERIOR_MEDIAN_PX,
    N_DEF,
    SEARCH,
    STEP,
    W,
    WIN,
    X0,
    X1,
    Y0,
    Y1,
    assert_frame_accuracy,
    bbox,
    drift_data,
    interior_median,
    motion,
    range_of,
)

FRAMES = range(1, N_DEF + 1)


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------


class Run(NamedTuple):
    result: PipelineResult
    n_mesh_setup: int


def _identity_planner(roi0, *_args, **_kwargs) -> ReferenceGrids:
    """Keep the frame-0 grid for every reference frame."""
    return ReferenceGrids(roi0=roi0)


def run_counted(para, imgs, masks, identity=False, **kwargs) -> Run:
    """``run_aldic`` with a spy on ``mesh_setup`` (and the identity planner)."""
    calls: list[int] = []
    real_setup = pipeline_mod.mesh_setup

    def spy(*args, **kw):
        calls.append(1)
        return real_setup(*args, **kw)

    kwargs.setdefault("compute_strain", False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline_mod, "mesh_setup", spy)
        if identity:
            mp.setattr(pipeline_mod, "plan_reference_grids", _identity_planner)
        result = run_aldic(para, imgs, masks, **kwargs)
    return Run(result, len(calls))


def drift_para(roi: GridxyROIRange, **overrides) -> DICPara:
    kw = dict(
        winsize=WIN, winstepsize=STEP, winsize_min=16, img_size=(H, W),
        gridxy_roi_range=roi, reference_mode="incremental",
        size_of_fft_search_region=SEARCH, init_guess_mode="fft",
        show_plots=False,
    )
    kw.update(overrides)
    return dicpara_default(**kw)


# ---------------------------------------------------------------------------
# Comparing results
# ---------------------------------------------------------------------------


def _ndarray_fields(obj, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}/{f.name}": getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if isinstance(getattr(obj, f.name), np.ndarray)
    }


def result_arrays(result: PipelineResult) -> dict[str, np.ndarray]:
    """Every array a run returns, keyed by where it lives."""
    out = _ndarray_fields(result.dic_mesh, "canonical")
    for name in ("result_disp", "result_def_grad", "result_strain",
                 "result_fe_mesh_each_frame"):
        for i, item in enumerate(getattr(result, name)):
            out.update(_ndarray_fields(item, f"{name}[{i}]"))
    roi = result.dic_para.gridxy_roi_range
    out["dic_para/gridxy_roi_range"] = np.array(roi.gridx + roi.gridy)
    out["dic_para/img_size"] = np.array(result.dic_para.img_size)
    out["ref_switch_frames"] = np.array(result.ref_switch_frames)
    return out


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def assert_identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    differ = [k for k in a if not _same(a[k], b[k])]
    assert not differ, f"arrays differ: {differ}"


def ref0_pair_arrays(result: PipelineResult) -> dict[str, np.ndarray]:
    """Arrays of the canonical mesh and of every pair whose reference is 0."""
    out = _ndarray_fields(result.dic_mesh, "canonical")
    for i, ref in enumerate(result.frame_schedule.ref_indices):
        if ref == 0:
            out.update(_ndarray_fields(result.result_disp[i], f"disp[{i}]"))
            out.update(_ndarray_fields(result.result_def_grad[i], f"grad[{i}]"))
            out.update(_ndarray_fields(
                result.result_fe_mesh_each_frame[i], f"mesh[{i}]",
            ))
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _translation(dx: float, dy: float):
    return (lambda x, y: np.full_like(x, dx), lambda x, y: np.full_like(y, dy))


def _translating_speckle(h, w, seed, n_frames, step_xy):
    ref = generate_speckle(h, w, sigma=3.0, seed=seed)
    return [ref] + [
        apply_displacement_lagrangian(
            ref, *_translation(k * step_xy[0], k * step_xy[1]),
        )
        for k in range(1, n_frames)
    ]


def _rect(shape, x0, x1, y0, y1) -> np.ndarray:
    m = np.zeros(shape)
    m[y0:y1, x0:x1] = 1.0
    return m


def _all_ones_case():
    """API style: one all-ones mask per frame, range (30, w - 30)."""
    h, w = 160, 192
    imgs = _translating_speckle(h, w, seed=17, n_frames=4, step_xy=(1.2, 0.5))
    para = dicpara_default(
        winsize=WIN, winstepsize=STEP, winsize_min=16, img_size=(h, w),
        gridxy_roi_range=GridxyROIRange(gridx=(30, w - 30), gridy=(30, h - 30)),
        reference_mode="incremental", size_of_fft_search_region=10,
        init_guess_mode="fft", show_plots=False,
    )
    return para, imgs, [np.ones((h, w)) for _ in imgs]


def _inherited_case(imgs, masks, roi=None):
    """GUI style: a reference frame without an ROI inherits mask 0 (uint8).

    With ``roi`` strictly inside bbox(mask 0), the inherited mask leaves
    the range, so only the same-as-frame-0 skip keeps the frame-0 grid.
    """
    m0 = masks[0].astype(np.uint8)
    inherited = [m0, m0.copy(), np.ones((H, W), dtype=np.uint8)]
    return drift_para(roi or range_of(masks[0])), imgs[:3], inherited


def _roi_then_ones_case(imgs, masks):
    """API style: a frame-0 ROI, then np.ones ("no mask") for the rest."""
    ones = [np.ones((H, W)), np.ones((H, W))]
    return drift_para(range_of(masks[0])), imgs[:3], [masks[0], *ones]


def _no_expansion_case(name: str, drift):
    # Drift cases stop at frame 2: later frames are only slow to search
    # (the specimen leaves the frame-0 grid or the FFT range).
    imgs, masks = drift
    if name == "accumulative_per_frame_masks":
        para = drift_para(range_of(masks[0]), reference_mode="accumulative")
        return para, imgs[:3], masks[:3]
    if name == "masks_inside_range":
        return drift_para(range_of(*masks)), imgs, masks
    if name == "all_ones_range30":
        return _all_ones_case()
    if name == "inherited_mask0":
        return _inherited_case(imgs, masks)
    if name == "inherited_mask0_inner_range":
        inner = GridxyROIRange(
            gridx=(X0 + 16, X1 - 17), gridy=(Y0 + 16, Y1 - 17),
        )
        return _inherited_case(imgs, masks, roi=inner)
    if name == "roi_then_ones":
        return _roi_then_ones_case(imgs, masks)
    raise KeyError(name)


def _center_seed_set(para: DICPara, mask0: np.ndarray) -> SeedSet:
    """One Starting Point at the specimen center, on the frame-0 lattice."""
    x0, y0 = build_grid_for_roi(para, H, W)
    coords = mesh_setup(x0, y0, para).coordinates_fem
    idx = int(np.argmin(np.hypot(coords[:, 0] - CX, coords[:, 1] - CY)))
    rmap = precompute_node_regions(coords, mask0, (H, W))
    region = next(
        r for r, nodes in enumerate(rmap.region_node_lists)
        if idx in {int(n) for n in nodes}
    )
    return SeedSet(seeds=(Seed(node_idx=idx, region_id=region),))


# A static textured patch outside the frame-0 range that mask 3 also covers
# (a stray ROI component over a grip or fixture): [x0, x1) x [y0, y1).
STRAY = (296, 344, 16, 64)


def _stray_blob_data(drift):
    imgs, masks = drift
    x0, x1, y0, y1 = STRAY
    patch = _rect((H, W), *STRAY) > 0.5
    tex = generate_speckle(H, W, sigma=3.0, seed=29)
    stray_imgs = [np.where(patch, tex, img) for img in imgs]
    stray_masks = list(masks)
    stray_masks[3] = np.maximum(masks[3], patch.astype(np.float64))
    assert not np.any(np.asarray(masks[0])[y0:y1, x0:x1])
    return stray_imgs, stray_masks


# ---------------------------------------------------------------------------
# Module fixtures (each configuration runs once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drift():
    return drift_data()


@pytest.fixture(scope="module")
def roi0(drift):
    return range_of(drift[1][0])


@pytest.fixture(scope="module")
def drift_run(drift, roi0):
    return run_counted(drift_para(roi0), *drift, compute_strain=True)


@pytest.fixture(scope="module")
def drift_identity_run(drift, roi0):
    return run_counted(drift_para(roi0), *drift, identity=True)


NO_EXPANSION_CASES = (
    "accumulative_per_frame_masks",
    "masks_inside_range",
    "all_ones_range30",
    "inherited_mask0",
    "inherited_mask0_inner_range",
    "roi_then_ones",
)


@pytest.fixture(scope="module", params=NO_EXPANSION_CASES)
def no_expansion_pair(request, drift):
    para, imgs, masks = _no_expansion_case(request.param, drift)
    real = run_counted(para, imgs, masks)
    ident = run_counted(para, imgs, masks, identity=True)
    return real, ident


# ---------------------------------------------------------------------------
# Accuracy and coverage on the drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", FRAMES)
def test_drift_interior_accuracy(drift_run, k):
    assert_frame_accuracy(drift_run.result, k)


def test_k_pair_mesh_covers_ref_mask(drift_run, drift):
    """K-pair hull brackets bbox(mask K), on the frame-0 phase, tight (D5)."""
    result, masks = drift_run.result, drift[1]
    xs0 = np.unique(result.dic_mesh.coordinates_fem[:, 0])
    ys0 = np.unique(result.dic_mesh.coordinates_fem[:, 1])
    checked = 0
    for i, ref in enumerate(result.frame_schedule.ref_indices):
        if ref == 0:
            continue
        coords = result.result_fe_mesh_each_frame[i].coordinates_fem
        b = bbox(masks[ref])
        for nodes, first, lo, hi, dim in (
            (np.unique(coords[:, 0]), xs0[0], b[0], b[1], W),
            (np.unique(coords[:, 1]), ys0[0], b[2], b[3], H),
        ):
            assert nodes[0] <= max(lo, HW) and nodes[-1] >= min(hi, dim - 1 - HW)
            assert np.all(np.mod(nodes - first, STEP) == 0)
            assert nodes[0] > lo - (PAD_STEPS + 1) * STEP
            assert nodes[-1] < hi + (PAD_STEPS + 1) * STEP
        checked += 1
    assert checked == N_DEF - 1


def test_k_pair_U_finite(drift_run):
    result = drift_run.result
    for i, ref in enumerate(result.frame_schedule.ref_indices):
        if ref != 0:
            assert np.isfinite(result.result_disp[i].U).all(), f"pair {i}"


def test_ref0_pairs_and_canonical_identical(drift_run, drift_identity_run, roi0):
    real, ident = drift_run.result, drift_identity_run.result
    assert_identical(ref0_pair_arrays(real), ref0_pair_arrays(ident))
    clamped = compute_clamped_roi((H, W), roi0)
    assert real.dic_para.gridxy_roi_range == clamped
    assert ident.dic_para.gridxy_roi_range == clamped
    for name in ("winsize", "winstepsize", "img_size", "reference_mode",
                 "init_fft_search_method"):
        assert getattr(real.dic_para, name) == getattr(ident.dic_para, name)


def test_mesh_setup_call_count(drift_run, drift_identity_run):
    """Frame 0 plus one lattice switch for each of the references 1-4."""
    assert drift_run.n_mesh_setup == 5
    assert drift_identity_run.n_mesh_setup == 1


def test_no_expansion_byte_identical(no_expansion_pair):
    real, ident = no_expansion_pair
    assert_identical(result_arrays(real.result), result_arrays(ident.result))
    assert real.n_mesh_setup == 1
    assert ident.n_mesh_setup == 1


def test_external_mesh_untouched(drift, roi0):
    imgs, masks = drift
    para = drift_para(roi0)
    xs = np.arange(roi0.gridx[0] + 8, roi0.gridx[1], STEP, dtype=np.float64)
    ys = np.arange(roi0.gridy[0] + 8, roi0.gridy[1], STEP, dtype=np.float64)
    mesh = mesh_setup(xs, ys, para)
    c = mesh.coordinates_fem
    u1, v1 = motion(1)
    U0 = np.empty(2 * len(c))
    U0[0::2], U0[1::2] = u1(c[:, 0], c[:, 1]), v1(c[:, 0], c[:, 1])
    run = run_counted(para, imgs, masks, mesh=mesh, U0=U0)
    assert run.n_mesh_setup == 0
    given = {tuple(e) for e in mesh.elements_fem}
    for m in (run.result.dic_mesh, *run.result.result_fe_mesh_each_frame):
        np.testing.assert_array_equal(m.coordinates_fem, c)
        assert {tuple(e) for e in m.elements_fem} <= given


# ---------------------------------------------------------------------------
# Lattice switches: re-trim and a return to reference 0
# ---------------------------------------------------------------------------

RT_SHAPE = (192, 256)
RT_SCHEDULE = (0, 1, 2, 3, 1)


def _two_mask_case(schedule: tuple[int, ...], init_guess_mode: str = "fft"):
    """Translating speckle (1.5, 0.5) px per frame. Frame 1 inherits mask
    0; frames 2 and 3 share one mask wider than it (one ROI batch-imported
    to several frames); frames 4 and 5 are unmasked deformed frames."""
    h, w = RT_SHAPE
    imgs = _translating_speckle(h, w, seed=23, n_frames=6, step_xy=(1.5, 0.5))
    m0 = _rect(RT_SHAPE, 60, 160, 50, 140)
    wide = _rect(RT_SHAPE, 60, 220, 50, 140)
    masks = [m0, m0.copy(), wide, wide.copy(), np.ones(RT_SHAPE), np.ones(RT_SHAPE)]
    para = dicpara_default(
        winsize=WIN, winstepsize=STEP, winsize_min=16, img_size=RT_SHAPE,
        gridxy_roi_range=range_of(m0), reference_mode="incremental",
        frame_schedule=FrameSchedule(ref_indices=schedule),
        size_of_fft_search_region=16, init_guess_mode=init_guess_mode,
        show_plots=False,
    )
    return para, imgs, masks


def _trimmed_mesh(para, roi, mask):
    """Lattice of ``roi`` trimmed against ``mask``, as the pipeline builds it."""
    p = dataclasses.replace(para, gridxy_roi_range=roi)
    x0, y0 = build_grid_for_roi(p, *RT_SHAPE)
    base = mesh_setup(x0, y0, p)
    keep = trimmed_keep_indices(
        base.coordinates_fem, base.elements_fem, np.asarray(mask, np.float64),
    )
    return base.coordinates_fem, base.elements_fem[keep], len(base.elements_fem)


def test_inherited_mask_ref_retrims_after_switch():
    """Every pair's mesh is its reference's lattice trimmed to that mask.

    Refs 2 and 3 get equal but separate grids, so the switch 2 -> 3 rebuilds
    the lattice while the mask hash stays the same: the trim must still run.
    """
    para, imgs, masks = _two_mask_case(RT_SCHEDULE)
    result = run_aldic(para, imgs, masks, compute_strain=False)
    roi0 = compute_clamped_roi(RT_SHAPE, para.gridxy_roi_range)
    grids = plan_reference_grids(
        roi0, masks, set(RT_SCHEDULE), RT_SHAPE, WIN, STEP,
    )
    assert grids.roi_for(1) is roi0
    assert grids.roi_for(2) == grids.roi_for(3) != roi0
    for i, ref in enumerate(RT_SCHEDULE):
        coords, elems, n_untrimmed = _trimmed_mesh(
            para, grids.roi_for(ref), masks[ref],
        )
        if ref in (2, 3):
            assert len(elems) < n_untrimmed, "fixture must need a trim"
        mesh = result.result_fe_mesh_each_frame[i]
        np.testing.assert_array_equal(mesh.coordinates_fem, coords)
        np.testing.assert_array_equal(mesh.elements_fem, elems)


@pytest.mark.parametrize("mode", ["fft", "previous"])
def test_schedule_returning_to_ref0(mode):
    """Refs 0, 1, 2, 0, 3: the return to 0 rebuilds the frame-0 lattice."""
    para, imgs, masks = _two_mask_case((0, 1, 2, 0, 3), init_guess_mode=mode)
    real = run_counted(para, imgs, masks)
    ident = run_counted(para, imgs, masks, identity=True)
    assert real.n_mesh_setup == 4  # frame 0, ref 2, back to 0, ref 3
    assert_identical(ref0_pair_arrays(real.result), ref0_pair_arrays(ident.result))


# ---------------------------------------------------------------------------
# Option interplay on the drift
# ---------------------------------------------------------------------------


def _assert_interior_medians(result: PipelineResult) -> None:
    medians = [interior_median(result, k) for k in FRAMES]
    assert max(medians) < INTERIOR_MEDIAN_PX, f"interior medians {medians}"


def _seed_prop_para(roi0, mask0) -> DICPara:
    seed_set = _center_seed_set(drift_para(roi0), mask0)
    return drift_para(roi0, init_guess_mode="seed_propagation", seed_set=seed_set)


def test_seed_propagation(drift, roi0):
    result = run_counted(_seed_prop_para(roi0, drift[1][0]), *drift).result
    assert not result.stopped_early
    assert result.reseed_events == ()
    _assert_interior_medians(result)


def test_seed_propagation_with_stray_blob(drift, roi0):
    """A stray component of mask 3 gets nodes; auto-place must seed it."""
    imgs, masks = _stray_blob_data(drift)
    result = run_counted(_seed_prop_para(roi0, masks[0]), imgs, masks).result
    assert not result.stopped_early
    pair = result.frame_schedule.ref_indices.index(3)
    x, y = result.result_fe_mesh_each_frame[pair].coordinates_fem.T
    in_blob = (x >= STRAY[0]) & (x < STRAY[1]) & (y >= STRAY[2]) & (y < STRAY[3])
    assert in_blob.any(), "the stray component must be gridded"
    _assert_interior_medians(result)


def _flat_speck_data(drift):
    """Like ``_stray_blob_data`` but the speck lies on featureless background."""
    imgs, masks = drift
    x0, x1, y0, y1 = STRAY
    patch = _rect((H, W), *STRAY) > 0.5
    speck_masks = list(masks)
    speck_masks[3] = np.maximum(masks[3], patch.astype(np.float64))
    assert not np.any(np.asarray(masks[0])[y0:y1, x0:x1])
    # The warp leaves ~1e-13 interpolation noise on the constant background.
    assert np.ptp(np.asarray(imgs[3])[y0:y1, x0:x1]) < 1e-9, "speck must be untextured"
    return list(imgs), speck_masks


def test_seed_propagation_stops_on_untextured_speck(drift, roi0):
    """Accepted behavior change (user decision, 2026-10-15).

    A speck of a reference ROI over featureless background, outside frame 1's
    box, now gets nodes on that reference's grid. Auto-place cannot seed it, so
    the run stops at that frame and keeps the completed frames. Before the fix
    the speck was off the grid and ignored. The speck-free control is
    ``test_seed_propagation`` (same data, completes); the old frozen grid could
    not finish this drift at all (it stops at frame 6 for lost seeds).
    """
    imgs, masks = _flat_speck_data(drift)
    with pytest.raises(PartialResultError, match="no working seed") as info:
        run_counted(_seed_prop_para(roi0, masks[0]), imgs, masks)
    partial = info.value.partial_result
    assert partial.stopped_early
    # 1-based frame number: image index 4, the pair solved on ref 3's grid.
    assert info.value.stopped_at_frame == 5
    assert len(partial.result_disp) == 3  # frames 1-3 are kept
    # The partial result carries the tracking diagnostics of those frames.
    frames = partial.tracking_diagnostics.frames
    assert [s.frame for s in frames] == [1, 2, 3]


def test_presolve_refinement_and_brush(drift, roi0):
    brush = _rect((H, W), 72, 120, 104, 152)
    policy = build_refinement_policy(
        refine_inner_boundary=False, refine_outer_boundary=True,
        refinement_mask=brush, min_element_size=4, half_win=HW,
    )
    result = run_counted(drift_para(roi0), *drift, refinement_policy=policy).result
    _assert_interior_medians(result)


def test_every_second_frame_is_a_reference(drift, roi0):
    schedule = FrameSchedule.from_every_n(2, N_DEF + 1)
    result = run_counted(drift_para(roi0, frame_schedule=schedule), *drift).result
    _assert_interior_medians(result)


def test_previous_pyramid_periodic_reset(drift, roi0):
    """Pyramid NCC mis-tracks pair 0->1 of this drift whatever the grid (a
    separate issue), so only "no exception, ref-0 pairs identical" is pinned."""
    para = drift_para(
        roi0, init_guess_mode="previous", init_fft_search_method=2,
        fft_reset_interval=3,
    )
    imgs, masks = drift[0][:3], drift[1][:3]  # pairs 0->1 and 1->2
    real = run_counted(para, imgs, masks)
    ident = run_counted(para, imgs, masks, identity=True)
    assert real.n_mesh_setup == 2
    assert_identical(ref0_pair_arrays(real.result), ref0_pair_arrays(ident.result))
