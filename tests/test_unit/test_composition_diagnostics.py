"""Tracking diagnostics counted by the cumulative composition (plan step 3.3).

``_compose_cumulative`` returns ``(result_disp, TrackingDiagnostics | None)``.
Per composed frame it counts the queries that fell outside the pair mesh's
convex hull (their displacement was copied from the nearest node), how far
that node was, and the nodes the crack-aware rules stopped tracking. The
values must not change: ``[0]`` is checked against outputs recorded at
04834a8 (``tests/fixtures/cumulative_composition_04834a8.npz``), and a
diagnostics failure must leave them intact and yield ``None``.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import al_dic.core.tracking_diagnostics as td
from al_dic.core.data_structures import (
    DICMesh,
    FrameResult,
    FrameSchedule,
    FrameTrackingStats,
    TrackingDiagnostics,
)
from al_dic.core.pipeline import (
    _compose_cumulative,
    _compute_cumulative_displacements_tree,
)

from ._composition_oracle_inputs import (
    CASE_NAMES,
    FIXTURE_PATH,
    build_case,
    fixture_key,
    input_fingerprint,
    run_case,
)

STEP = 16.0
DIAG_LOGGER = "al_dic.core.tracking_diagnostics"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _grid(nx: int = 5, ny: int = 5) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(nx) * STEP, np.arange(ny) * STEP)
    return np.column_stack([xs.ravel(), ys.ravel()])


def _frame(u, v, ref_frame: int) -> FrameResult:
    U = np.empty(2 * len(u))
    U[0::2] = u
    U[1::2] = v
    return FrameResult(U=U, ref_frame=ref_frame)


def _mesh(coords: np.ndarray, elems: np.ndarray | None = None) -> DICMesh:
    if elems is None:
        elems = np.empty((0, 8), dtype=np.int64)
    return DICMesh(coordinates_fem=coords.copy(), elements_fem=elems.copy())


def _translation_chain(us, schedule, coords_per_pair=None):
    """Pairs of uniform x-translation ``us[i]`` on the 5x5 grid."""
    coords = _grid()
    n = len(coords)
    disp = [
        _frame(np.full(n, u), np.zeros(n), ref)
        for u, ref in zip(us, schedule.ref_indices)
    ]
    meshes = [
        _mesh(coords if coords_per_pair is None else coords_per_pair[i])
        for i in range(len(us))
    ]
    return disp, meshes, len(us) + 1, schedule


def _two_step_drift(**kw):
    """Incremental u=40 then u=1: step 2 queries at x + 40 on the same grid."""
    args = _translation_chain(
        (40.0, 1.0), FrameSchedule.from_mode("incremental", 3),
    )
    return _compose_cumulative(*args, **kw)


def _cut_mesh(h: int = 120, w: int = 200):
    xs = np.arange(STEP, w - 15.0, STEP)
    ys = np.arange(STEP, h - 15.0, STEP)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    nx, ny = len(xs), len(ys)
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    elems = np.full((len(ii), 8), -1, np.int64)
    elems[:, 0] = ii * ny + jj
    elems[:, 1] = (ii + 1) * ny + jj
    elems[:, 2] = (ii + 1) * ny + (jj + 1)
    elems[:, 3] = ii * ny + (jj + 1)
    return coords, elems


def _crack_setup(born_dead: bool = False):
    """Same as test_cumulative_composition's: frames 1 and 2 show an open
    band at rows 56..72, so node row y=64 (11 nodes) dies at frame 1."""
    coords, elems = _cut_mesh()
    n = len(coords)
    m_full = np.ones((120, 200))
    m_open = np.ones((120, 200))
    m_open[56:73, :] = 0.0
    v12 = np.where(coords[:, 1] < 64.0, -6.0, 6.0)
    cent = coords[elems[:, :4]].mean(axis=1)
    elems_cut = elems[np.abs(cent[:, 1] - 64.0) >= 16.0]
    disp = [
        _frame(np.zeros(n), np.zeros(n), 0),
        _frame(np.zeros(n), v12, 1),
    ]
    meshes = [_mesh(coords, elems), _mesh(coords, elems_cut)]
    m0 = m_full
    if born_dead:
        m0 = np.ones((120, 200))
        m0[62:67, :] = 0.0
    masks = [m0, m_open, m_open]
    schedule = FrameSchedule.from_mode("incremental", 3)
    return (disp, meshes, 3, schedule), dict(masks=masks, crack_radius=32.0)


def _by_frame(diag: TrackingDiagnostics) -> dict[int, FrameTrackingStats]:
    return {s.frame: s for s in diag.frames}


def _assert_loss_invariant(diag: TrackingDiagnostics) -> None:
    for s in diag.frames:
        lost = s.n_lost_parent_mask + s.n_lost_child_mask + s.n_lost_no_data
        assert s.n_queried - s.n_tracked == lost, s
        assert 0 <= s.n_tracked_extrapolated <= s.n_tracked
        assert s.n_extrapolated_border <= s.n_extrapolated_far
        assert s.n_extrapolated_far <= s.n_extrapolated <= s.n_queried


def _assert_tracked_matches_u_accum(results, diag) -> None:
    """n_tracked of a composed frame == its finite U_accum rows."""
    for s in diag.frames:
        ua = results[s.frame - 1].U_accum
        finite = np.isfinite(ua[0::2]) & np.isfinite(ua[1::2])
        assert int(finite.sum()) == s.n_tracked, s


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


class TestCompositionDiagnostics:
    def test_out_of_hull_counts_and_distance(self):
        results, diag = _two_step_drift(far_px=STEP)
        f1, f2 = _by_frame(diag)[1], _by_frame(diag)[2]
        assert (f1.ref_frame, f1.n_queried, f1.n_extrapolated) == (0, 25, 0)
        assert f1.max_extrapolation_px == 0.0
        # Queries at x + 40: columns x=72, 88, 104 lie beyond the hull (x<=64)
        # at 8, 24 and 40 px from the last column; 24 and 40 exceed 16 px.
        assert (f2.ref_frame, f2.n_queried) == (1, 25)
        assert f2.n_extrapolated == 15
        assert f2.n_extrapolated_far == 10
        assert f2.n_extrapolated_border == 0
        assert f2.max_extrapolation_px == pytest.approx(40.0, abs=1e-12)
        assert f2.n_tracked == 25
        assert f2.n_tracked_extrapolated == 10
        np.testing.assert_allclose(results[1].U_accum[0::2], 41.0, atol=1e-12)
        assert diag.far_extrapolation_px == STEP
        assert diag.n_nodes == 25 and diag.n_never_material == 0
        assert diag.n_material == 25
        assert diag.has_later_references
        assert diag.grid_extents == ()
        _assert_loss_invariant(diag)

    def test_border_and_in_region_split(self):
        # x=104 queries lie outside the measurable box, x=88 ones inside.
        _, diag = _two_step_drift(far_px=STEP, measurable_box=(0, 100, 0, 100))
        f2 = _by_frame(diag)[2]
        assert (f2.n_extrapolated_far, f2.n_extrapolated_border) == (10, 5)
        assert f2.n_tracked_extrapolated == 10
        assert diag.measurable_box == (0, 100, 0, 100)

    def test_without_far_threshold_nothing_is_far(self):
        _, diag = _two_step_drift()
        f2 = _by_frame(diag)[2]
        assert f2.n_extrapolated == 15
        assert f2.n_extrapolated_far == 0 and f2.n_tracked_extrapolated == 0
        assert diag.far_extrapolation_px is None
        assert diag.measurable_box is None

    def test_taint_is_inherited_along_the_chain(self):
        # The third pair is solved on a grid that sits where the points
        # are (x + 41), so it extrapolates nothing; the ten nodes copied
        # from far at frame 2 are still flagged.
        grid = _grid()
        args = _translation_chain(
            (40.0, 1.0, 0.0), FrameSchedule.from_mode("incremental", 4),
            coords_per_pair=[grid, grid, grid + np.array([41.0, 0.0])],
        )
        _, diag = _compose_cumulative(*args, far_px=STEP)
        f3 = _by_frame(diag)[3]
        assert f3.n_extrapolated == 0
        assert f3.n_tracked_extrapolated == 10

    def test_tree_counts_shared_segments_once_and_inherits_taint(self):
        # Frame 2's pair mesh covers only x <= 32, so frame-0 nodes at x=48
        # (16 px away, not far) and x=64 (32 px, far) are extrapolated there.
        # Frame 3 (reference 2) reuses frame 2's cached segment.
        grid = _grid()
        narrow = grid[grid[:, 0] <= 32.0]
        schedule = FrameSchedule(ref_indices=(0, 0, 2))
        n = len(grid)
        disp = [
            _frame(np.full(n, 1.0), np.zeros(n), 0),
            _frame(np.full(len(narrow), 2.0), np.zeros(len(narrow)), 0),
            _frame(np.zeros(n), np.zeros(n), 2),
        ]
        moved = grid + np.array([2.0, 0.0])
        meshes = [_mesh(grid), _mesh(narrow), _mesh(moved)]
        _, diag = _compose_cumulative(disp, meshes, 4, schedule, far_px=STEP)
        frames = _by_frame(diag)
        assert sorted(frames) == [1, 2, 3]
        assert [frames[k].ref_frame for k in (1, 2, 3)] == [0, 0, 2]
        assert frames[1].n_extrapolated == 0
        assert frames[1].n_tracked_extrapolated == 0
        assert (frames[2].n_extrapolated, frames[2].n_extrapolated_far) == (10, 5)
        assert frames[2].n_tracked_extrapolated == 5
        assert frames[3].n_extrapolated == 0  # segment 0 -> 2 not recounted
        assert frames[3].n_tracked_extrapolated == 5
        assert diag.has_later_references

    def test_accumulative_never_extrapolates(self):
        args = _translation_chain(
            (40.0, 3.0, -25.0), FrameSchedule.from_mode("accumulative", 4),
        )
        _, diag = _compose_cumulative(*args, far_px=STEP)
        assert len(diag.frames) == 3
        for s in diag.frames:
            assert s.ref_frame == 0
            assert (s.n_extrapolated, s.n_extrapolated_far) == (0, 0)
            assert s.max_extrapolation_px == 0.0
        assert not diag.has_later_references

    def test_masks_none_gives_no_kills(self):
        _, diag = _two_step_drift(far_px=STEP)
        for s in diag.frames:
            assert s.n_lost_parent_mask == s.n_lost_child_mask == 0
        assert diag.n_never_material == 0

    def test_crack_kills_counted_once(self):
        args, kw = _crack_setup()
        results, diag = _compose_cumulative(*args, **kw, far_px=STEP)
        f1, f2 = _by_frame(diag)[1], _by_frame(diag)[2]
        assert f1.n_queried == 66
        assert f1.n_lost_child_mask == 11 and f1.n_lost_parent_mask == 0
        assert f1.n_tracked == 55
        # Dead rows are never queried again, and nothing new dies.
        assert f2.n_queried == 55
        assert f2.n_lost_parent_mask == f2.n_lost_child_mask == 0
        assert f2.n_lost_no_data == 0 and f2.n_tracked == 55
        assert diag.n_never_material == 0 and diag.n_material == 66
        _assert_loss_invariant(diag)
        _assert_tracked_matches_u_accum(results, diag)

    def test_born_dead_nodes_are_never_material(self):
        args, kw = _crack_setup(born_dead=True)
        results, diag = _compose_cumulative(*args, **kw, far_px=STEP)
        assert diag.n_never_material == 11
        assert diag.n_material == 55
        f1 = _by_frame(diag)[1]
        assert f1.n_queried == 55 and f1.n_lost_child_mask == 0
        _assert_loss_invariant(diag)
        _assert_tracked_matches_u_accum(results, diag)

    def test_all_nan_pair_loses_every_query_and_nan_rows_are_not_queried(self):
        grid = _grid()
        n = len(grid)
        disp = [
            _frame(np.zeros(n), np.zeros(n), 0),
            _frame(np.full(n, np.nan), np.full(n, np.nan), 1),
            _frame(np.zeros(n), np.zeros(n), 2),
        ]
        meshes = [_mesh(grid), _mesh(grid), _mesh(grid)]
        schedule = FrameSchedule.from_mode("incremental", 4)
        results, diag = _compose_cumulative(disp, meshes, 4, schedule)
        frames = _by_frame(diag)
        assert (frames[2].n_queried, frames[2].n_lost_no_data) == (25, 25)
        assert frames[2].n_tracked == 0
        assert (frames[3].n_queried, frames[3].n_tracked) == (0, 0)
        _assert_loss_invariant(diag)
        _assert_tracked_matches_u_accum(results, diag)

    def test_missing_first_pair_gives_no_diagnostics(self):
        case = build_case("jitter_missing_first_pair")
        args, kw = case.call_args()
        _, diag = _compose_cumulative(*args, **kw)
        assert diag is None


# ---------------------------------------------------------------------------
# Values: recorded oracle, and the legacy entry point
# ---------------------------------------------------------------------------


def _compose_values(*args, **kwargs):
    return _compose_cumulative(*args, **kwargs)[0]


@pytest.fixture(scope="module")
def oracle():
    with np.load(FIXTURE_PATH) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.mark.parametrize("compose", [
    _compose_values, _compute_cumulative_displacements_tree,
], ids=["compose_cumulative", "legacy_tree"])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_values_match_recorded_oracle(oracle, name, compose):
    case = build_case(name)
    assert input_fingerprint(case) == str(oracle[fixture_key(name, "input_sha256")])
    u_accum, has_u_accum = run_case(case, compose)
    np.testing.assert_array_equal(has_u_accum, oracle[fixture_key(name, "has_u_accum")])
    np.testing.assert_allclose(
        u_accum, oracle[fixture_key(name, "u_accum")],
        rtol=0, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("name", [
    n for n in CASE_NAMES if n != "jitter_missing_first_pair"
])
def test_oracle_cases_satisfy_the_invariants(name):
    case = build_case(name)
    args, kw = case.call_args()
    results, diag = _compose_cumulative(*args, **kw, far_px=STEP)
    assert diag is not None and diag.n_nodes == case.n_nodes
    _assert_loss_invariant(diag)
    _assert_tracked_matches_u_accum(results, diag)
    frames = [s.frame for s in diag.frames]
    assert frames == sorted(set(frames))
    refs = case.schedule.ref_indices
    for s in diag.frames:
        assert s.ref_frame == refs[s.frame - 1]
    assert diag.has_later_references == any(s.ref_frame != 0 for s in diag.frames)


def test_chain_break_composes_only_the_frames_it_reaches():
    case = build_case("jitter_chain_break")
    args, kw = case.call_args()
    _, diag = _compose_cumulative(*args, **kw)
    assert [s.frame for s in diag.frames] == [1]


# ---------------------------------------------------------------------------
# Fail-safe collector
# ---------------------------------------------------------------------------


_COLLECTOR_STAGES = (
    "_setup", "_record_born_dead", "_begin_step", "_parent_kills",
    "_after_increment", "_child_kills", "_end_step", "_build",
)


@pytest.mark.parametrize("stage", _COLLECTOR_STAGES)
def test_collector_failure_keeps_values_and_drops_diagnostics(
    monkeypatch, caplog, stage,
):
    case = build_case("crack_tree")
    expected, _ = run_case(case, _compose_values)

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"forced failure in {stage}")

    monkeypatch.setattr(td.TrackingCollector, stage, boom)
    args, kw = case.call_args()
    with caplog.at_level(logging.ERROR, logger=DIAG_LOGGER):
        results, diag = _compose_cumulative(*args, **kw, far_px=STEP)
    assert diag is None
    errors = [r for r in caplog.records if r.name == DIAG_LOGGER]
    assert len(errors) == 1 and errors[0].exc_info is not None
    got = np.vstack([r.U_accum for r in results])
    np.testing.assert_array_equal(got, expected, strict=True)


def test_inputs_are_not_mutated():
    case = build_case("crack_tree")
    args, kw = case.call_args()
    disp, meshes = args[0], args[1]
    before_u = [r.U.copy() for r in disp]
    before_xy = [m.coordinates_fem.copy() for m in meshes]
    before_masks = [m.copy() for m in kw["masks"]]
    originals = list(disp)
    _compose_cumulative(*args, **kw, far_px=STEP)
    for r, u in zip(originals, before_u):
        np.testing.assert_array_equal(r.U, u)
        assert r.U_accum is None
    for m, xy in zip(meshes, before_xy):
        np.testing.assert_array_equal(m.coordinates_fem, xy)
    for m, m0 in zip(kw["masks"], before_masks):
        np.testing.assert_array_equal(m, m0)


@pytest.mark.parametrize("name", [
    "TrackingDiagnostics", "FrameTrackingStats", "RefGridExtent",
])
def test_types_are_public(name):
    import al_dic
    import al_dic.core
    import al_dic.core.data_structures as ds

    for package in (al_dic, al_dic.core):
        assert getattr(package, name) is getattr(ds, name)
        assert name in package.__all__


def test_frame_stats_are_frozen():
    _, diag = _two_step_drift(far_px=STEP)
    with pytest.raises(FrozenInstanceError):
        diag.frames[0].n_tracked = 0  # type: ignore[misc]
    assert replace(diag.frames[0], n_tracked=3).n_tracked == 3
