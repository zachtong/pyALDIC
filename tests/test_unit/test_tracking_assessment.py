"""Assessment and logging of tracking diagnostics (plan steps 3.5 / D16-D17).

``assess_tracking`` turns a ``TrackingDiagnostics`` into findings with
English keys; each rule is pinned at boundary-1 / boundary / boundary+1.
``log_tracking_summary`` writes one INFO summary plus one WARNING per
WARNING finding to the ``al_dic.core.tracking_diagnostics`` logger.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import al_dic.core.tracking_diagnostics as td
from al_dic.core._grid_extent import ReferenceGrids, grid_node_count
from al_dic.core.data_structures import (
    DICMesh,
    FrameResult,
    FrameSchedule,
    FrameTrackingStats,
    GridxyROIRange,
    RefGridExtent,
    TrackingDiagnostics,
)
from al_dic.core.pipeline import _compose_cumulative
from al_dic.core.tracking_diagnostics import (
    ADVICE_FRAME_ROI,
    ADVICE_REFERENCE_ROI,
    FINDING_GRID_MISS,
    FINDING_LEFT_MEASURABLE_AREA,
    FINDING_MATERIAL_LOSS,
    GRID_MISS_FRACTION,
    GRID_MISS_MIN_POINTS,
    LOSS_BURST_FRACTION,
    LOSS_NOTICE_FRACTION,
    LOSS_RULE_BURST,
    LOSS_RULE_CUMULATIVE,
    LOSS_WARN_FRACTION,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    TrackingFinding,
    assess_tracking,
    finish_tracking_diagnostics,
    log_tracking_summary,
    tracking_advice,
)

from ._composition_oracle_inputs import build_case

DIAG_LOGGER = "al_dic.core.tracking_diagnostics"
N = 1000


def _stats(frame, n_tracked, *, ref_frame=None, n_queried=N, extrap=0,
           far=0, border=0, max_px=0.0, tainted=0) -> FrameTrackingStats:
    lost = n_queried - n_tracked
    return FrameTrackingStats(
        frame=frame, ref_frame=frame - 1 if ref_frame is None else ref_frame,
        n_queried=n_queried, n_extrapolated=max(extrap, far),
        n_extrapolated_far=far, n_extrapolated_border=border,
        max_extrapolation_px=max_px, n_lost_parent_mask=0,
        n_lost_child_mask=lost, n_lost_no_data=0, n_tracked=n_tracked,
        n_tracked_extrapolated=tainted,
    )


def _diag(*frames, n_nodes=N, never=0, later=True) -> TrackingDiagnostics:
    return TrackingDiagnostics(
        n_nodes=n_nodes, n_never_material=never, far_extrapolation_px=16.0,
        measurable_box=(16, 367, 16, 239), has_later_references=later,
        frames=tuple(frames),
    )


def _gradual(last_tracked: int, n_frames: int = 5, **kw) -> TrackingDiagnostics:
    """Equal losses per frame (no burst) ending at ``last_tracked``."""
    lost = N - last_tracked
    frames = [
        _stats(k, N - round(lost * k / n_frames)) for k in range(1, n_frames + 1)
    ]
    return _diag(*frames, **kw)


def _find(findings, key) -> TrackingFinding | None:
    hits = [f for f in findings if f.key == key]
    assert len(hits) <= 1
    return hits[0] if hits else None


def _severity(diag, key) -> str | None:
    f = _find(assess_tracking(diag), key)
    return None if f is None else f.severity


def test_thresholds_are_the_users_decision():
    assert LOSS_WARN_FRACTION == 0.50
    assert LOSS_BURST_FRACTION == 0.20
    assert LOSS_NOTICE_FRACTION == 0.01
    assert (GRID_MISS_MIN_POINTS, GRID_MISS_FRACTION) == (3, 0.005)


# ---------------------------------------------------------------------------
# grid_miss: in-region far extrapolations >= max(3, ceil(0.5% of queried))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_region, expected", [
    (4, None), (5, SEVERITY_WARNING), (6, SEVERITY_WARNING),
])
def test_grid_miss_fraction_boundary(in_region, expected):
    # 0.5% of 1000 queried = 5 points.
    diag = _diag(_stats(1, N), _stats(2, N, far=in_region + 7, border=7))
    assert _severity(diag, FINDING_GRID_MISS) == expected


@pytest.mark.parametrize("in_region, expected", [
    (2, None), (3, SEVERITY_WARNING), (4, SEVERITY_WARNING),
])
def test_grid_miss_minimum_points_boundary(in_region, expected):
    # 0.5% of 100 queried rounds up to 1, below the 3-point minimum.
    diag = _diag(_stats(1, 100, n_queried=100, far=in_region), n_nodes=100)
    assert _severity(diag, FINDING_GRID_MISS) == expected


def test_grid_miss_reports_the_worst_frame():
    diag = _diag(
        _stats(1, N), _stats(2, N, far=6, max_px=30.0),
        _stats(3, N, ref_frame=1, far=9, border=1, max_px=41.5),
        _stats(4, N, far=2, max_px=50.0),
    )
    f = _find(assess_tracking(diag), FINDING_GRID_MISS)
    assert (f.frame, f.ref_frame, f.count, f.n_frames) == (3, 1, 8, 2)
    assert f.max_px == 41.5
    assert f.fraction == pytest.approx(8 / N)
    assert (f.rule, f.burst_count) == ("", 0)  # material_loss fields only


def test_raw_out_of_hull_count_never_warns():
    diag = _diag(_stats(1, N, extrap=211, far=2, max_px=1.4))
    assert assess_tracking(diag) == ()


# ---------------------------------------------------------------------------
# material_loss: WARNING > 50% by the last frame or a > 20% burst between
# consecutive frames; INFO from 1%. Fractions of n_material.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lost, expected", [
    (499, SEVERITY_INFO), (500, SEVERITY_INFO), (501, SEVERITY_WARNING),
])
def test_cumulative_loss_boundary(lost, expected):
    diag = _gradual(N - lost)
    assert _severity(diag, FINDING_MATERIAL_LOSS) == expected


@pytest.mark.parametrize("burst, expected", [
    (199, SEVERITY_INFO), (200, SEVERITY_INFO), (201, SEVERITY_WARNING),
])
def test_burst_boundary(burst, expected):
    diag = _diag(_stats(1, N), _stats(2, N - burst), _stats(3, N - burst))
    assert _severity(diag, FINDING_MATERIAL_LOSS) == expected


@pytest.mark.parametrize("lost, expected", [
    (9, None), (10, SEVERITY_INFO), (11, SEVERITY_INFO),
])
def test_notice_boundary(lost, expected):
    diag = _diag(_stats(1, N - lost))
    assert _severity(diag, FINDING_MATERIAL_LOSS) == expected


def test_first_frame_burst_is_measured_from_all_material():
    diag = _diag(_stats(1, N - 201), _stats(2, N - 201))
    f = _find(assess_tracking(diag), FINDING_MATERIAL_LOSS)
    assert (f.severity, f.frame, f.count) == (SEVERITY_WARNING, 1, 201)
    assert (f.rule, f.burst_count) == (LOSS_RULE_BURST, 201)


def test_burst_finding_names_the_rule_and_the_burst():
    # 25% lost by frame 2, 24% of it on frame 2 alone: below the 50% rule,
    # above the 20% burst rule. count stays the total lost by that frame.
    diag = _diag(_stats(1, N - 10), _stats(2, N - 250), _stats(3, N - 260))
    f = _find(assess_tracking(diag), FINDING_MATERIAL_LOSS)
    assert (f.severity, f.rule, f.frame, f.count, f.burst_count) == (
        SEVERITY_WARNING, LOSS_RULE_BURST, 2, 250, 240,
    )
    assert f.fraction == pytest.approx(0.25)


def test_fractions_use_material_nodes_not_all_nodes():
    # 100 never-material nodes: 501 of 1000 material nodes is > 50%, but
    # 501 of 1100 nodes would not be.
    diag = _gradual(N - 501, n_nodes=N + 100, never=100)
    f = _find(assess_tracking(diag), FINDING_MATERIAL_LOSS)
    assert f.severity == SEVERITY_WARNING
    assert f.fraction == pytest.approx(0.501)
    # 10 of 1000 material nodes is the 1% notice; 10 of 1100 would not be.
    diag = _diag(_stats(1, N - 10), n_nodes=N + 100, never=100)
    assert _severity(diag, FINDING_MATERIAL_LOSS) == SEVERITY_INFO


def test_burst_uses_the_previous_frame_in_time():
    # Accumulative style: each frame is composed from frame 0, so its
    # tracked count is already the total. 24% lost by frame 4, but never
    # more than 10% between consecutive frames: no burst.
    diag = _diag(
        _stats(1, 900, ref_frame=0), _stats(2, 850, ref_frame=0),
        _stats(3, 800, ref_frame=0), _stats(4, 760, ref_frame=0),
        later=False,
    )
    f = _find(assess_tracking(diag), FINDING_MATERIAL_LOSS)
    assert (f.severity, f.frame, f.count) == (SEVERITY_INFO, 4, 240)
    assert (f.rule, f.burst_count) == (LOSS_RULE_CUMULATIVE, 40)


def test_loss_finding_reports_the_last_frame_when_cumulative():
    diag = _gradual(300)
    f = _find(assess_tracking(diag), FINDING_MATERIAL_LOSS)
    assert (f.frame, f.count) == (5, 700)
    assert f.fraction == pytest.approx(0.7)
    # 140 lost on frame 5 itself (14%): the 50% rule fired, not the burst.
    assert (f.rule, f.burst_count) == (LOSS_RULE_CUMULATIVE, 140)


def _growing_crack_accumulative():
    """Accumulative pairs on an 11 x 20 Q4 lattice (step 16); frame k's
    mask opens a band of half-width 4 + 8k px around y = 168, so the band
    takes one more node row on each side every second frame (10% of the
    material) and 30% by frame 6. No motion: only the masks change."""
    step, nx, ny, n_frames = 16.0, 11, 20, 7
    xx, yy = np.meshgrid(
        16.0 + step * np.arange(nx), 16.0 + step * np.arange(ny), indexing="ij",
    )
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    elems = np.full((len(ii), 8), -1, np.int64)
    elems[:, :4] = np.column_stack([
        ii * ny + jj, (ii + 1) * ny + jj, (ii + 1) * ny + jj + 1, ii * ny + jj + 1,
    ])
    masks = []
    for k in range(n_frames):
        m = np.ones((352, 208))
        hw = 4 + 8 * k
        m[168 - hw:168 + hw + 1, :] = 0.0
        masks.append(m)
    n = len(coords)
    zero = np.zeros(2 * n)
    disp = [FrameResult(U=zero.copy(), ref_frame=0) for _ in range(n_frames - 1)]
    meshes = [
        DICMesh(coordinates_fem=coords.copy(), elements_fem=elems.copy())
        for _ in range(n_frames - 1)
    ]
    schedule = FrameSchedule.from_mode("accumulative", n_frames)
    return (disp, meshes, n_frames, schedule), dict(masks=masks, crack_radius=32.0)


def test_growing_crack_accumulative_run_stays_below_warning():
    args, kw = _growing_crack_accumulative()
    _, diag = _compose_cumulative(*args, **kw, far_px=16.0)
    assert not diag.has_later_references
    assert diag.n_material == 220
    assert [s.n_tracked for s in diag.frames] == [198, 198, 176, 176, 154, 154]
    findings = assess_tracking(diag)
    assert [(f.key, f.severity) for f in findings] == [
        (FINDING_MATERIAL_LOSS, SEVERITY_INFO),
    ]
    assert findings[0].fraction == pytest.approx(0.3)


def test_abrupt_band_in_the_coarse_crack_case_is_a_burst():
    # The recorded oracle's crack case is deliberately extreme: its band
    # swallows two of eight node rows at once (24 of 84 material nodes at
    # frame 3, 25 at frame 5), a burst above 20% by the user's rule.
    f = _find(assess_tracking(_crack_accumulative_diag()), FINDING_MATERIAL_LOSS)
    assert (f.severity, f.frame, f.count) == (SEVERITY_WARNING, 5, 26)
    # 26 of 84 (31%) is below the 50% rule: the 25 lost on frame 5 alone
    # raised it.
    assert (f.rule, f.burst_count) == (LOSS_RULE_BURST, 25)


def _crack_accumulative_diag() -> TrackingDiagnostics:
    case = build_case("crack_accumulative")
    args, kw = case.call_args()
    _, diag = _compose_cumulative(*args, **kw, far_px=16.0)
    return diag


# ---------------------------------------------------------------------------
# left_measurable_area (INFO), advice, ordering, degenerate input
# ---------------------------------------------------------------------------


def test_left_measurable_area_is_info_on_any_border_extrapolation():
    assert _find(assess_tracking(_diag(_stats(1, N))),
                 FINDING_LEFT_MEASURABLE_AREA) is None
    diag = _diag(_stats(1, N, far=1, border=1, max_px=20.0),
                 _stats(2, N, far=3, border=3, max_px=25.0))
    f = _find(assess_tracking(diag), FINDING_LEFT_MEASURABLE_AREA)
    assert (f.severity, f.frame, f.count, f.n_frames) == (SEVERITY_INFO, 2, 3, 2)


@pytest.mark.parametrize("later, advice", [
    (True, ADVICE_REFERENCE_ROI), (False, ADVICE_FRAME_ROI),
])
def test_advice_follows_the_tracking_mode(later, advice):
    assert tracking_advice(_diag(_stats(1, N), later=later)) == advice


def test_warnings_come_first():
    diag = _diag(
        _stats(1, N - 300, far=20, border=10, max_px=40.0),
        _stats(2, N - 600, far=0),
    )
    findings = assess_tracking(diag)
    keys = [f.key for f in findings]
    assert keys == [FINDING_GRID_MISS, FINDING_MATERIAL_LOSS,
                    FINDING_LEFT_MEASURABLE_AREA]
    assert [f.severity for f in findings] == [
        SEVERITY_WARNING, SEVERITY_WARNING, SEVERITY_INFO,
    ]


def test_no_frames_or_no_material_gives_no_findings():
    assert assess_tracking(_diag()) == ()
    assert assess_tracking(_diag(_stats(1, 0, n_queried=0), never=N)) == ()


# ---------------------------------------------------------------------------
# Logging (English, logger al_dic.core.tracking_diagnostics)
# ---------------------------------------------------------------------------


def _records(caplog, level):
    return [r for r in caplog.records
            if r.name == DIAG_LOGGER and r.levelno == level]


def test_one_warning_per_warning_finding(caplog):
    diag = _diag(
        _stats(1, N - 300, far=20, border=10, max_px=40.0),
        _stats(2, N - 600),
    )
    findings = assess_tracking(diag)
    with caplog.at_level(logging.INFO, logger=DIAG_LOGGER):
        log_tracking_summary(diag, findings)
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 2
    miss, loss = (w.getMessage() for w in warnings)
    assert "on image index 1, 10 node(s) inside the measurable area" in miss
    # max_px covers every out-of-hull point of the frame, not only these.
    assert "the farthest out-of-hull point on that frame was 40.0 px" in miss
    assert "600 of 1000 material node(s) (60.0%)" in loss
    assert "by image index 2" in loss
    assert len(_records(caplog, logging.INFO)) == 1


def test_burst_warning_names_the_burst_and_both_causes(caplog):
    diag = _crack_accumulative_diag()
    with caplog.at_level(logging.INFO, logger=DIAG_LOGGER):
        log_tracking_summary(diag, assess_tracking(diag))
    (warning,) = _records(caplog, logging.WARNING)
    text = warning.getMessage()
    assert (
        "25 material node(s) (29.8%) stopped being tracked between the "
        "previous frame and image index 5" in text
    )
    assert "26 of 84 (31.0%) lost by then" in text
    assert "outside the Region of Interest or" in text
    assert "no measured displacement" in text


def test_healthy_run_logs_only_the_summary(caplog):
    diag = _diag(_stats(1, N, extrap=40, max_px=1.2), _stats(2, N - 3))
    with caplog.at_level(logging.INFO, logger=DIAG_LOGGER):
        log_tracking_summary(diag, assess_tracking(diag))
    assert _records(caplog, logging.WARNING) == []
    info = _records(caplog, logging.INFO)
    assert len(info) == 1 and "Tracking summary" in info[0].getMessage()


# ---------------------------------------------------------------------------
# finish_tracking_diagnostics: grid extents + assess + log, fail-safe
# ---------------------------------------------------------------------------


def _grids() -> ReferenceGrids:
    roi0 = GridxyROIRange(gridx=(40, 151), gridy=(72, 183))
    roi2 = GridxyROIRange(gridx=(72, 199), gridy=(72, 199))
    return ReferenceGrids(roi0=roi0, by_ref=((2, roi2),))


def test_finish_attaches_grid_extents_and_logs(caplog):
    diag = _diag(_stats(1, N), _stats(2, N), _stats(3, N))  # frame 3: ref 2
    with caplog.at_level(logging.INFO, logger=DIAG_LOGGER):
        out = finish_tracking_diagnostics(diag, _grids(), (256, 384), 32, 16)
    roi2 = _grids().by_ref[0][1]
    n2 = grid_node_count(roi2, (256, 384), 32, 16)
    assert out.grid_extents == (RefGridExtent(ref_frame=2, roi=roi2, n_nodes=n2),)
    assert out.frames == diag.frames
    assert len(_records(caplog, logging.INFO)) == 1


def test_finish_keeps_only_the_grids_of_composed_references():
    # A canceled run planned grids for references 2 and 3 but composed
    # frames 1-3 only (references 0-2): reference 3's grid was never used.
    roi3 = GridxyROIRange(gridx=(104, 231), gridy=(72, 199))
    grids = ReferenceGrids(
        roi0=_grids().roi0, by_ref=(*_grids().by_ref, (3, roi3)),
    )
    diag = _diag(_stats(1, N), _stats(2, N), _stats(3, N))
    out = finish_tracking_diagnostics(diag, grids, (256, 384), 32, 16)
    assert [e.ref_frame for e in out.grid_extents] == [2]


def test_finish_passes_none_through(caplog):
    with caplog.at_level(logging.INFO, logger=DIAG_LOGGER):
        assert finish_tracking_diagnostics(None, _grids(), (256, 384), 32, 16) is None
    assert _records(caplog, logging.WARNING) == []


def test_finish_failure_returns_none_and_logs_once(monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(td, "assess_tracking", boom)
    with caplog.at_level(logging.INFO, logger=DIAG_LOGGER):
        out = finish_tracking_diagnostics(
            _diag(_stats(1, N)), _grids(), (256, 384), 32, 16,
        )
    assert out is None
    errors = _records(caplog, logging.ERROR)
    assert len(errors) == 1 and errors[0].exc_info is not None
