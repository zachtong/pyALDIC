"""``run_aldic`` reports how well the nodes were tracked (plan 3.5 / 3.6 / 3.8).

On the drift sequence (``_grid_drift.py``) with one grid per reference frame,
nothing is copied from far and nothing is lost. With the planner patched to
keep the frame-0 grid (the behavior before layer 1), the later pairs copy
displacements from nodes up to ~116 px away and lose material: both
``grid_miss`` and ``material_loss`` are WARNING findings. Reference frames
that reuse frame 0's mask (scenario E, which layer 1 cannot fix) lose
material. Healthy accumulative and growing-crack runs stay below WARNING.

Every run logs one INFO summary and one WARNING per WARNING finding on the
``al_dic.core.tracking_diagnostics`` logger; diagnostics never cost a run.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pytest

import al_dic.core.pipeline as pipeline_mod
import al_dic.core.tracking_diagnostics as td
from al_dic.core._grid_extent import ReferenceGrids, plan_reference_grids
from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import DICPara, GridxyROIRange, PipelineResult
from al_dic.core.pipeline import run_aldic
from al_dic.core.tracking_diagnostics import (
    ADVICE_REFERENCE_ROI,
    FINDING_GRID_MISS,
    FINDING_MATERIAL_LOSS,
    SEVERITY_WARNING,
    assess_tracking,
    tracking_advice,
)

from tests.conftest import apply_displacement_lagrangian, generate_speckle
from tests.test_integration import test_crack_wake_strain as crack
from tests.test_integration._grid_drift import (
    H,
    N_DEF,
    SEARCH,
    STEP,
    W,
    WIN,
    drift_data,
    range_of,
)

DIAG_LOGGER = "al_dic.core.tracking_diagnostics"


class Run(NamedTuple):
    result: PipelineResult
    records: list[logging.LogRecord]

    def warnings(self) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno == logging.WARNING]

    def infos(self) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno == logging.INFO]


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _identity_planner(roi0, *_args, **_kwargs) -> ReferenceGrids:
    """Keep the frame-0 grid for every reference frame."""
    return ReferenceGrids(roi0=roi0)


def run_logged(para, imgs, masks, identity=False, **kwargs) -> Run:
    """``run_aldic`` collecting this logger's records (module fixtures cannot
    use caplog, which is function-scoped)."""
    log = logging.getLogger(DIAG_LOGGER)
    handler, level = _Collect(), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    kwargs.setdefault("compute_strain", False)
    try:
        with pytest.MonkeyPatch.context() as mp:
            if identity:
                mp.setattr(pipeline_mod, "plan_reference_grids", _identity_planner)
            result = run_aldic(para, imgs, masks, **kwargs)
    finally:
        log.removeHandler(handler)
        log.setLevel(level)
    return Run(result, handler.records)


def drift_para(roi: GridxyROIRange, **overrides) -> DICPara:
    kw = dict(
        winsize=WIN, winstepsize=STEP, winsize_min=16, img_size=(H, W),
        gridxy_roi_range=roi, reference_mode="incremental",
        size_of_fft_search_region=SEARCH, init_guess_mode="fft",
        show_plots=False,
    )
    kw.update(overrides)
    return dicpara_default(**kw)


def _warning_keys(result: PipelineResult) -> list[str]:
    return [
        f.key for f in assess_tracking(result.tracking_diagnostics)
        if f.severity == SEVERITY_WARNING
    ]


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
def real_run(drift, roi0) -> Run:
    return run_logged(drift_para(roi0), *drift)


@pytest.fixture(scope="module")
def identity_run(drift, roi0) -> Run:
    return run_logged(drift_para(roi0), *drift, identity=True)


@pytest.fixture(scope="module")
def inherited_run(drift, roi0) -> Run:
    """Scenario E: every reference frame reuses frame 0's mask."""
    imgs, masks = drift
    return run_logged(drift_para(roi0), imgs[:3], [masks[0]] * 3)


@pytest.fixture(scope="module")
def accumulative_run(drift, roi0) -> Run:
    """Accumulative with a mask per frame (later frames are slow to search)."""
    imgs, masks = drift
    para = drift_para(roi0, reference_mode="accumulative")
    return run_logged(para, imgs[:3], masks[:3])


@pytest.fixture(scope="module")
def crack_run() -> Run:
    """Incremental, growing crack (test_crack_wake_strain's sequence)."""
    ref = generate_speckle(crack.H, crack.W, sigma=3.0, seed=5)
    imgs, masks = [ref], [crack._mask(0)]
    for i in range(1, crack.N_DEF + 1):
        imgs.append(apply_displacement_lagrangian(ref, *crack._uv(i)))
        masks.append(crack._mask(i))
    pad, step = crack.PAD, crack.STEP
    para = dicpara_default(
        winsize=32, winstepsize=step, winsize_min=16,
        img_size=(crack.H, crack.W),
        gridxy_roi_range=GridxyROIRange(
            gridx=(pad, crack.W - pad - 1), gridy=(pad, crack.H - pad - 1),
        ),
        reference_mode="incremental", show_plots=False,
    )
    return run_logged(para, imgs, masks)


# ---------------------------------------------------------------------------
# Findings and logging
# ---------------------------------------------------------------------------


def test_real_drift_run_is_clean(real_run):
    diag = real_run.result.tracking_diagnostics
    assert diag is not None
    assert [s.frame for s in diag.frames] == list(range(1, N_DEF + 1))
    assert [s.ref_frame for s in diag.frames] == list(range(N_DEF))
    assert diag.has_later_references
    assert diag.measurable_box == (WIN // 2, W - 1 - WIN // 2, WIN // 2, H - 1 - WIN // 2)
    assert diag.far_extrapolation_px == STEP
    assert sum(s.n_extrapolated_far for s in diag.frames) == 0
    assert assess_tracking(diag) == ()
    assert real_run.warnings() == []
    assert len(real_run.infos()) == 1


def test_real_drift_run_records_the_reference_grids(real_run, drift):
    result = real_run.result
    _, masks = drift
    grids = plan_reference_grids(
        result.dic_para.gridxy_roi_range, masks,
        result.frame_schedule.ref_frame_set, (H, W), WIN, STEP,
    )
    extents = result.tracking_diagnostics.grid_extents
    assert [(e.ref_frame, e.roi) for e in extents] == list(grids.by_ref)
    assert [e.ref_frame for e in extents] == [1, 2, 3, 4]
    assert all(e.n_nodes > len(result.dic_mesh.coordinates_fem) for e in extents)


def test_frame0_grid_for_every_reference_warns_twice(identity_run):
    diag = identity_run.result.tracking_diagnostics
    assert diag.grid_extents == ()
    assert _warning_keys(identity_run.result) == [
        FINDING_GRID_MISS, FINDING_MATERIAL_LOSS,
    ]
    miss = assess_tracking(diag)[0]
    assert (miss.frame, miss.ref_frame) == (N_DEF, N_DEF - 1)
    assert miss.max_px > 4 * STEP
    warnings = identity_run.warnings()
    assert len(warnings) == 2
    assert "image index 5" in warnings[0].getMessage()


def test_inherited_masks_warn_about_lost_material(inherited_run):
    diag = inherited_run.result.tracking_diagnostics
    assert _warning_keys(inherited_run.result) == [FINDING_MATERIAL_LOSS]
    assert tracking_advice(diag) == ADVICE_REFERENCE_ROI
    assert len(inherited_run.warnings()) == 1


@pytest.mark.parametrize("name", ["accumulative_run", "crack_run"])
def test_healthy_runs_stay_below_warning(request, name):
    run = request.getfixturevalue(name)
    assert run.result.tracking_diagnostics is not None
    assert _warning_keys(run.result) == []
    assert run.warnings() == []
    assert len(run.infos()) == 1


# ---------------------------------------------------------------------------
# Partial runs and failures
# ---------------------------------------------------------------------------


def test_canceled_run_reports_the_completed_frames(drift, roi0):
    calls = iter(range(100))

    def stop_after_two_frames() -> bool:
        return next(calls) >= 2

    run = run_logged(drift_para(roi0), *drift, stop_fn=stop_after_two_frames)
    assert run.result.stopped_early
    assert len(run.result.result_disp) == 2
    diag = run.result.tracking_diagnostics
    assert [s.frame for s in diag.frames] == [1, 2]
    # Grids were planned for references 1-4; only reference 1 was used.
    assert [e.ref_frame for e in diag.grid_extents] == [1]


@pytest.mark.parametrize("target, name", [
    (td.TrackingCollector, "_end_step"),
    (td, "assess_tracking"),
])
def test_diagnostics_failure_keeps_the_run(
    monkeypatch, accumulative_run, drift, roi0, target, name,
):
    def boom(*_args, **_kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(target, name, boom)
    imgs, masks = drift
    para = drift_para(roi0, reference_mode="accumulative")
    run = run_logged(para, imgs[:3], masks[:3])
    assert run.result.tracking_diagnostics is None
    errors = [r for r in run.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    for got, want in zip(run.result.result_disp, accumulative_run.result.result_disp):
        np.testing.assert_array_equal(got.U_accum, want.U_accum)
