"""The "no mask" skip of the reference-grid planner (core/_grid_extent.py).

``run_aldic`` documents ``np.ones((H, W))`` as "no masking", so an API run
that passes a frame-0 ROI and all-ones masks for the other frames must keep
the frame-0 grid for every reference frame instead of gridding the whole
image (skip reason ``unmasked``). Kept apart from ``test_grid_extent.py``,
which is at its size budget; its property loop and oracle cover this skip
too.
"""

from __future__ import annotations

import logging

import numpy as np

from al_dic.core._grid_extent import (
    SKIP_SAME_AS_FRAME0,
    SKIP_UNMASKED,
    log_grid_plan,
    plan_reference_grids,
)
from al_dic.core.data_structures import GridxyROIRange

H, W = 256, 384
SHAPE = (H, W)
WIN, STEP = 32, 16
ROI0 = GridxyROIRange(gridx=(100, 220), gridy=(80, 170))
LOGGER_NAME = "al_dic.core._grid_extent"


def mask0() -> np.ndarray:
    """Specimen ROI whose bbox is ``ROI0``."""
    m = np.zeros(SHAPE)
    m[80:171, 100:221] = 1.0
    return m


def plan(masks, roi0=ROI0):
    refs = set(range(len(masks)))
    return plan_reference_grids(roi0, masks, refs, SHAPE, WIN, STEP)


def test_roi_then_ones_keeps_frame0_grid():
    grids = plan([mask0(), np.ones(SHAPE), np.ones(SHAPE, dtype=np.uint8)])
    assert grids.is_identity
    assert grids.skipped == ((1, SKIP_UNMASKED), (2, SKIP_UNMASKED))
    assert grids.roi_for(1) is ROI0
    assert grids.roi_for(2) is ROI0


def test_material_everywhere_in_any_dtype_is_unmasked():
    # "No masked-out pixel" uses the same > 0.5 threshold as the material.
    masks = [
        mask0(), np.full(SHAPE, 0.6), np.full(SHAPE, 255, dtype=np.uint8),
        np.ones(SHAPE, dtype=bool),
    ]
    grids = plan(masks)
    assert grids.is_identity
    assert grids.skipped == tuple((k, SKIP_UNMASKED) for k in (1, 2, 3))


def test_same_as_frame0_is_checked_first():
    # The API pattern [np.ones] * n keeps its "same_as_frame0" reason.
    ones = np.ones(SHAPE)
    roi0 = GridxyROIRange(gridx=(30, W - 31), gridy=(30, H - 31))
    grids = plan([ones, ones, np.ones(SHAPE)], roi0=roi0)
    assert grids.is_identity
    assert grids.skipped == ((1, SKIP_SAME_AS_FRAME0), (2, SKIP_SAME_AS_FRAME0))


def test_unmasked_is_checked_before_four_borders():
    m0 = np.ones(SHAPE)
    m0[100:110, :200] = 0.0  # a crack: mask 0 touches all four borders
    grids = plan([m0, np.ones(SHAPE)])
    assert grids.skipped == ((1, SKIP_UNMASKED),)


def test_unmasked_skips_are_logged(caplog):
    grids = plan([mask0(), np.ones(SHAPE), np.ones(SHAPE)])
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        log_grid_plan(grids, SHAPE, WIN, STEP)
    msgs = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
    assert len(msgs) == 1
    assert f"{SKIP_UNMASKED}=2" in msgs[0]
