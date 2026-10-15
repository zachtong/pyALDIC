"""Unit tests for the incremental reference-grid planner (core/_grid_extent.py).

In incremental mode, every reference frame K != 0 whose material leaves the
frame-0 grid range gets its own phase-aligned grid range. These tests pin:

- identity cases (``roi_for(k) is roi0``) that keep no-expansion runs
  byte-identical;
- the range-box trigger (not the node hull);
- the tight per-side growth/shrink rule with ``pad_steps`` rings;
- the skip reasons (a)-(d) (the all-ones skip: test_grid_extent_unmasked.py);
- the ``> 0.5`` material threshold and dtype independence;
- the half-window border cap;
- a seeded property loop against an enumeration oracle.

Lattice consistency of the three grid builders on planned ranges (step 1.3)
lives in ``test_grid_lattice_consistency.py``.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from al_dic.core._grid_extent import (
    GRID_GROWTH_WARN_MIN_NODES,
    GRID_GROWTH_WARN_RATIO,
    PAD_STEPS,
    SKIP_ALL_BORDERS,
    SKIP_EMPTY,
    SKIP_NO_NODES,
    SKIP_REASONS,
    SKIP_SAME_AS_FRAME0,
    SKIP_UNMASKED,
    ReferenceGrids,
    grid_node_count,
    log_grid_plan,
    plan_reference_grids,
)
from al_dic.core.data_structures import DICPara, FrameSchedule, GridxyROIRange
from al_dic.solver.integer_search import _centered_arange
from al_dic.solver.seed_prop_pipeline import build_grid_for_roi

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

H, W = 256, 384
WIN, STEP = 32, 16
HW = WIN // 2
SHAPE = (H, W)
# Frame-0 range (GUI style: the bbox of mask 0).
# x: effective 100..220, nodes 104..216 (offset 4)
# y: effective  80..170, nodes  85..165 (offset 5)
ROI0 = GridxyROIRange(gridx=(100, 220), gridy=(80, 170))
LOGGER_NAME = "al_dic.core._grid_extent"


def rect_mask(x_lo, x_hi, y_lo, y_hi, shape=SHAPE, value=1.0):
    """Float64 mask with ``value`` on the inclusive rectangle (clipped)."""
    m = np.zeros(shape, dtype=np.float64)
    h, w = shape
    xa, xb = max(x_lo, 0), min(x_hi, w - 1)
    ya, yb = max(y_lo, 0), min(y_hi, h - 1)
    if xa <= xb and ya <= yb:
        m[ya:yb + 1, xa:xb + 1] = value
    return m


def mask0():
    return rect_mask(100, 220, 80, 170)


def nodes_1d(lo, hi, dim, win=WIN, step=STEP):
    """Node coordinates on one axis, using the integer_search clamp."""
    hw = win // 2
    return _centered_arange(max(lo, hw), min(hi, dim - 1 - hw), step)


def lattice(roi, shape=SHAPE, win=WIN, step=STEP):
    h, w = shape
    return (
        nodes_1d(roi.gridx[0], roi.gridx[1], w, win, step),
        nodes_1d(roi.gridy[0], roi.gridy[1], h, win, step),
    )


def plan(masks, refs=None, roi0=ROI0, shape=SHAPE, win=WIN, step=STEP, **kw):
    if refs is None:
        refs = set(range(len(masks)))
    return plan_reference_grids(roi0, masks, refs, shape, win, step, **kw)


# Hand-computed expectations (pad_steps=1, see D5 in tasks/todo.md):
#   a = floor((b_lo - n_lo)/step) - pad, b = ceil((b_hi - n_hi)/step) + pad,
#   capped so the effective edges stay inside [hw, dim-1-hw].
LEFT_RECT = (80, 200, 90, 160)
LEFT_ROI = GridxyROIRange(gridx=(52, 220), gridy=(64, 186))
RIGHT_RECT = (130, 250, 90, 160)
RIGHT_ROI = GridxyROIRange(gridx=(100, 284), gridy=(64, 186))
TOP_RECT = (110, 210, 50, 150)
TOP_ROI = GridxyROIRange(gridx=(84, 236), gridy=(16, 186))
BOTTOM_RECT = (110, 210, 120, 200)
BOTTOM_ROI = GridxyROIRange(gridx=(84, 236), gridy=(96, 234))
INSIDE_RECT = (110, 210, 90, 160)


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_constants(self):
        assert PAD_STEPS == 1
        assert (GRID_GROWTH_WARN_RATIO, GRID_GROWTH_WARN_MIN_NODES) == (2.0, 5000)
        assert set(SKIP_REASONS) == {
            SKIP_EMPTY, SKIP_SAME_AS_FRAME0, SKIP_UNMASKED, SKIP_ALL_BORDERS,
            SKIP_NO_NODES,
        }
        assert len(SKIP_REASONS) == 5

    def test_reference_grids_is_frozen(self):
        grids = plan([mask0(), rect_mask(*LEFT_RECT)])
        assert isinstance(grids, ReferenceGrids)
        with pytest.raises(dataclasses.FrozenInstanceError):
            grids.roi0 = ROI0  # type: ignore[misc]

    def test_reference_grids_lookup(self):
        grids = ReferenceGrids(roi0=ROI0, by_ref=((2, LEFT_ROI), (5, RIGHT_ROI)))
        assert grids.roi_for(2) is LEFT_ROI
        assert grids.roi_for(5) is RIGHT_ROI
        assert grids.roi_for(0) is ROI0 and grids.roi_for(3) is ROI0
        assert not grids.is_identity
        assert ReferenceGrids(roi0=ROI0).is_identity

    def test_edges_are_python_ints(self):
        grids = plan([mask0(), rect_mask(*LEFT_RECT)])
        (ref, roi), = grids.by_ref
        assert ref == 1
        for v in (*roi.gridx, *roi.gridy):
            assert type(v) is int

    def test_inputs_not_mutated(self):
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*RIGHT_RECT)]
        before = [m.copy() for m in masks]
        plan(masks)
        for m, b in zip(masks, before, strict=True):
            assert m.dtype == b.dtype
            np.testing.assert_array_equal(m, b)

    def test_ref_order_and_duplicates(self):
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*INSIDE_RECT),
                 rect_mask(*RIGHT_RECT)]
        grids = plan(masks, refs=[3, 1, 1, 0, 3])
        assert [k for k, _ in grids.by_ref] == [1, 3]


class TestValidation:
    @pytest.mark.parametrize("kw", [
        {"step": 0}, {"step": -4}, {"win": 0}, {"pad_steps": -1},
    ])
    def test_bad_scalars_raise(self, kw):
        with pytest.raises(ValueError):
            plan([mask0(), rect_mask(*LEFT_RECT)], **kw)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            plan([mask0(), rect_mask(*LEFT_RECT)], shape=(0, W))

    def test_ref_out_of_range_raises(self):
        with pytest.raises(ValueError, match="reference frame"):
            plan([mask0(), rect_mask(*LEFT_RECT)], refs={0, 1, 5})

    def test_non_2d_mask_raises(self):
        bad = np.ones((H, W, 3))
        with pytest.raises(ValueError, match="2-D"):
            plan([mask0(), bad])


# ---------------------------------------------------------------------------
# Identity cases
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_accumulative_never_triggers(self):
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*RIGHT_RECT)]
        schedule = FrameSchedule.from_mode("accumulative", len(masks))
        grids = plan(masks, refs=schedule.ref_frame_set)
        assert grids.is_identity
        assert grids.by_ref == () and grids.skipped == ()
        for k in range(len(masks)):
            assert grids.roi_for(k) is ROI0
        assert grids.roi0 is ROI0

    def test_masks_inside_range(self):
        masks = [mask0(), rect_mask(*INSIDE_RECT), mask0() * 1.0]
        grids = plan(masks)
        assert grids.is_identity
        assert grids.roi_for(1) is ROI0
        assert grids.roi_for(2) is ROI0

    def test_inside_range_box_but_outside_node_hull_does_not_trigger(self):
        # Range edge x=100 but first node x=104: a mask edge at 101 lies
        # beyond the node hull yet inside the range box -> no trigger.
        grids = plan([mask0(), rect_mask(101, 219, 80, 170)])
        assert grids.is_identity
        assert grids.roi_for(1) is ROI0

    def test_just_outside_range_box_triggers(self):
        grids = plan([mask0(), rect_mask(99, 219, 80, 170)])
        assert not grids.is_identity
        assert [k for k, _ in grids.by_ref] == [1]

    def test_degenerate_frame0_range_plans_nothing(self):
        roi0 = GridxyROIRange()  # zero-size default
        grids = plan([mask0(), rect_mask(*LEFT_RECT)], roi0=roi0)
        assert grids.is_identity
        assert grids.roi_for(1) is roi0


# ---------------------------------------------------------------------------
# Growth / shrink
# ---------------------------------------------------------------------------


class TestGrowth:
    @pytest.mark.parametrize("rect, expected", [
        (LEFT_RECT, LEFT_ROI),
        (RIGHT_RECT, RIGHT_ROI),
        (TOP_RECT, TOP_ROI),
        (BOTTOM_RECT, BOTTOM_ROI),
    ], ids=["left", "right", "top", "bottom"])
    def test_each_side_grows(self, rect, expected):
        grids = plan([mask0(), rect_mask(*rect)])
        assert grids.by_ref == ((1, expected),)
        assert grids.roi_for(1) == expected
        assert not grids.is_identity
        # The lattice covers the mask bbox with one ring of nodes (none of
        # these cases is capped) and with fewer than two rings of slack.
        xs, ys = lattice(expected)
        x_lo, x_hi, y_lo, y_hi = rect
        for nodes, lo, hi in ((xs, x_lo, x_hi), (ys, y_lo, y_hi)):
            assert lo - 2 * STEP < nodes[0] <= lo - STEP
            assert hi + STEP <= nodes[-1] < hi + 2 * STEP

    def test_opposite_side_shrinks_to_tight_cover(self):
        # Specimen far to the right: low side shrinks, high side is capped.
        grids = plan([mask0(), rect_mask(300, 360, 90, 160)])
        roi = grids.roi_for(1)
        assert roi == GridxyROIRange(gridx=(276, 364), gridy=(64, 186))
        xs, _ = lattice(roi)
        assert xs[0] == 280.0 and xs[-1] == 360.0
        assert xs[0] > ROI0.gridx[0]  # shrunk
        assert xs[-1] + STEP > W - 1 - HW  # capped at the border band

    def test_independent_refs(self):
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*INSIDE_RECT),
                 rect_mask(*RIGHT_RECT)]
        grids = plan(masks)
        assert grids.by_ref == ((1, LEFT_ROI), (3, RIGHT_ROI))
        assert grids.roi_for(2) is ROI0
        assert grids.roi_for(0) is ROI0
        # Each ref's extent equals planning that ref on its own.
        alone = plan([masks[0], masks[3]])
        assert alone.roi_for(1) == grids.roi_for(3)

    def test_non_ref_masks_ignored_every_n(self):
        # every_n=2 over 5 frames: refs {0, 2}; masks 1 and 3 stick out.
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*INSIDE_RECT),
                 rect_mask(*RIGHT_RECT), rect_mask(*TOP_RECT)]
        schedule = FrameSchedule.from_every_n(2, len(masks))
        assert schedule.ref_frame_set == {0, 2}
        grids = plan(masks, refs=schedule.ref_frame_set)
        assert grids.is_identity
        assert grids.skipped == ()

    def test_non_ref_masks_ignored_custom_schedule(self):
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*RIGHT_RECT),
                 rect_mask(*TOP_RECT)]
        schedule = FrameSchedule(ref_indices=(0, 0, 2))
        grids = plan(masks, refs=schedule.ref_frame_set)
        assert grids.by_ref == ((2, RIGHT_ROI),)
        assert grids.roi_for(1) is ROI0
        assert grids.roi_for(3) is ROI0


# ---------------------------------------------------------------------------
# Skip reasons (D3)
# ---------------------------------------------------------------------------


class TestSkips:
    def test_a_empty_mask(self):
        grids = plan([mask0(), np.zeros(SHAPE)])
        assert grids.skipped == ((1, SKIP_EMPTY),)
        assert grids.is_identity
        assert grids.roi_for(1) is ROI0

    def test_b_same_object_as_mask0(self):
        # The API pattern [np.ones]*n with a range smaller than mask 0.
        ones = np.ones(SHAPE)
        roi0 = GridxyROIRange(gridx=(30, W - 31), gridy=(30, H - 31))
        grids = plan([ones] * 4, roi0=roi0)
        assert grids.skipped == (
            (1, SKIP_SAME_AS_FRAME0), (2, SKIP_SAME_AS_FRAME0),
            (3, SKIP_SAME_AS_FRAME0),
        )
        assert grids.is_identity
        assert grids.roi_for(2) is roi0

    def test_b_equal_by_value_and_by_material(self):
        m0 = mask0()
        grids = plan([m0, m0.copy(), (m0 > 0.5).astype(np.uint8) * 255,
                      m0 > 0.5])
        assert grids.skipped == (
            (1, SKIP_SAME_AS_FRAME0), (2, SKIP_SAME_AS_FRAME0),
            (3, SKIP_SAME_AS_FRAME0),
        )
        assert grids.is_identity

    def test_b_inherited_mask0_larger_than_range_does_not_grow(self):
        # mask 0 is bigger than the user range; refs inherit it.
        m0 = rect_mask(40, 300, 40, 220)
        grids = plan([m0, m0, m0.copy()])
        assert grids.is_identity
        assert {r for _, r in grids.skipped} == {SKIP_SAME_AS_FRAME0}

    def test_c_four_border_mask_with_four_border_mask0(self):
        m0 = np.ones(SHAPE)
        m0[100:110, :200] = 0.0  # crack
        mk = np.ones(SHAPE)
        mk[120:130, 50:300] = 0.0  # a different crack
        grids = plan([m0, mk])
        assert grids.skipped == ((1, SKIP_ALL_BORDERS),)
        assert grids.is_identity

    def test_c_four_border_mask_with_specimen_mask0_not_skipped(self):
        mk = np.ones(SHAPE)
        mk[120:130, 50:300] = 0.0
        grids = plan([mask0(), mk])
        assert grids.skipped == ()
        roi = grids.roi_for(1)
        assert roi is not ROI0
        xs, ys = lattice(roi)
        # Grows to the capped full lattice on every side.
        assert xs[0] - STEP < HW and xs[-1] + STEP > W - 1 - HW
        assert ys[0] - STEP < HW and ys[-1] + STEP > H - 1 - HW

    def test_c_top_bottom_only_mask_not_skipped(self):
        # A vertical band touching only the top and bottom borders
        # (as in a user's cyclic-loading masks), with a four-border mask 0.
        mk = rect_mask(150, 300, 0, H - 1)
        grids = plan([np.ones(SHAPE), mk])
        assert grids.skipped == ()
        assert [k for k, _ in grids.by_ref] == [1]

    @pytest.mark.parametrize("rect", [
        (0, 10, 90, 160),            # left band only
        (W - 10, W - 1, 90, 160),    # right band only
        (110, 210, 0, HW - 1),       # top band only
        (110, 210, H - HW, H - 1),   # bottom band only
    ], ids=["left", "right", "top", "bottom"])
    def test_d_mask_inside_border_band(self, rect):
        grids = plan([mask0(), rect_mask(*rect)])
        assert grids.skipped == ((1, SKIP_NO_NODES),)
        assert grids.is_identity

    def test_d_capped_lattice_with_fewer_than_two_nodes(self):
        # x bbox 5..16 reaches the measurable band only at x=16. With
        # pad 0 the capped lattice keeps a single node column -> skip;
        # with pad 1 it keeps two columns -> planned.
        masks = [mask0(), rect_mask(5, 16, 90, 160)]
        g0 = plan(masks, pad_steps=0)
        assert g0.skipped == ((1, SKIP_NO_NODES),)
        assert g0.is_identity
        g1 = plan(masks, pad_steps=1)
        assert g1.skipped == ()
        xs, _ = lattice(g1.roi_for(1))
        assert len(xs) == 2

    def test_skip_order_empty_before_same(self):
        grids = plan([np.zeros(SHAPE), np.zeros(SHAPE)])
        assert grids.skipped == ((1, SKIP_EMPTY),)


# ---------------------------------------------------------------------------
# Material threshold and dtype independence
# ---------------------------------------------------------------------------


class TestThreshold:
    @pytest.mark.parametrize("halo, triggers", [
        (0.3, False), (0.5, False), (0.51, True),
    ])
    def test_soft_halo(self, halo, triggers):
        mk = rect_mask(60, 250, 60, 200, value=halo)
        mk[90:161, 110:211] = 1.0  # material core inside the range
        grids = plan([mask0(), mk])
        assert (not grids.is_identity) is triggers
        if not triggers:
            assert grids.roi_for(1) is ROI0
            assert grids.skipped == ()

    def test_all_soft_mask_is_empty(self):
        grids = plan([mask0(), np.full(SHAPE, 0.3)])
        assert grids.skipped == ((1, SKIP_EMPTY),)

    def test_dtypes_give_identical_plans(self):
        rects = [None, LEFT_RECT, INSIDE_RECT, RIGHT_RECT, (0, 10, 90, 160)]
        base = [mask0() > 0.5] + [
            rect_mask(*r) > 0.5 for r in rects[1:]
        ] + [np.zeros(SHAPE, bool), mask0() > 0.5]
        variants = {
            "bool": base,
            "uint8": [m.astype(np.uint8) for m in base],
            "uint8_255": [m.astype(np.uint8) * 255 for m in base],
            "float64": [m.astype(np.float64) for m in base],
        }
        plans = {k: plan(v) for k, v in variants.items()}
        ref = plans["float64"]
        assert ref.by_ref == ((1, LEFT_ROI), (3, RIGHT_ROI))
        assert ref.skipped == (
            (4, SKIP_NO_NODES), (5, SKIP_EMPTY), (6, SKIP_SAME_AS_FRAME0),
        )
        for name, grids in plans.items():
            assert grids == ref, name


# ---------------------------------------------------------------------------
# Border cap and padding
# ---------------------------------------------------------------------------


class TestCapAndPad:
    def test_half_window_cap_left(self):
        grids = plan([mask0(), rect_mask(0, 150, 90, 160)])
        roi = grids.roi_for(1)
        assert roi.gridx == (20, 172)
        xs, _ = lattice(roi)
        assert xs[0] == 24.0 and xs[0] - STEP < HW

    def test_half_window_cap_right_and_bottom(self):
        grids = plan([mask0(), rect_mask(300, W - 1, 120, H - 1)])
        roi = grids.roi_for(1)
        assert roi.gridx[1] == 364 and roi.gridy[1] == 234
        assert roi.gridx[1] <= W - 1 - HW and roi.gridy[1] <= H - 1 - HW
        assert roi.gridx[1] + STEP > W - 1 - HW
        assert roi.gridy[1] + STEP > H - 1 - HW

    def test_range_below_half_window_capped_to_same_lattice_is_identity(self):
        # Range starts at x=5 < half_w: effective edge 16, nodes 22..214.
        # The mask leaves the range box (x=2 < 5) but the capped tight
        # cover is exactly the frame-0 lattice -> identity, no rebuild.
        roi0 = GridxyROIRange(gridx=(5, 220), gridy=(80, 170))
        grids = plan([rect_mask(5, 220, 80, 170), rect_mask(2, 198, 101, 149)],
                     roi0=roi0)
        assert grids.is_identity
        assert grids.roi_for(1) is roi0
        assert grids.skipped == ()

    def test_range_below_half_window_grows_other_side(self):
        roi0 = GridxyROIRange(gridx=(5, 220), gridy=(80, 170))
        grids = plan([rect_mask(5, 220, 80, 170), rect_mask(2, 250, 101, 149)],
                     roi0=roi0)
        roi = grids.roi_for(1)
        assert roi == GridxyROIRange(gridx=(16, 284), gridy=(80, 170))
        xs, _ = lattice(roi)
        x0, _ = lattice(roi0)
        assert xs[0] == x0[0] == 22.0

    def test_pad_zero_and_one(self):
        masks = [mask0(), rect_mask(*LEFT_RECT)]
        r0 = plan(masks, pad_steps=0).roi_for(1)
        r1 = plan(masks, pad_steps=1).roi_for(1)
        assert r0 == GridxyROIRange(gridx=(68, 204), gridy=(80, 170))
        assert r1 == LEFT_ROI
        # pad 1 is pad 0 plus one ring on every (uncapped) side.
        assert r1.gridx == (r0.gridx[0] - STEP, r0.gridx[1] + STEP)
        assert r1.gridy == (r0.gridy[0] - STEP, r0.gridy[1] + STEP)
        # pad 0 still covers the bbox.
        xs, ys = lattice(r0)
        assert xs[0] <= 80 and xs[-1] >= 200 and ys[0] <= 90 and ys[-1] >= 160

    def test_default_pad_is_pad_steps(self):
        masks = [mask0(), rect_mask(*LEFT_RECT)]
        assert plan(masks) == plan(masks, pad_steps=PAD_STEPS)


# ---------------------------------------------------------------------------
# Seeded property loop against an enumeration oracle
# ---------------------------------------------------------------------------


def _bbox(material):
    cols = np.flatnonzero(material.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(material.any(axis=1))
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def _oracle_axis(g_lo, g_hi, b_lo, b_hi, dim, hw, step, pad):
    """Tight phase-aligned node pair by enumeration (independent of D5's
    floor/ceil arithmetic). Returns (lo_node, hi_node, lo_gap, hi_gap)."""
    e_lo, e_hi = max(g_lo, hw), min(g_hi, dim - 1 - hw)
    nodes = _centered_arange(e_lo, e_hi, step)
    lo_gap, hi_gap = nodes[0] - e_lo, e_hi - nodes[-1]
    k = np.arange(-(dim // step) - 3, 2 * (dim // step) + 4)
    pos = nodes[0] + k * step
    lo_ok = pos[pos - lo_gap >= hw]
    hi_ok = pos[pos + hi_gap <= dim - 1 - hw]
    want_lo = pos[pos <= b_lo - pad * step]
    want_hi = pos[pos >= b_hi + pad * step]
    lo = max(want_lo.max(), lo_ok.min())
    hi = min(want_hi.min(), hi_ok.max())
    return lo, hi, lo_gap, hi_gap


def _oracle(roi0, m0, mk, shape, win, step, pad):
    """Expected outcome for ref 1: ('skip', reason) | ('roi0', why) |
    ('roi', range)."""
    h, w = shape
    hw = win // 2
    mat_k, mat_0 = mk > 0.5, m0 > 0.5
    bk, b0 = _bbox(mat_k), _bbox(mat_0)
    if bk is None:
        return ("skip", SKIP_EMPTY)
    if np.array_equal(mat_k, mat_0):
        return ("skip", SKIP_SAME_AS_FRAME0)
    if mat_k.all():
        return ("skip", SKIP_UNMASKED)

    def four(b):
        return b is not None and b == (0, w - 1, 0, h - 1)

    if four(bk) and four(b0):
        return ("skip", SKIP_ALL_BORDERS)
    if (min(bk[1], w - 1 - hw) < max(bk[0], hw)
            or min(bk[3], h - 1 - hw) < max(bk[2], hw)):
        return ("skip", SKIP_NO_NODES)
    inside = (bk[0] >= roi0.gridx[0] and bk[1] <= roi0.gridx[1]
              and bk[2] >= roi0.gridy[0] and bk[3] <= roi0.gridy[1])
    if inside:
        return ("roi0", "inside")
    ax = _oracle_axis(*roi0.gridx, bk[0], bk[1], w, hw, step, pad)
    ay = _oracle_axis(*roi0.gridy, bk[2], bk[3], h, hw, step, pad)
    if ax[1] - ax[0] < step or ay[1] - ay[0] < step:
        return ("skip", SKIP_NO_NODES)
    gx = (int(ax[0] - ax[2]), int(ax[1] + ax[3]))
    gy = (int(ay[0] - ay[2]), int(ay[1] + ay[3]))
    e0 = ((max(roi0.gridx[0], hw), min(roi0.gridx[1], w - 1 - hw)),
          (max(roi0.gridy[0], hw), min(roi0.gridy[1], h - 1 - hw)))
    if (gx, gy) == e0:
        return ("roi0", "capped_to_frame0")
    return ("roi", GridxyROIRange(gridx=gx, gridy=gy))


def _random_span(rng, dim):
    lo = int(rng.integers(0, dim - 1))
    hi = int(min(dim - 1, lo + rng.integers(0, dim)))
    return lo, hi


def _capped_back_rect(rng, shape, win, step, pad, roi0):
    """A rect that leaves the range box only into the low x border band,
    so the capped tight cover is the frame-0 lattice itself."""
    (h, w), hw = shape, win // 2
    xs = nodes_1d(*roi0.gridx, w, win, step).astype(int)
    ys = nodes_1d(*roi0.gridy, h, win, step).astype(int)
    x_lo = int(rng.integers(0, roi0.gridx[0]))
    x_hi = int(rng.integers(xs[-1] - (pad + 1) * step + 1,
                            xs[-1] - pad * step + 1))
    y_lo = int(rng.integers(ys[0] + pad * step, ys[0] + (pad + 1) * step))
    y_hi = int(rng.integers(ys[-1] - (pad + 1) * step + 1,
                            ys[-1] - pad * step + 1))
    assert x_lo < roi0.gridx[0] <= hw
    return x_lo, x_hi, y_lo, y_hi


def _random_ref_mask(rng, kind, shape, win, step, pad, roi0, m0):
    """Mask K for one random case (and mask 0, which some kinds replace)."""
    h, w = shape
    gx, gy = roi0.gridx, roi0.gridy
    if kind < 0.04:
        return m0, np.zeros(shape, bool)
    if kind < 0.06:
        return m0, m0.copy()
    if kind < 0.08:
        return m0, np.ones(shape, bool)
    if kind < 0.24:
        # Inside the range box, possibly reaching into the border band.
        x_lo = int(rng.integers(gx[0], gx[1] + 1))
        y_lo = int(rng.integers(gy[0], gy[1] + 1))
        x_hi = int(rng.integers(x_lo, gx[1] + 1))
        y_hi = int(rng.integers(y_lo, gy[1] + 1))
        return m0, rect_mask(x_lo, x_hi, y_lo, y_hi, shape=shape) > 0.5
    if kind < 0.30:
        m0 = np.ones(shape, bool)
        m0[h // 3, : w // 2] = False
        mk = np.ones(shape, bool)
        mk[int(rng.integers(0, h)), int(rng.integers(0, w))] = False
        return m0, mk
    if kind < 0.38:
        rect = _capped_back_rect(rng, shape, win, step, pad, roi0)
        return m0, rect_mask(*rect, shape=shape) > 0.5
    x_lo = int(rng.integers(-w // 4, w))
    y_lo = int(rng.integers(-h // 4, h))
    x_hi = x_lo + int(rng.integers(0, w))
    y_hi = y_lo + int(rng.integers(0, h))
    return m0, rect_mask(x_lo, x_hi, y_lo, y_hi, shape=shape) > 0.5


def _random_case(rng):
    """One random (shape, win, step, pad, roi0, m0, mk) with a frame-0
    lattice of >= 2 nodes per axis, or None to redraw."""
    shape = (int(rng.integers(96, 420)), int(rng.integers(96, 520)))
    win = int(rng.choice([8, 12, 16, 24, 31, 32]))
    step = int(rng.choice([2, 4, 6, 8, 10, 16]))
    pad = int(rng.choice([0, 1, 1, 2]))
    kind = rng.random()
    gx, gy = _random_span(rng, shape[1]), _random_span(rng, shape[0])
    if 0.30 <= kind < 0.38:
        # Range starting inside the low x border band (but not at 0).
        gx = (int(rng.integers(1, win // 2 + 1)), gx[1])
    roi0 = GridxyROIRange(gridx=gx, gridy=gy)
    xs, ys = lattice(roi0, shape, win, step)
    if len(xs) < 2 or len(ys) < 2:
        return None
    m0 = rect_mask(gx[0], gx[1], gy[0], gy[1], shape=shape) > 0.5
    m0, mk = _random_ref_mask(rng, kind, shape, win, step, pad, roi0, m0)
    return shape, win, step, pad, roi0, m0, mk


def _check_planned(roi, roi0, bk, shape, win, step, pad, stats):
    """Semantic properties of one planned K range."""
    h, w = shape
    hw = win // 2
    for v in (*roi.gridx, *roi.gridy):
        assert type(v) is int
    x0, y0 = lattice(roi0, shape, win, step)
    xs, ys = lattice(roi, shape, win, step)
    for nodes, base, (e_lo, e_hi), (g_lo, g_hi), (b_lo, b_hi), dim in (
        (xs, x0, roi.gridx, roi0.gridx, bk[:2], w),
        (ys, y0, roi.gridy, roi0.gridy, bk[2:], h),
    ):
        lim_hi = dim - 1 - hw
        # The range needs no clamp: integer_search sees it verbatim.
        assert hw <= e_lo and e_hi <= lim_hi
        assert len(nodes) >= 2
        # Phase: every K node lies on the frame-0 lattice.
        assert np.all(np.mod(nodes - base[0], step) == 0)
        cap_lo = e_lo - step < hw
        cap_hi = e_hi + step > lim_hi
        # Coverage: >= pad rings beyond the bbox unless capped.
        assert nodes[0] <= b_lo - pad * step or cap_lo
        assert nodes[-1] >= b_hi + pad * step or cap_hi
        # Tightness: < pad + 1 rings beyond the bbox.
        assert nodes[0] > b_lo - (pad + 1) * step
        assert nodes[-1] < b_hi + (pad + 1) * step
        stats["capped"] += int(cap_lo or cap_hi)
        e0_lo, e0_hi = max(g_lo, hw), min(g_hi, lim_hi)
        stats["shrunk"] += int(e_lo > e0_lo or e_hi < e0_hi)
    grows_only = (
        roi.gridx[0] <= max(roi0.gridx[0], hw)
        and roi.gridx[1] >= min(roi0.gridx[1], w - 1 - hw)
        and roi.gridy[0] <= max(roi0.gridy[0], hw)
        and roi.gridy[1] >= min(roi0.gridy[1], h - 1 - hw)
    )
    if grows_only:
        stats["grows_only"] += 1
        assert set(x0.tolist()) <= set(xs.tolist())
        assert set(y0.tolist()) <= set(ys.tolist())


N_PROPERTY_CASES = 2500


def _run_property_case(case, stats):
    """Plan one random case and check it against the oracle + invariants."""
    shape, win, step, pad, roi0, m0, mk = case
    grids = plan_reference_grids(
        roi0, [m0, mk], {0, 1}, shape, win, step, pad_steps=pad,
    )
    expected = _oracle(roi0, m0, mk, shape, win, step, pad)
    ctx = (shape, win, step, pad, roi0, _bbox(mk))
    if expected[0] == "skip":
        stats[expected[1]] += 1
        assert grids.skipped == ((1, expected[1]),), ctx
        assert grids.is_identity and grids.roi_for(1) is roi0, ctx
        return
    assert grids.skipped == (), ctx
    if expected[0] == "roi0":
        stats[expected[1]] += 1
        assert grids.is_identity and grids.roi_for(1) is roi0, ctx
        return
    stats["planned"] += 1
    roi = grids.roi_for(1)
    assert grids.by_ref == ((1, roi),), ctx
    assert roi == expected[1], (ctx, roi, expected[1])
    _check_planned(roi, roi0, _bbox(mk), shape, win, step, pad, stats)
    # Idempotence: planning from the planned range is a no-op.
    again = plan_reference_grids(
        roi, [m0, mk], {0, 1}, shape, win, step, pad_steps=pad,
    )
    assert again.is_identity and again.roi_for(1) is roi, ctx
    stats["idempotent"] += 1


def test_property_loop_matches_oracle_and_invariants():
    rng = np.random.default_rng(20261015)
    stats = dict.fromkeys((
        "cases", "planned", "inside", "capped_to_frame0", "capped", "shrunk",
        "grows_only", "idempotent", *SKIP_REASONS,
    ), 0)
    while stats["cases"] < N_PROPERTY_CASES:
        case = _random_case(rng)
        if case is None:
            continue
        stats["cases"] += 1
        _run_property_case(case, stats)
    # The generator must exercise every branch meaningfully.
    assert stats["planned"] >= 800, stats
    assert stats["inside"] >= 100, stats
    assert stats["capped_to_frame0"] >= 20, stats
    assert stats["capped"] >= 200, stats
    assert stats["shrunk"] >= 200, stats
    assert stats["grows_only"] >= 50, stats
    for reason in SKIP_REASONS:
        assert stats[reason] >= 20, stats


# ---------------------------------------------------------------------------
# Node count and logging
# ---------------------------------------------------------------------------


class TestCountAndLog:
    def test_grid_node_count_matches_builder(self):
        for roi in (ROI0, LEFT_ROI, RIGHT_ROI, TOP_ROI, BOTTOM_ROI):
            para = DICPara(winsize=WIN, winstepsize=STEP, img_size=SHAPE,
                           gridxy_roi_range=roi)
            x0, y0 = build_grid_for_roi(para, H, W)
            assert grid_node_count(roi, SHAPE, WIN, STEP) == len(x0) * len(y0)

    def test_grid_node_count_empty_range(self):
        assert grid_node_count(GridxyROIRange(), SHAPE, WIN, STEP) == 0

    def test_log_identity(self, caplog):
        grids = plan([mask0(), rect_mask(*INSIDE_RECT), np.zeros(SHAPE)])
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_grid_plan(grids, SHAPE, WIN, STEP)
        recs = [r for r in caplog.records if r.name == LOGGER_NAME]
        assert [r.levelno for r in recs] == [logging.INFO]
        msg = recs[0].getMessage()
        n0 = grid_node_count(ROI0, SHAPE, WIN, STEP)
        assert str(n0) in msg
        assert f"{SKIP_EMPTY}=1" in msg

    def test_log_planned_refs(self, caplog):
        masks = [mask0(), rect_mask(*LEFT_RECT), rect_mask(*RIGHT_RECT)]
        grids = plan(masks)
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_grid_plan(grids, SHAPE, WIN, STEP)
        recs = [r for r in caplog.records if r.name == LOGGER_NAME]
        assert [r.levelno for r in recs] == [logging.INFO]
        msg = recs[0].getMessage()
        n_left = grid_node_count(LEFT_ROI, SHAPE, WIN, STEP)
        n_right = grid_node_count(RIGHT_ROI, SHAPE, WIN, STEP)
        largest_ref = 1 if n_left >= n_right else 2
        assert "2 reference frame(s)" in msg
        assert str(max(n_left, n_right)) in msg
        assert f"image index {largest_ref}" in msg

    @pytest.mark.parametrize("step, warns", [(2, True), (STEP, False)])
    def test_log_growth_warning(self, caplog, step, warns):
        # A stray component spanning the image inflates ref 2's grid; the
        # coarse grid (step 16) stays below GRID_GROWTH_WARN_MIN_NODES extra.
        stray = rect_mask(*INSIDE_RECT)
        stray[:, :20] = 1.0
        stray[200:, :] = 1.0
        grids = plan([mask0(), rect_mask(*LEFT_RECT), stray], step=step)
        n0, n1, n2 = (grid_node_count(grids.roi_for(k), SHAPE, WIN, step)
                      for k in (0, 1, 2))
        assert n1 <= GRID_GROWTH_WARN_RATIO * n0 < n2
        assert (n2 - n0 >= GRID_GROWTH_WARN_MIN_NODES) is warns
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_grid_plan(grids, SHAPE, WIN, step)
        recs = [r for r in caplog.records
                if r.name == LOGGER_NAME and r.levelno == logging.WARNING]
        assert len(recs) == int(warns)
        assert not warns or ("[2]" in recs[0].getMessage()
                             and str(n0) in recs[0].getMessage())
