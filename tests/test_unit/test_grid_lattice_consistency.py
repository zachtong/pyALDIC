"""Grid builders agree on every range the reference-grid planner returns.

Three code paths turn ``gridxy_roi_range`` into node coordinates:
``integer_search`` (FFT path), the pyramid's ``_remap_to_standard_grid``
and ``build_grid_for_roi`` (seed propagation). A planned reference range
must give identical x0/y0 in all three, and those nodes must lie on the
frame-0 phase lattice; otherwise an equal-length U0 would silently shift
(step 1.3 of the incremental grid-coverage plan).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from al_dic.core._grid_extent import plan_reference_grids
from al_dic.core.data_structures import DICPara, GridxyROIRange
from al_dic.solver.integer_search import (
    _remap_to_standard_grid,
    integer_search,
    integer_search_pyramid,
)
from al_dic.solver.seed_prop_pipeline import build_grid_for_roi

SHAPE = (256, 384)
ROI0 = GridxyROIRange(gridx=(100, 220), gridy=(80, 170))
MASK0_RECT = (100, 220, 80, 170)
LEFT_RECT = (80, 200, 90, 160)


def rect_mask(rect, shape):
    """Float64 mask of ones on the inclusive (x_lo, x_hi, y_lo, y_hi) rect."""
    x_lo, x_hi, y_lo, y_hi = rect
    m = np.zeros(shape, dtype=np.float64)
    m[max(y_lo, 0):y_hi + 1, max(x_lo, 0):x_hi + 1] = 1.0
    return m


def speckle(shape, seed):
    return np.random.default_rng(seed).random(shape)


def planned_para(shape, win, step, roi0, rect0, rect_k):
    """(frame-0 para, planned K para); asserts that K triggered."""
    masks = [rect_mask(rect0, shape), rect_mask(rect_k, shape)]
    grids = plan_reference_grids(roi0, masks, {0, 1}, shape, win, step)
    assert [k for k, _ in grids.by_ref] == [1], "case must trigger"
    para0 = DICPara(winsize=win, winstepsize=step, img_size=shape,
                    gridxy_roi_range=roi0, size_of_fft_search_region=6)
    return para0, replace(para0, gridxy_roi_range=grids.roi_for(1))


CASES = {
    # name: (shape, winsize, step, roi0, mask-0 rect, mask-K rect)
    "left": (SHAPE, 32, 16, ROI0, MASK0_RECT, LEFT_RECT),
    "shrink_cap": (SHAPE, 32, 16, ROI0, MASK0_RECT, (300, 360, 90, 160)),
    "top_cap": (SHAPE, 32, 16, ROI0, MASK0_RECT, (110, 210, 0, 150)),
    "odd_win": (SHAPE, 31, 8, GridxyROIRange(gridx=(5, 220), gridy=(80, 170)),
                (5, 220, 80, 170), (2, 250, 60, 200)),
    "step4": ((300, 420), 24, 4,
              GridxyROIRange(gridx=(37, 151), gridy=(41, 199)),
              (37, 151, 41, 199), (120, 400, 10, 120)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_builders_agree_on_planned_range(name):
    shape, win, step, roi0, rect0, rect_k = CASES[name]
    h, w = shape
    para0, para = planned_para(shape, win, step, roi0, rect0, rect_k)

    xb, yb = build_grid_for_roi(para, h, w)
    img = speckle(shape, 7)
    xi, yi, *_ = integer_search(img, img, para)
    x_pyr, y_pyr = xb + 0.5, yb + 0.5  # off-lattice pyramid grid
    zeros = np.zeros((len(y_pyr), len(x_pyr)))
    xr, yr, *_ = _remap_to_standard_grid(
        x_pyr, y_pyr, zeros, zeros, zeros, para, h, w,
    )
    for x, y in ((xi, yi), (xr, yr)):
        np.testing.assert_array_equal(x, xb)
        np.testing.assert_array_equal(y, yb)
    # And every planned node lies on the frame-0 phase lattice.
    x0, y0 = build_grid_for_roi(para0, h, w)
    assert np.all(np.mod(xb - x0[0], step) == 0)
    assert np.all(np.mod(yb - y0[0], step) == 0)


def test_pyramid_search_end_to_end_agrees():
    shape = SHAPE
    _, para = planned_para(shape, 32, 16, ROI0, MASK0_RECT, LEFT_RECT)
    assert para.gridxy_roi_range == GridxyROIRange(gridx=(52, 220),
                                                   gridy=(64, 186))
    img = speckle(shape, 11)
    xp, yp, *_ = integer_search_pyramid(img, img, para)
    xb, yb = build_grid_for_roi(para, *shape)
    np.testing.assert_array_equal(xp, xb)
    np.testing.assert_array_equal(yp, yb)
