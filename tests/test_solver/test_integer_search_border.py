"""Nodes near the image border are searched with a clamped window.

A node's NCC search box (half subset + search radius on each side) used to have
to fit entirely inside the image, or the node was skipped and filled from its
neighbours later. With a large radius (auto-expand) that skipped most of a
specimen near the border, and the fill could not follow a non-uniform field.
The box is now clamped to the image, as the Starting Points search already
does, so every node whose match lies in the image is searched. These tests pin:

- border nodes are searched and accurate (FFT search and one pyramid level);
- a peak on a clamped edge never asks for a wider search, a peak on the
  search-radius edge still does;
- a specimen touching the border, rotated 5 degrees and moved 120 px, is
  measured end to end (it failed before: the fill missed the rotation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.solver.integer_search import _refine_at_level, integer_search

from tests.conftest import apply_displacement_lagrangian, generate_speckle

H, W = 256, 320
HW = 16  # half subset (winsize 32)


def _shifted(ref, du, dv):
    return apply_displacement_lagrangian(
        ref, lambda x, y: np.full_like(x, du), lambda x, y: np.full_like(y, dv),
    )


@pytest.fixture(scope="module")
def ref():
    return generate_speckle(H, W, sigma=3.0, seed=5)


def _para(search):
    return dicpara_default(
        winsize=2 * HW, winstepsize=8, img_size=(H, W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, W), gridy=(0, H)),
        size_of_fft_search_region=search, show_plots=False,
    )


def _measurable(x0, y0, du, dv):
    """Nodes whose matched subset lies inside the deformed image."""
    X, Y = np.meshgrid(x0, y0)
    return ((X + du - HW >= 0) & (X + du + HW + 1 <= W)
            & (Y + dv - HW >= 0) & (Y + dv + HW + 1 <= H))


def test_border_nodes_are_searched(ref):
    du, dv = 60.0, 0.0
    x0, y0, u, v, info = integer_search(ref, _shifted(ref, du, dv), _para(80))
    meas = _measurable(x0, y0, du, dv)
    found = np.isfinite(u) & np.isfinite(v)
    assert found[meas].mean() >= 0.95, f"searched {found[meas].mean():.2f}"
    err = np.hypot(u - du, v - dv)[found & meas]
    assert np.median(err) < 0.05
    assert err.max() < 0.5
    assert not info["peak_clipped"], "radius 80 covers the 60 px motion"


def test_radius_edge_peak_still_asks_for_a_wider_search(ref):
    _, _, _, _, info = integer_search(ref, _shifted(ref, 60.0, 0.0), _para(40))
    assert info["peak_clipped"]
    assert info["n_clipped"] > 0


def test_clamped_axis_does_not_hide_clipping_on_the_open_axis():
    """Every node's window is clamped in y (short image), none in x: a peak
    on the x radius edge must still ask for a wider search."""
    h = 96
    ref = generate_speckle(h, W, sigma=3.0, seed=11)
    para = dicpara_default(
        winsize=2 * HW, winstepsize=8, img_size=(h, W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, W), gridy=(0, h)),
        size_of_fft_search_region=40, show_plots=False,
    )
    x0, y0, _, _, info = integer_search(ref, _shifted(ref, 60.0, 0.0), para)
    assert np.all((y0 - HW - 40 < 0) | (y0 + HW + 1 + 40 > h)), "all y-clamped"
    assert info["peak_clipped"]


def test_pyramid_level_searches_border_nodes(ref):
    du, dv = 60.0, 0.0
    x0 = np.arange(HW, W - HW, 8, dtype=np.float64)
    y0 = np.arange(HW, H - HW, 8, dtype=np.float64)
    shape = (len(y0), len(x0))
    u, v, _ = _refine_at_level(
        ref, _shifted(ref, du, dv), x0, y0,
        np.full(shape, du), np.full(shape, dv), HW, 12,
    )
    meas = _measurable(x0, y0, du, dv)
    found = np.isfinite(u) & np.isfinite(v)
    assert found[meas].mean() >= 0.95, f"searched {found[meas].mean():.2f}"
    err = np.hypot(u - du, v - dv)[found & meas]
    assert np.median(err) < 0.05
    assert err.max() < 0.5


def test_rotated_specimen_touching_the_border():
    """A scaled-down version of a specimen that fills the frame height.

    The motion (5 degrees about the specimen centre + 120 px) makes the
    search widen past the border margin of most nodes; before the fix those
    nodes were filled from the middle with a uniform value and the ends of the
    specimen were off by tens of pixels.
    """
    h, w = 576, 480
    ref = generate_speckle(h, w, sigma=3.0, seed=7)
    mask = np.zeros((h, w))
    mask[8:568, 16:240] = 1.0
    cx, cy = 127.5, 287.5
    t = math.radians(5.0)
    c, s = math.cos(t), math.sin(t)

    def u(x, y):
        return cx + 120.0 + c * (x - cx) - s * (y - cy) - x

    def v(x, y):
        return cy + s * (x - cx) + c * (y - cy) - y

    deformed = apply_displacement_lagrangian(ref, u, v)
    mask_def = (apply_displacement_lagrangian(mask, u, v) > 0.5).astype(np.float64)
    para = dicpara_default(
        winsize=32, winstepsize=16, img_size=(h, w),
        gridxy_roi_range=GridxyROIRange(gridx=(16, 239), gridy=(8, 567)),
        size_of_fft_search_region=40, use_global_step=False, show_plots=False,
    )
    res = run_aldic(para, [ref, deformed], [mask, mask_def], compute_strain=False)
    xy = res.dic_mesh.coordinates_fem
    U = res.result_disp[0].U
    tx, ty = u(xy[:, 0], xy[:, 1]), v(xy[:, 0], xy[:, 1])
    inside = mask[np.round(xy[:, 1]).astype(int), np.round(xy[:, 0]).astype(int)] > 0
    lands = ((xy[:, 0] + tx >= HW) & (xy[:, 0] + tx < w - HW)
             & (xy[:, 1] + ty >= HW) & (xy[:, 1] + ty < h - HW))
    sel = inside & lands
    err = np.hypot(U[0::2] - tx, U[1::2] - ty)[sel]
    assert np.median(err) < 0.05, f"median error {np.median(err):.2f} px"
    assert (err < 0.5).mean() >= 0.95, f"{(err < 0.5).mean():.2f} within 0.5 px"
