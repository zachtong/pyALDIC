"""VizController.render_field with "Hide untracked regions"."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from al_dic.gui.controllers.viz_controller import VizController

H = W = 96
STEP = 8


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _alpha(pixmap):
    img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = img.height(), img.width()
    raw = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
    return raw.reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)[:, :, 3] > 0


def _mesh():
    xs = np.arange(STEP, W - STEP + 1, STEP, float)
    XX, YY = np.meshgrid(xs, xs)
    return np.column_stack([XX.ravel(), YY.ravel()])


def _render(nodes, values, *, hide, tracked=None, roi=None, blank=False, ctrl=None,
            field="disp_u"):
    ctrl = ctrl or VizController()
    roi = np.ones((H, W), bool) if roi is None else roi
    pm, xg, yg, step = ctrl.render_field(
        0, field, nodes, values, (H, W), STEP, vmin=-2.0, vmax=2.0, roi_mask=roi,
        blank_invalid_nodes=blank, hide_untracked=hide, tracked=tracked)
    return _alpha(pm), xg, yg, ctrl


def _at(alpha, xg, yg, x, y):
    """Is image point (x, y) painted?  Points outside the render grid are not."""
    step = xg[0, 1] - xg[0, 0]
    if not (xg[0, 0] - step <= x <= xg[0, -1] + step and yg[0, 0] - step <= y <= yg[-1, 0] + step):
        return False
    j = int(np.argmin(np.abs(xg[0] - x)))
    i = int(np.argmin(np.abs(yg[:, 0] - y)))
    return bool(alpha[i, j])


def test_dead_region_is_transparent_in_reference_view(qapp):
    nodes = _mesh()
    values = np.ones(len(nodes))
    dead = nodes[:, 0] >= 56
    values[dead] = np.nan
    a_off, xg, yg, _ = _render(nodes, values, hide=False)
    a_on, xg2, yg2, _ = _render(nodes, values, hide=True, tracked=~dead)
    assert _at(a_off, xg, yg, 85, 50)            # legacy: nearest-filled
    assert not _at(a_on, xg2, yg2, 85, 50)       # hidden: > reach from tracked
    assert _at(a_on, xg2, yg2, 30, 50)           # tracked interior painted
    assert _at(a_on, xg2, yg2, 55, 50)           # within reach of x=48 nodes


def test_dead_nodes_with_nan_positions_deformed_view(qapp):
    nodes = _mesh()
    values = np.ones(len(nodes))
    dead = nodes[:, 1] >= 56
    pos = nodes.copy()
    pos[dead] = np.nan                            # deformed view: no position
    values[dead] = np.nan
    a_on, xg, yg, _ = _render(pos, values, hide=True, tracked=~dead)
    assert _at(a_on, xg, yg, 40, 20)
    assert not _at(a_on, xg, yg, 40, 85)


def test_edges_are_painted_to_the_roi_edge(qapp):
    nodes = _mesh()
    values = nodes[:, 0] / 50.0
    a_off, *_ = _render(nodes, values, hide=False)
    a_on, xg, yg, _ = _render(nodes, values, hide=True, tracked=np.ones(len(nodes), bool))
    assert a_on.sum() > a_off.sum()               # border band beyond the node hull
    assert _at(a_on, xg, yg, 2, 2) and _at(a_on, xg, yg, W - 2, H - 2)


def test_crack_stays_sharp(qapp):
    nodes = _mesh()
    roi = np.ones((H, W), bool)
    roi[51:53, :] = False                          # full-width slit between rows 48 and 56
    values = np.where(nodes[:, 1] < 52, -1.0, 1.0)
    ctrl = VizController()
    ctrl.render_field(0, "disp_u", nodes, values, (H, W), STEP, vmin=-2.0, vmax=2.0,
                      roi_mask=roi, hide_untracked=True, tracked=np.ones(len(nodes), bool))
    grid, xg, yg, _ = next(iter(ctrl._interp_cache.values()))
    i_above = int(np.argmin(np.abs(yg[:, 0] - 50)))
    i_below = int(np.argmin(np.abs(yg[:, 0] - 54)))
    assert np.nanmax(grid[i_above]) <= -0.99
    assert np.nanmin(grid[i_below]) >= 0.99


def test_strain_trim_still_shown_when_blanking(qapp):
    nodes = _mesh()
    strain = np.ones(len(nodes))
    trimmed = (nodes[:, 0] == STEP) | (nodes[:, 0] == W - STEP)
    strain[trimmed] = np.nan
    tracked = np.ones(len(nodes), bool)               # trimmed but tracked
    a_on, xg, yg, _ = _render(nodes, strain, hide=True, tracked=tracked, blank=True,
                              field="strain_exx")
    assert not _at(a_on, xg, yg, STEP, 48)         # trimmed band stays transparent
    assert _at(a_on, xg, yg, 48, 48)


def test_fill_trimmed_edges_stops_at_untracked_region(qapp):
    nodes = _mesh()
    strain = np.ones(len(nodes))
    trimmed = nodes[:, 0] == STEP
    dead = nodes[:, 0] >= 72
    strain[trimmed | dead] = np.nan
    tracked = ~dead
    a_on, xg, yg, _ = _render(nodes, strain, hide=True, tracked=tracked, blank=False,
                              field="strain_exx")
    assert _at(a_on, xg, yg, STEP, 48)             # trim band filled (tracked)
    assert not _at(a_on, xg, yg, 90, 48)           # untracked: not filled


def test_on_and_off_do_not_share_cache_entries(qapp):
    nodes = _mesh()
    values = np.ones(len(nodes))
    dead = nodes[:, 0] >= 56
    values[dead] = np.nan
    ctrl = VizController()
    _render(nodes, values, hide=True, tracked=~dead, ctrl=ctrl)
    a_off_after, *_ = _render(nodes, values, hide=False, ctrl=ctrl)
    a_off_fresh, *_ = _render(nodes, values, hide=False)
    np.testing.assert_array_equal(a_off_after, a_off_fresh)


def test_tracked_defaults_to_finite_values_and_positions(qapp):
    nodes = _mesh()
    values = np.ones(len(nodes))
    values[nodes[:, 0] >= 56] = np.nan
    a_default, *_ = _render(nodes, values, hide=True, tracked=None)
    a_explicit, *_ = _render(nodes, values, hide=True, tracked=np.isfinite(values))
    np.testing.assert_array_equal(a_default, a_explicit)


def test_without_a_mask_nothing_is_painted_beyond_the_node_hull(qapp):
    nodes = _mesh()
    ctrl = VizController()
    pm, xg, yg, _ = ctrl.render_field(0, "disp_u", nodes, nodes[:, 0] / 50.0, (H, W), STEP,
                                      vmin=-2.0, vmax=2.0, roi_mask=None,
                                      hide_untracked=True, tracked=np.ones(len(nodes), bool))
    a = _alpha(pm)
    assert _at(a, xg, yg, 48, 48)
    assert not _at(a, xg, yg, 2, 2)


def test_deformed_view_without_frame_mask_uses_the_warped_roi(qapp):
    """No per-frame ROI: the screen must warp the reference ROI like the export
    does, so trimmed edges inside it are filled exactly as in the exported file."""
    from al_dic.export.export_png import _compute_warped_mask

    nodes = _mesh()
    u = np.full(len(nodes), 3.0)
    v = np.zeros(len(nodes))
    strain = np.ones(len(nodes))
    ring = (nodes[:, 0] == STEP) | (nodes[:, 0] == W - STEP)
    strain[ring] = np.nan
    roi = np.ones((H, W), bool)
    deformed = nodes + np.column_stack([u, v])
    ctrl = VizController()
    pm, xg, yg, _ = ctrl.render_field(
        0, "strain_exx", deformed, strain, (H, W), STEP, vmin=0.0, vmax=2.0, roi_mask=roi,
        deformed=True, ref_uv=(u, v), deformed_mask=None, blank_invalid_nodes=False,
        hide_untracked=True, tracked=np.ones(len(nodes), bool))
    a = _alpha(pm)
    warped = _compute_warped_mask(nodes, deformed, roi, (H, W))
    # x = 13: inside the warped ROI (it starts at the shifted ring, x = 11),
    # between the trimmed ring node and the first valid node (x = 19).
    assert warped[48, 12:15].all()
    assert _at(a, xg, yg, 13, 48)                # trimmed ring inside the warped ROI: filled


def test_invalidate_masks_drops_mask_dependent_grids(qapp):
    nodes = _mesh()
    ctrl = VizController()
    ctrl.render_field(0, "disp_u", nodes, np.ones(len(nodes)), (H, W), STEP, vmin=0.0,
                      vmax=2.0, roi_mask=np.ones((H, W), bool), hide_untracked=True,
                      tracked=np.ones(len(nodes), bool))
    assert any(k[3] == "tracked" for k in ctrl._interp_cache)
    ctrl.invalidate_masks()
    assert not any(k[3] == "tracked" for k in ctrl._interp_cache)
