"""Tests for canvas_area._apply_preview_refinement.

Verify that the preview-mesh helper actually subdivides elements when
refinement is enabled, and that it gracefully handles disabled / failure
cases without raising into the GUI.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    for sig in (
        state.images_changed,
        state.current_frame_changed,
        state.roi_changed,
        state.params_changed,
        state.run_state_changed,
        state.progress_updated,
        state.results_changed,
        state.display_changed,
        state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


def _build_uniform_q4(
    x_min: int, x_max: int, y_min: int, y_max: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a uniform Q8-padded mesh matching what _generate_preview_mesh
    constructs (Q4 corners with -1 in cols 4-7)."""
    x0 = np.arange(x_min, x_max + 1, step, dtype=np.float64)
    y0 = np.arange(y_min, y_max + 1, step, dtype=np.float64)
    nx, ny = len(x0), len(y0)
    xx, yy = np.meshgrid(x0, y0, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii_flat, jj_flat = ii.ravel(), jj.ravel()
    n0 = ii_flat * ny + jj_flat
    n1 = (ii_flat + 1) * ny + jj_flat
    n2 = (ii_flat + 1) * ny + (jj_flat + 1)
    n3 = ii_flat * ny + (jj_flat + 1)

    elements = np.full((len(n0), 8), -1, dtype=np.int64)
    elements[:, 0] = n0
    elements[:, 1] = n1
    elements[:, 2] = n2
    elements[:, 3] = n3
    return coords, elements


def _make_canvas_stub(state: AppState):
    """Build the smallest stub that exposes _apply_preview_refinement.

    We avoid constructing the real CanvasArea (which needs a full GUI tree).
    The helper only depends on ``self._state``, so a tiny shim is enough.
    """
    from al_dic.gui.panels.canvas_area import CanvasArea

    class _Shim:
        def __init__(self, state):
            self._state = state

    # Bind the unbound method onto the shim.
    _Shim._apply_preview_refinement = (
        CanvasArea._apply_preview_refinement
    )
    return _Shim(state)


class TestApplyPreviewRefinement:
    def test_returns_none_when_both_disabled(self, qapp):
        state = AppState.instance()
        state.subset_size = 40
        state.subset_step = 16
        state.refine_inner = False
        state.refine_outer = False

        coords, elements = _build_uniform_q4(0, 128, 0, 128, 16)
        f_mask = np.ones((129, 129), dtype=np.float64)

        shim = _make_canvas_stub(state)
        result = shim._apply_preview_refinement(coords, elements, f_mask)
        assert result is None

    def test_inner_refinement_subdivides_boundary_elements(self, qapp):
        """Place a circular hole in the middle of the mask. Inner-boundary
        refinement should add nodes near the hole edge so element count
        increases vs. the uniform mesh."""
        state = AppState.instance()
        state.subset_size = 40
        state.subset_step = 16
        state.refine_inner = True
        state.refine_outer = False
        state.refinement_level = 2  # min_size = 16/4 = 4

        coords, elements = _build_uniform_q4(0, 128, 0, 128, 16)
        n_elem_before = elements.shape[0]
        n_nodes_before = coords.shape[0]

        # Mask with a hole at center
        h = w = 129
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        radius = 25
        f_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 > radius**2).astype(np.float64)

        shim = _make_canvas_stub(state)
        result = shim._apply_preview_refinement(coords, elements, f_mask)
        assert result is not None
        new_coords, new_elements = result

        assert new_coords.shape[0] > n_nodes_before, \
            "refinement should add nodes near the hole boundary"
        assert new_elements.shape[0] > 0
        # Refined elements are still Q8 (8 columns)
        assert new_elements.shape[1] == 8

    def test_outer_refinement_subdivides_roi_edge_elements(self, qapp):
        """A solid rectangular ROI: outer-boundary refinement should add
        nodes near the rectangle edges."""
        state = AppState.instance()
        state.subset_size = 40
        state.subset_step = 16
        state.refine_inner = False
        state.refine_outer = True
        state.refinement_level = 2

        coords, elements = _build_uniform_q4(0, 128, 0, 128, 16)
        n_nodes_before = coords.shape[0]

        # Solid rectangular mask (no holes)
        h = w = 129
        f_mask = np.zeros((h, w), dtype=np.float64)
        f_mask[20:108, 20:108] = 1.0

        shim = _make_canvas_stub(state)
        result = shim._apply_preview_refinement(coords, elements, f_mask)
        assert result is not None
        new_coords, _ = result
        assert new_coords.shape[0] > n_nodes_before

    def test_failure_returns_none_silently(self, qapp, monkeypatch):
        """If refine_mesh raises, the helper must swallow it and return
        None so the GUI keeps showing the uniform mesh."""
        from al_dic.mesh import refinement as ref_module

        state = AppState.instance()
        state.subset_size = 40
        state.subset_step = 16
        state.refine_inner = True

        def boom(*_args, **_kwargs):
            raise RuntimeError("simulated failure inside refine_mesh")

        monkeypatch.setattr(ref_module, "refine_mesh", boom)

        coords, elements = _build_uniform_q4(0, 64, 0, 64, 16)
        f_mask = np.ones((65, 65), dtype=np.float64)

        shim = _make_canvas_stub(state)
        result = shim._apply_preview_refinement(coords, elements, f_mask)
        assert result is None
