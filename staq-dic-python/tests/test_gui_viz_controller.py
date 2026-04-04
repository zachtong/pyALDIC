"""Tests for visualization controller -- colormap and caching."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from staq_dic.gui.controllers.viz_controller import (
    apply_colormap,
    VizController,
)


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication exists for QPixmap operations."""
    app = QApplication.instance() or QApplication([])
    yield app


class TestApplyColormap:
    def test_output_shape(self):
        data = np.random.rand(100, 200).astype(np.float64)
        rgba = apply_colormap(data, vmin=0, vmax=1, cmap="jet")
        assert rgba.shape == (100, 200, 4)
        assert rgba.dtype == np.uint8

    def test_nan_transparent(self):
        data = np.full((10, 10), np.nan)
        rgba = apply_colormap(data, vmin=0, vmax=1)
        assert np.all(rgba[:, :, 3] == 0)  # alpha = 0 for NaN

    def test_range_clamping(self):
        data = np.array([[0.0, 0.5, 1.0, 2.0]])
        rgba = apply_colormap(data, vmin=0, vmax=1)
        # value=2.0 should be clamped to max color, not crash
        assert rgba[0, 3, 3] > 0  # not NaN, has alpha

    def test_equal_vmin_vmax(self):
        data = np.array([[5.0, 5.0]])
        rgba = apply_colormap(data, vmin=5.0, vmax=5.0)
        assert rgba.shape == (1, 2, 4)
        assert rgba.dtype == np.uint8

    def test_valid_pixels_opaque(self):
        data = np.array([[0.0, 0.5, 1.0]])
        rgba = apply_colormap(data, vmin=0, vmax=1)
        # All valid pixels should have full alpha (255)
        assert np.all(rgba[0, :, 3] == 255)

    def test_different_colormaps(self):
        data = np.linspace(0, 1, 50).reshape(5, 10)
        for cmap_name in ("jet", "viridis", "coolwarm"):
            rgba = apply_colormap(data, vmin=0, vmax=1, cmap=cmap_name)
            assert rgba.shape == (5, 10, 4)


class TestVizController:
    def test_cache_hit(self, qapp):
        ctrl = VizController()
        data = np.random.rand(50, 50)
        key = (0, "disp_u")

        ctrl.store_interp_result(key, data, None, None)
        cached = ctrl.get_interp_result(key)
        assert cached is not None
        np.testing.assert_array_equal(cached[0], data)

    def test_cache_miss(self, qapp):
        ctrl = VizController()
        assert ctrl.get_interp_result((99, "disp_v")) is None

    def test_clear_all(self, qapp):
        ctrl = VizController()
        ctrl.store_interp_result((0, "disp_u"), np.ones((5, 5)), None, None)
        ctrl.clear_all()
        assert ctrl.get_interp_result((0, "disp_u")) is None

    def test_clear_pixmap_only(self, qapp):
        ctrl = VizController()
        ctrl.store_interp_result((0, "disp_u"), np.ones((5, 5)), None, None)
        ctrl.clear_pixmap_cache()
        # Tier 1 should survive
        assert ctrl.get_interp_result((0, "disp_u")) is not None

    def test_store_with_grids(self, qapp):
        ctrl = VizController()
        data = np.random.rand(20, 30)
        xg = np.arange(30, dtype=np.float64)
        yg = np.arange(20, dtype=np.float64)
        key = (1, "disp_v")
        ctrl.store_interp_result(key, data, xg, yg)
        cached = ctrl.get_interp_result(key)
        assert cached is not None
        np.testing.assert_array_equal(cached[1], xg)
        np.testing.assert_array_equal(cached[2], yg)

    def test_multiple_keys_independent(self, qapp):
        ctrl = VizController()
        d1 = np.ones((5, 5))
        d2 = np.zeros((5, 5))
        ctrl.store_interp_result((0, "disp_u"), d1, None, None)
        ctrl.store_interp_result((0, "disp_v"), d2, None, None)

        c1 = ctrl.get_interp_result((0, "disp_u"))
        c2 = ctrl.get_interp_result((0, "disp_v"))
        assert c1 is not None
        assert c2 is not None
        assert np.all(c1[0] == 1)
        assert np.all(c2[0] == 0)
