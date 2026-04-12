"""Tests for warp_mask: binary mask warping from reference to deformed frame."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label as ndimage_label

from al_dic.utils.warp_mask import warp_mask


def _topology(mask):
    """Return (n_domains, n_holes) for a binary mask."""
    h, w = mask.shape
    _, n_domains = ndimage_label(mask > 0.5)
    labeled_z, n_zero = ndimage_label(mask < 0.5)
    border = set()
    border.update(labeled_z[0, :].tolist())
    border.update(labeled_z[h - 1, :].tolist())
    border.update(labeled_z[:, 0].tolist())
    border.update(labeled_z[:, w - 1].tolist())
    border.discard(0)
    return n_domains, n_zero - len(border)


class TestWarpMaskBasic:
    """Basic correctness tests with known displacements."""

    def test_zero_displacement_identity(self):
        """Zero displacement should return identical mask."""
        mask = np.zeros((64, 64), dtype=np.float64)
        mask[16:48, 16:48] = 1.0
        u = np.zeros_like(mask)
        v = np.zeros_like(mask)

        result = warp_mask(mask, u, v)

        np.testing.assert_array_equal(result, mask)

    def test_uniform_translation_x(self):
        """Uniform x-translation should shift mask right."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=np.float64)
        mask[16:48, 10:30] = 1.0
        shift = 8.0
        u = np.full((h, w), shift, dtype=np.float64)
        v = np.zeros((h, w), dtype=np.float64)

        result = warp_mask(mask, u, v)

        expected = np.zeros((h, w), dtype=np.float64)
        expected[16:48, 18:38] = 1.0
        np.testing.assert_array_equal(result, expected)

    def test_uniform_translation_y(self):
        """Uniform y-translation should shift mask down."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=np.float64)
        mask[10:30, 16:48] = 1.0
        shift = 5.0
        u = np.zeros((h, w), dtype=np.float64)
        v = np.full((h, w), shift, dtype=np.float64)

        result = warp_mask(mask, u, v)

        expected = np.zeros((h, w), dtype=np.float64)
        expected[15:35, 16:48] = 1.0
        np.testing.assert_array_equal(result, expected)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        mask = np.ones((64, 64), dtype=np.float64)
        u = np.zeros((32, 32), dtype=np.float64)
        v = np.zeros((64, 64), dtype=np.float64)

        with pytest.raises(ValueError, match="Shape mismatch"):
            warp_mask(mask, u, v)

    def test_output_is_binary(self):
        """Output should always be exactly 0 or 1."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=np.float64)
        mask[20:44, 20:44] = 1.0
        rng = np.random.default_rng(42)
        u = rng.uniform(-2, 2, (h, w))
        v = rng.uniform(-2, 2, (h, w))

        result = warp_mask(mask, u, v)

        assert set(np.unique(result)).issubset({0.0, 1.0})


class TestWarpMaskWithHoles:
    """Tests with masks containing internal holes."""

    def test_circle_with_hole_translation(self):
        """Circle mask with hole under translation preserves topology."""
        h, w = 128, 128
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = 50.0, 50.0

        # Outer circle r=30, inner hole r=10
        outer = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30**2
        inner = ((xx - cx) ** 2 + (yy - cy) ** 2) < 10**2
        mask = (outer & ~inner).astype(np.float64)

        shift_x, shift_y = 10.0, 5.0
        u = np.full((h, w), shift_x, dtype=np.float64)
        v = np.full((h, w), shift_y, dtype=np.float64)

        result = warp_mask(mask, u, v)

        # Check that the warped mask has roughly the same area
        area_orig = mask.sum()
        area_warped = result.sum()
        assert abs(area_warped - area_orig) / area_orig < 0.05

        # Check center of mass shifted correctly
        orig_cx = np.mean(np.where(mask > 0.5)[1])
        orig_cy = np.mean(np.where(mask > 0.5)[0])
        warp_cx = np.mean(np.where(result > 0.5)[1])
        warp_cy = np.mean(np.where(result > 0.5)[0])
        assert abs(warp_cx - orig_cx - shift_x) < 1.5
        assert abs(warp_cy - orig_cy - shift_y) < 1.5

        # Hole should still exist (center of warped region should be 0)
        new_cx = int(round(cx + shift_x))
        new_cy = int(round(cy + shift_y))
        assert result[new_cy, new_cx] == 0.0


class TestWarpMaskLargeDeformation:
    """Tests with non-uniform, larger displacement fields."""

    def test_stretch_expands_area(self):
        """Uniform stretch should expand the mask area."""
        h, w = 128, 128
        mask = np.zeros((h, w), dtype=np.float64)
        mask[44:84, 44:84] = 1.0  # 40x40 square centered at (64, 64)

        # 10% stretch from center
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = 64.0, 64.0
        strain = 0.10
        u = strain * (xx - cx)
        v = strain * (yy - cy)

        result = warp_mask(mask, u, v)

        # Warped area should be ~(1.1)^2 = 1.21x original
        area_ratio = result.sum() / mask.sum()
        assert 1.15 < area_ratio < 1.30

    def test_convergence_with_iterations(self):
        """More iterations should improve accuracy for moderate deformation."""
        h, w = 128, 128
        mask = np.zeros((h, w), dtype=np.float64)
        mask[30:98, 30:98] = 1.0

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = 64.0, 64.0
        strain = 0.20
        u = strain * (xx - cx)
        v = strain * (yy - cy)

        result_1 = warp_mask(mask, u, v, n_iter=1)
        result_5 = warp_mask(mask, u, v, n_iter=5)
        result_10 = warp_mask(mask, u, v, n_iter=10)

        # Results should converge (5 and 10 iterations nearly identical)
        diff_1_5 = np.abs(result_1 - result_5).sum()
        diff_5_10 = np.abs(result_5 - result_10).sum()
        assert diff_5_10 <= diff_1_5


class TestTopologyPreservation:
    """Verify connected-domain and hole counts are preserved after warping."""

    def test_square_translation_topology(self):
        """Single domain, no holes — topology preserved under translation."""
        h, w = 128, 128
        mask = np.zeros((h, w), dtype=np.float64)
        mask[20:100, 20:100] = 1.0
        u = np.full((h, w), 10.0, dtype=np.float64)
        v = np.full((h, w), 5.0, dtype=np.float64)

        result = warp_mask(mask, u, v)
        assert _topology(result) == (1, 0)

    def test_annular_stretch_topology(self):
        """1 domain, 1 hole — topology preserved under 15% stretch."""
        h, w = 256, 256
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = 128.0, 128.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = ((r < 90) & (r > 25)).astype(np.float64)

        strain = 0.15
        u = strain * (xx - cx)
        v = strain * (yy - cy)

        result = warp_mask(mask, u, v)
        assert _topology(result) == (1, 1)

    def test_multi_hole_quadratic_topology(self):
        """1 domain, 3 holes — topology preserved under quadratic deformation."""
        h, w = 256, 256
        mask = np.zeros((h, w), dtype=np.float64)
        mask[32:224, 32:224] = 1.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

        # 3 circular holes
        for hx, hy, hr in [(77, 90, 20), (154, 128, 26), (102, 179, 15)]:
            mask[np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2) < hr] = 0.0

        assert _topology(mask) == (1, 3)

        # Quadratic deformation (max ~35 px, ~22% peak strain)
        cx, cy = 128.0, 128.0
        dx = (xx - cx) / cx
        dy = (yy - cy) / cy
        amp = 10.0
        u = amp * (1.5 * dx**2 + 0.8 * dy**2 + 0.5 * dx * dy)
        v = amp * (0.6 * dx**2 + 1.2 * dy**2 - 0.4 * dx * dy)

        result = warp_mask(mask, u, v, n_iter=8)
        d, holes = _topology(result)
        assert d == 1, f"Expected 1 domain, got {d}"
        assert holes == 3, f"Expected 3 holes, got {holes}"

    def test_multi_domain_shear_topology(self):
        """3 separate domains, 1 hole — topology preserved under shear."""
        h, w = 256, 256
        mask = np.zeros((h, w), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

        # Domain 1: rectangle with hole
        mask[40:210, 20:100] = 1.0
        mask[np.sqrt((xx - 60) ** 2 + (yy - 128) ** 2) < 15] = 0.0
        # Domain 2: circle
        mask[np.sqrt((xx - 180) ** 2 + (yy - 100) ** 2) < 40] = 1.0
        # Domain 3: small square
        mask[180:220, 160:200] = 1.0

        assert _topology(mask) == (3, 1)

        # Moderate shear + stretch
        cx, cy = 128.0, 128.0
        u = 0.08 * (yy - cy) + 0.05 * (xx - cx)
        v = 0.04 * (yy - cy)

        result = warp_mask(mask, u, v)
        d, holes = _topology(result)
        assert d == 3, f"Expected 3 domains, got {d}"
        assert holes == 1, f"Expected 1 hole, got {holes}"

    def test_cleanup_disabled(self):
        """min_fragment_ratio=0 disables cleanup, may have fragments."""
        h, w = 128, 128
        mask = np.zeros((h, w), dtype=np.float64)
        mask[16:112, 16:112] = 1.0

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        u = 0.4 * (xx - 64.0)
        v = 0.4 * (yy - 64.0)

        result_clean = warp_mask(mask, u, v, n_iter=8, min_fragment_ratio=0.01)
        result_raw = warp_mask(mask, u, v, n_iter=8, min_fragment_ratio=0)

        d_clean, _ = _topology(result_clean)
        d_raw, _ = _topology(result_raw)
        # Cleaned should have fewer or equal domains
        assert d_clean <= d_raw


# ---------------------------------------------------------------------------
# Downsample / upsample and edge cases
# ---------------------------------------------------------------------------

class TestWarpMaskExtended:
    """Extended edge case tests for warp_mask internals."""

    def test_downsample_upsample_roundtrip(self):
        """Downsample then upsample should approximately preserve the mask."""
        from al_dic.utils.warp_mask import _downsample, _upsample

        h, w = 256, 256
        mask = np.zeros((h, w), dtype=np.float64)
        mask[40:216, 40:216] = 1.0
        u = np.full((h, w), 3.0, dtype=np.float64)
        v = np.full((h, w), 2.0, dtype=np.float64)

        scale = 0.5
        mask_ds, u_ds, v_ds = _downsample(mask, u, v, scale)
        mask_up = _upsample(mask_ds, h, w)

        # Area should be approximately preserved
        area_orig = mask.sum()
        area_up = mask_up.sum()
        assert abs(area_up - area_orig) / area_orig < 0.1

    def test_downsample_preserves_structure(self):
        """Downsampled mask should still have the same overall shape."""
        from al_dic.utils.warp_mask import _downsample

        h, w = 512, 512
        mask = np.zeros((h, w), dtype=np.float64)
        mask[100:412, 100:412] = 1.0
        u = np.full((h, w), 1.0, dtype=np.float64)
        v = np.full((h, w), 1.0, dtype=np.float64)

        mask_ds, u_ds, v_ds = _downsample(mask, u, v, scale=0.25)
        assert mask_ds.shape[0] < h
        assert mask_ds.shape[1] < w
        assert mask_ds.sum() > 0

    def test_upsample_pad_when_undersized(self):
        """Upsample should pad with zeros if repeat undershoots target."""
        from al_dic.utils.warp_mask import _upsample

        small = np.ones((10, 10), dtype=np.float64)
        result = _upsample(small, 256, 256)
        assert result.shape == (256, 256)

    def test_total_area_zero_returns_unchanged(self):
        """All-zero mask should pass through without error."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=np.float64)
        u = np.zeros((h, w), dtype=np.float64)
        v = np.zeros((h, w), dtype=np.float64)

        result = warp_mask(mask, u, v)
        assert result.sum() == 0.0

    def test_large_image_triggers_downsample(self):
        """Images > max_warp_pixels should trigger downsample path."""
        h, w = 1024, 1024
        mask = np.zeros((h, w), dtype=np.float64)
        mask[200:824, 200:824] = 1.0
        u = np.full((h, w), 5.0, dtype=np.float64)
        v = np.full((h, w), 3.0, dtype=np.float64)

        # max_warp_pixels = 512*512, so 1024*1024 triggers downsample
        result = warp_mask(mask, u, v, max_warp_pixels=512 * 512)
        assert result.shape == (h, w)
        assert result.sum() > 0

    def test_max_warp_pixels_zero_disables_downsample(self):
        """max_warp_pixels=0 should disable downsampling."""
        h, w = 128, 128
        mask = np.zeros((h, w), dtype=np.float64)
        mask[20:108, 20:108] = 1.0
        u = np.full((h, w), 3.0, dtype=np.float64)
        v = np.full((h, w), 2.0, dtype=np.float64)

        result = warp_mask(mask, u, v, max_warp_pixels=0)
        assert result.shape == (h, w)
        assert result.sum() > 0
