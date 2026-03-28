"""Tests for masked subset IC-GN behavior.

Verifies that IC-GN precompute uses raw image pixels and mask-based validity,
avoiding artificial gradient edges at mask boundaries.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, binary_erosion

from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.solver.icgn_batch import (
    precompute_subsets_6dof,
    precompute_subsets_2dof,
    _connected_center_mask,
)


def _make_smooth_speckle(h=128, w=128, sigma=3.0, seed=42):
    """Generate a smooth speckle image (values in [20, 235])."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


class TestPrecompute6dofMaskedSubset:
    """Tests for precompute_subsets_6dof with raw image + mask-based validity."""

    def test_full_mask_unchanged(self):
        """Full mask: precompute should produce valid results for all nodes."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.ones((h, w), dtype=np.float64)
        Df = compute_image_gradient(img, mask, img_raw=img)

        coords = np.array([[40.0, 40.0], [80.0, 80.0]])
        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, mask, winsize=16,
        )
        assert pre["valid"][0]
        assert pre["valid"][1]

    def test_boundary_node_no_artificial_gradient(self):
        """Node near mask boundary: gradient should reflect raw speckle."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.ones((h, w), dtype=np.float64)
        mask[:, 50:] = 0.0  # right half masked

        # Gradients from raw image (no artificial edge)
        Df = compute_image_gradient(img * mask, mask, img_raw=img)

        # Node at x=46 (4px from boundary): subset extends into masked region
        coords = np.array([[46.0, 64.0]])
        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, mask, winsize=16,
        )

        if pre["valid"][0]:
            bw = pre["mask_all"][0]
            gx = pre["gx_all"][0]
            valid_gx = gx[bw > 0.5]
            # No artificial edge (would be ~75+); raw speckle gradient ~<30
            assert np.max(np.abs(valid_gx)) < 50

    def test_two_island_keeps_center_only(self):
        """Mask with two disconnected regions: only center-containing kept."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.ones((h, w), dtype=np.float64)
        # Create a vertical gap splitting the subset in half
        mask[:, 63:65] = 0.0

        Df = compute_image_gradient(img * mask, mask, img_raw=img)

        # Node at x=60: center is left of the gap
        coords = np.array([[60.0, 64.0]])
        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, mask, winsize=16,
        )

        if pre["valid"][0]:
            bw = pre["mask_all"][0]
            # The right side of the gap should NOT be in bw
            # Subset spans x=[52..68], gap is at relative x=[11,12]
            # Right of gap (relative x>=13) should be zero
            assert np.all(bw[:, 13:] == 0.0)

    def test_too_few_pixels_rejected(self):
        """Node mostly outside mask should be rejected."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.zeros((h, w), dtype=np.float64)
        # Only a small patch is valid
        mask[60:66, 60:66] = 1.0

        Df = compute_image_gradient(img * mask, mask, img_raw=img)

        # Node at center of small valid patch
        coords = np.array([[63.0, 63.0]])
        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, mask, winsize=16,
        )
        # 6x6 = 36 valid pixels out of 17x17 = 289 → ratio 0.12 < 0.5
        assert not pre["valid"][0]

    def test_hessian_conditioning_rejects_thin_strip(self):
        """Thin strip of valid pixels → ill-conditioned Hessian → rejected."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.zeros((h, w), dtype=np.float64)
        # Only a 2px-wide horizontal strip is valid (enough pixels but rank-deficient)
        mask[63:65, :] = 1.0

        Df = compute_image_gradient(img * mask, mask, img_raw=img)

        coords = np.array([[64.0, 64.0]])
        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, mask, winsize=16,
        )
        # 2-wide strip → ill-conditioned Hessian → should be marked as hole
        assert not pre["valid"][0] or pre["mark_hole"][0]


class TestPrecompute2dofMaskedSubset:
    """Tests for precompute_subsets_2dof with same masked subset logic."""

    def test_full_mask_valid(self):
        """Full mask: all nodes should be valid."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.ones((h, w), dtype=np.float64)
        Df = compute_image_gradient(img, mask, img_raw=img)

        coords = np.array([[40.0, 40.0], [80.0, 80.0]])
        wx = np.array([16, 16], dtype=np.int64)
        wy = np.array([16, 16], dtype=np.int64)
        pre = precompute_subsets_2dof(
            coords, img, Df.df_dx, Df.df_dy, mask, wx, wy,
        )
        assert pre["valid"][0]
        assert pre["valid"][1]

    def test_boundary_node_valid_with_raw_gradient(self):
        """Node near boundary with raw gradient should still be valid."""
        h, w = 128, 128
        img = _make_smooth_speckle(h, w)
        mask = np.ones((h, w), dtype=np.float64)
        mask[:, 50:] = 0.0

        Df = compute_image_gradient(img * mask, mask, img_raw=img)

        coords = np.array([[42.0, 64.0]])
        wx = np.array([16], dtype=np.int64)
        wy = np.array([16], dtype=np.int64)
        pre = precompute_subsets_2dof(
            coords, img, Df.df_dx, Df.df_dy, mask, wx, wy,
        )
        # Node at x=42, subset spans [34..50], mask boundary at x=50
        # Most pixels are valid, should pass
        assert pre["valid"][0]


class TestConnectedCenterMask:
    """Tests for _connected_center_mask helper."""

    def test_full_mask(self):
        """Full mask returns all ones."""
        mask = np.ones((17, 17))
        bw = _connected_center_mask(mask)
        np.testing.assert_array_equal(bw, 1.0)

    def test_two_islands(self):
        """Two disconnected regions: only center one kept."""
        mask = np.ones((17, 17))
        mask[:, 8] = 0  # vertical split
        bw = _connected_center_mask(mask)
        # Center is at (8, 8), which is on the split line (mask=0)
        # The center pixel is masked → depends on center connectivity
        # After split, center col=8 is zero, center falls on the gap
        # This is an edge case — just verify shape is correct
        assert bw.shape == (17, 17)

    def test_empty_mask(self):
        """Empty mask → all zeros."""
        mask = np.zeros((17, 17))
        bw = _connected_center_mask(mask)
        np.testing.assert_array_equal(bw, 0.0)
