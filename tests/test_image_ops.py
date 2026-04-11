"""Tests for image normalization and gradient computation."""

import numpy as np
import pytest

from al_dic.core.data_structures import GridxyROIRange
from al_dic.io.image_ops import normalize_images, compute_image_gradient


class TestNormalizeImages:
    def test_single_image(self):
        img = np.random.default_rng(0).random((64, 64)) * 255
        roi = GridxyROIRange(gridx=(0, 63), gridy=(0, 63))
        normed, out_roi = normalize_images([img], roi)

        assert len(normed) == 1
        assert normed[0].shape == (64, 64)

        # Mean of the ROI region should be ~0
        roi_patch = normed[0][0:64, 0:64]
        assert abs(np.mean(roi_patch)) < 1e-10

    def test_multiple_images(self):
        rng = np.random.default_rng(1)
        imgs = [rng.random((64, 64)) * 255 for _ in range(3)]
        roi = GridxyROIRange(gridx=(10, 50), gridy=(10, 50))
        normed, _ = normalize_images(imgs, roi)
        assert len(normed) == 3

    def test_roi_clamping(self):
        img = np.ones((32, 32))
        roi = GridxyROIRange(gridx=(-5, 100), gridy=(-5, 100))
        _, clamped = normalize_images([img], roi)
        assert clamped.gridx == (0, 31)
        assert clamped.gridy == (0, 31)

    def test_empty_list(self):
        normed, roi = normalize_images([], GridxyROIRange())
        assert normed == []


class TestComputeImageGradient:
    def test_constant_image(self):
        img = np.ones((64, 64)) * 128.0
        grad = compute_image_gradient(img)
        # Gradient of constant should be zero
        np.testing.assert_allclose(grad.df_dx, 0, atol=1e-12)
        np.testing.assert_allclose(grad.df_dy, 0, atol=1e-12)

    def test_linear_x_ramp(self):
        """Image linearly increasing in x direction: gradient should be constant."""
        h, w = 64, 64
        img = np.tile(np.arange(w, dtype=np.float64), (h, 1))
        grad = compute_image_gradient(img)

        # Interior of the gradient should be approximately 1.0
        # (edges may have boundary effects)
        interior = grad.df_dx[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 1.0, atol=1e-6)

        # y-derivative should be ~0
        np.testing.assert_allclose(grad.df_dy[5:-5, 5:-5], 0, atol=1e-10)

    def test_output_shape(self):
        h, w = 100, 80
        img = np.random.default_rng(42).random((h, w))
        grad = compute_image_gradient(img)

        # Full-size output with zeros in the 3-pixel border
        assert grad.df_dx.shape == (h, w)
        assert grad.df_dy.shape == (h, w)
        # Border pixels should be zero
        assert np.all(grad.df_dx[:3, :] == 0)
        assert np.all(grad.df_dx[-3:, :] == 0)
        assert np.all(grad.df_dx[:, :3] == 0)
        assert np.all(grad.df_dx[:, -3:] == 0)
        assert grad.img_size == (h, w)

    def test_mask_applied(self):
        h, w = 64, 64
        img = np.random.default_rng(0).random((h, w)) * 100
        mask = np.zeros((h, w))
        mask[10:50, 10:50] = 1.0
        grad = compute_image_gradient(img, img_ref_mask=mask)

        # Outside mask region, gradient should be zero
        assert grad.df_dx[0, 0] == 0.0
        assert grad.df_dy[0, 0] == 0.0

    def test_gradient_from_raw_image(self):
        """Gradient from raw image should NOT have artificial edge at mask boundary."""
        from scipy.ndimage import gaussian_filter

        h, w = 64, 64
        rng = np.random.default_rng(42)
        # Smooth speckle-like image (sigma=3 makes features ~6px wide)
        noise = rng.standard_normal((h, w))
        img = gaussian_filter(noise, sigma=3.0, mode="nearest")
        img = 20.0 + 215.0 * (img - img.min()) / (img.max() - img.min())

        # Circular mask
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx - 32) ** 2 + (yy - 32) ** 2 < 20 ** 2).astype(np.float64)

        # Old way: gradient from masked image
        Df_old = compute_image_gradient(img * mask, mask)

        # New way: gradient from raw image
        Df_new = compute_image_gradient(img * mask, mask, img_raw=img)

        # At boundary pixels: old gradient has large artificial edge, new doesn't
        from scipy.ndimage import binary_erosion

        boundary = mask.astype(bool) & ~binary_erosion(mask.astype(bool), iterations=1)
        boundary_idx = np.where(boundary)

        old_grad_mag = np.sqrt(
            Df_old.df_dx[boundary_idx] ** 2 + Df_old.df_dy[boundary_idx] ** 2
        )
        new_grad_mag = np.sqrt(
            Df_new.df_dx[boundary_idx] ** 2 + Df_new.df_dy[boundary_idx] ** 2
        )

        # New gradient at boundary should be MUCH smaller (no artificial edge)
        assert np.median(new_grad_mag) < np.median(old_grad_mag) * 0.5
