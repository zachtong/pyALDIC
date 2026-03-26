"""Tests for image normalization and gradient computation."""

import numpy as np
import pytest

from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.io.image_ops import normalize_images, compute_image_gradient


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

        # Cropped by 3 pixels on each side
        assert grad.df_dx.shape == (h - 6, w - 6)
        assert grad.df_dy.shape == (h - 6, w - 6)
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
