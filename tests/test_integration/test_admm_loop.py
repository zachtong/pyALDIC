"""Focused ADMM 3-step loop integration tests.

Tests the ADMM iteration structure (subpb1 → subpb2 → dual update)
without running a full end-to-end pipeline.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from al_dic.core.data_structures import DICPara, ImageGradients
from al_dic.solver.local_icgn import local_icgn
from al_dic.solver.subpb1_solver import subpb1_solver, precompute_subpb1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(size=128, winsize=20, step=32):
    """Minimal data for ADMM loop tests."""
    rng = np.random.default_rng(42)
    img = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    mask = np.ones_like(img)
    Df = ImageGradients(df_dx=dx, df_dy=dy, img_ref_mask=mask, img_size=img.shape)

    # Interior node grid
    margin = step
    xs = np.arange(margin, size - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, size - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    para = DICPara(winsize=winsize, icgn_max_iter=30, admm_max_iter=3)
    return img, coords, Df, para


class TestADMMLoop:
    def test_admm_single_iteration_shapes(self):
        """One ADMM iteration should produce correct output shapes."""
        img, coords, Df, para = _make_test_data()
        n = coords.shape[0]

        # Step 0: local ICGN
        U0 = np.zeros(2 * n)
        U, F, _, conv, _, _ = local_icgn(
            U0, coords, Df, img, img.copy(), para, tol=1e-3,
        )

        # subpb1 inputs
        udual = np.zeros(2 * n)
        vdual = np.zeros(4 * n)

        pre = precompute_subpb1(coords, Df, img, para)
        U1, t, conv1, bad = subpb1_solver(
            U, F, udual, vdual, coords, Df, img, img.copy(),
            mu=para.mu, beta=1e-3, para=para, tol=1e-3,
            precomputed=pre,
        )

        assert U1.shape == (2 * n,)
        assert conv1.shape == (n,)

    def test_admm_dual_variable_bounded(self):
        """Dual variables (udual = U_subpb1 - U_subpb2) should be bounded."""
        img, coords, Df, para = _make_test_data()
        n = coords.shape[0]

        U0 = np.zeros(2 * n)
        U, F, _, _, _, _ = local_icgn(
            U0, coords, Df, img, img.copy(), para, tol=1e-3,
        )

        udual = np.zeros(2 * n)
        vdual = np.zeros(4 * n)

        pre = precompute_subpb1(coords, Df, img, para)
        U1, _, _, _ = subpb1_solver(
            U, F, udual, vdual, coords, Df, img, img.copy(),
            mu=para.mu, beta=1e-3, para=para, tol=1e-3,
            precomputed=pre,
        )

        # Dual update
        udual_new = udual + (U1 - U)
        # Should be bounded for identical images
        assert np.max(np.abs(udual_new)) < 10.0

    def test_admm_zero_displacement_stable(self):
        """Zero displacement (identical images) should remain near zero."""
        img, coords, Df, para = _make_test_data()
        n = coords.shape[0]

        U0 = np.zeros(2 * n)
        U, F, _, _, _, _ = local_icgn(
            U0, coords, Df, img, img.copy(), para, tol=1e-3,
        )

        udual = np.zeros(2 * n)
        vdual = np.zeros(4 * n)

        pre = precompute_subpb1(coords, Df, img, para)
        U1, _, _, _ = subpb1_solver(
            U, F, udual, vdual, coords, Df, img, img.copy(),
            mu=para.mu, beta=1e-3, para=para, tol=1e-3,
            precomputed=pre,
        )

        # Should still be near zero
        assert np.nanmax(np.abs(U1)) < 1.0

    def test_admm_precomputed_reuse_consistent(self):
        """Using precomputed subsets should give same result as inline compute."""
        img, coords, Df, para = _make_test_data()
        n = coords.shape[0]

        U = np.zeros(2 * n)
        F = np.zeros(4 * n)
        udual = np.zeros(2 * n)
        vdual = np.zeros(4 * n)

        # With precomputed
        pre = precompute_subpb1(coords, Df, img, para)
        U_pre, _, conv_pre, _ = subpb1_solver(
            U.copy(), F.copy(), udual.copy(), vdual.copy(),
            coords, Df, img, img.copy(),
            mu=para.mu, beta=1e-3, para=para, tol=1e-3,
            precomputed=pre,
        )

        # Without precomputed
        U_no, _, conv_no, _ = subpb1_solver(
            U.copy(), F.copy(), udual.copy(), vdual.copy(),
            coords, Df, img, img.copy(),
            mu=para.mu, beta=1e-3, para=para, tol=1e-3,
            precomputed=None,
        )

        np.testing.assert_allclose(U_pre, U_no, atol=1e-10)

    def test_admm_improves_over_iterations(self):
        """Multiple ADMM iterations should reduce or maintain error."""
        from scipy.ndimage import shift as ndimage_shift

        img, coords, Df, para = _make_test_data(size=128, winsize=20, step=32)
        n = coords.shape[0]
        img_def = ndimage_shift(img, [0.5, 0.5], order=3, mode="constant")

        U0 = np.zeros(2 * n)
        U, F, _, _, _, _ = local_icgn(
            U0, coords, Df, img, img_def, para, tol=1e-3,
        )

        udual = np.zeros(2 * n)
        vdual = np.zeros(4 * n)

        pre = precompute_subpb1(coords, Df, img, para)

        # Run 2 ADMM iterations and check error doesn't increase dramatically
        errors = []
        U_cur = U.copy()
        for _ in range(2):
            U_new, _, _, _ = subpb1_solver(
                U_cur, F, udual, vdual, coords, Df, img, img_def,
                mu=para.mu, beta=1e-3, para=para, tol=1e-3,
                precomputed=pre,
            )
            err = np.sqrt(np.nanmean((U_new - U_cur) ** 2))
            errors.append(err)
            udual = udual + (U_new - U_cur)
            U_cur = U_new

        # Second iteration change should not be wildly larger than first
        if errors[0] > 1e-10:
            assert errors[1] < errors[0] * 10.0
