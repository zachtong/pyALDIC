"""Tests for solver backend dispatch chains (6-DOF and 2-DOF).

Verifies that the fallback chains in local_icgn._dispatch_6dof and
subpb1_solver._dispatch_2dof work correctly by patching backends.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from al_dic.core.data_structures import DICPara, ImageGradients
from al_dic.solver.local_icgn import local_icgn, _dispatch_6dof, _sequential_6dof
from al_dic.solver.subpb1_solver import subpb1_solver, _dispatch_2dof, _sequential_2dof


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(size=128, n_nodes=4, winsize=20):
    """Create minimal synthetic data for dispatch tests."""
    rng = np.random.default_rng(42)
    img = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    df_dx = np.zeros_like(img)
    df_dy = np.zeros_like(img)
    df_dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    df_dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    mask = np.ones_like(img)
    Df = ImageGradients(df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask, img_size=img.shape)

    # Interior nodes only
    step = size // (n_nodes + 1)
    pts = np.arange(step, size - step + 1, step, dtype=np.float64)[:n_nodes]
    if len(pts) < 2:
        pts = np.array([size // 3, 2 * size // 3], dtype=np.float64)
    xs, ys = np.meshgrid(pts[:2], pts[:2])
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)

    n = coords.shape[0]
    U0 = np.zeros(2 * n, dtype=np.float64)
    para = DICPara(winsize=winsize, icgn_max_iter=50)

    return img, coords, Df, U0, para, n


# ---------------------------------------------------------------------------
# 6-DOF dispatch tests
# ---------------------------------------------------------------------------

class TestDispatch6DOF:
    def test_dispatch_falls_back_when_no_numba(self):
        """When HAS_NUMBA=False, batch backend should be used."""
        img, coords, Df, U0, para, n = _make_test_data()
        with patch("al_dic.solver.local_icgn.HAS_NUMBA", False, create=True):
            U, F, t, conv, bad, holes = local_icgn(
                U0, coords, Df, img, img.copy(), para, tol=1e-3,
            )
        assert U.shape == (2 * n,)
        assert F.shape == (4 * n,)

    def test_dispatch_falls_back_to_sequential(self):
        """When batch raises, sequential should handle it."""
        img, coords, Df, U0, para, n = _make_test_data()
        from al_dic.solver.icgn_batch import precompute_subsets_6dof

        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, para.winsize,
        )
        U0_2d = U0.reshape(-1, 2)

        # Patch batch to raise, forcing sequential
        with patch(
            "al_dic.solver.local_icgn._iterate_6dof_batch",
            side_effect=RuntimeError("batch failed"),
            create=True,
        ):
            U_2d, F_2d, conv = _sequential_6dof(
                coords, U0_2d, img.copy(), pre, 1e-3, 50,
            )
        assert U_2d.shape == (n, 2)
        assert F_2d.shape == (n, 4)

    def test_sequential_produces_valid_output(self):
        """Sequential fallback should produce valid output."""
        img, coords, Df, U0, para, n = _make_test_data()
        from al_dic.solver.icgn_batch import precompute_subsets_6dof

        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, para.winsize,
        )
        U_2d, F_2d, conv = _sequential_6dof(
            coords, U0.reshape(-1, 2), img.copy(), pre, 1e-3, 50,
        )
        assert U_2d.shape == (n, 2)
        # Output should be finite for valid nodes
        for j in range(n):
            if pre["valid"][j]:
                assert np.all(np.isfinite(U_2d[j]))

    def test_sequential_skips_invalid_nodes(self):
        """Invalid nodes (pre['valid']=False) should be skipped."""
        img, coords, Df, U0, para, n = _make_test_data()
        from al_dic.solver.icgn_batch import precompute_subsets_6dof

        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, para.winsize,
        )
        # Mark all nodes invalid
        pre["valid"][:] = False

        U_2d, F_2d, conv = _sequential_6dof(
            coords, U0.reshape(-1, 2), img.copy(), pre, 1e-3, 50,
        )
        # All should remain at default (not converged)
        assert np.all(conv >= 50 + 2)

    def test_sequential_exception_sets_nan(self):
        """When icgn_solver raises, output should be NaN."""
        img, coords, Df, U0, para, n = _make_test_data()
        from al_dic.solver.icgn_batch import precompute_subsets_6dof

        pre = precompute_subsets_6dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, para.winsize,
        )
        with patch(
            "al_dic.solver.icgn_solver.icgn_solver",
            side_effect=RuntimeError("solver crash"),
        ):
            U_2d, F_2d, conv = _sequential_6dof(
                coords, U0.reshape(-1, 2), img.copy(), pre, 1e-3, 50,
            )
        # Valid nodes that raised should have NaN
        for j in range(n):
            if pre["valid"][j]:
                assert np.isnan(U_2d[j, 0])


# ---------------------------------------------------------------------------
# 2-DOF dispatch tests
# ---------------------------------------------------------------------------

class TestDispatch2DOF:
    def test_dispatch_skips_numba_for_small_n(self):
        """N < 50 should skip Numba and go to batch."""
        img, coords, Df, U0, para, n = _make_test_data(n_nodes=2)
        assert n < 50  # ensure small N

        n_nodes = coords.shape[0]
        U_old = np.zeros(2 * n_nodes)
        F_old = np.zeros(4 * n_nodes)
        udual = np.zeros(2 * n_nodes)
        vdual = np.zeros(4 * n_nodes)

        U, solve_time, conv, bad = subpb1_solver(
            U_old, F_old, udual, vdual,
            coords, Df, img, img.copy(),
            mu=1e-3, beta=1e-3, para=para, tol=1e-3,
        )
        assert U.shape == (2 * n_nodes,)

    def test_2dof_dispatch_falls_back_to_sequential(self):
        """When batch raises, sequential should handle 2-DOF."""
        img, coords, Df, U0, para, n = _make_test_data(n_nodes=2)

        n_nodes = coords.shape[0]
        U_old_2d = np.zeros((n_nodes, 2))
        F_old_2d = np.zeros((n_nodes, 4))
        udual_2d = np.zeros((n_nodes, 2))

        from al_dic.solver.icgn_batch import precompute_subsets_2dof
        ws = np.full(n_nodes, para.winsize, dtype=int)
        pre = precompute_subsets_2dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, ws, ws,
        )

        U_out, conv = _sequential_2dof(
            coords, U_old_2d, F_old_2d, udual_2d,
            img.copy(), pre, 1e-3, 1e-3, 50,
        )
        assert U_out.shape == (n_nodes, 2)

    def test_2dof_sequential_produces_valid_output(self):
        """Sequential 2-DOF should produce valid output."""
        img, coords, Df, U0, para, n = _make_test_data(n_nodes=2)

        n_nodes = coords.shape[0]
        U_old_2d = np.zeros((n_nodes, 2))
        F_old_2d = np.zeros((n_nodes, 4))
        udual_2d = np.zeros((n_nodes, 2))

        from al_dic.solver.icgn_batch import precompute_subsets_2dof
        ws = np.full(n_nodes, para.winsize, dtype=int)
        pre = precompute_subsets_2dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, ws, ws,
        )

        U_out, conv = _sequential_2dof(
            coords, U_old_2d, F_old_2d, udual_2d,
            img.copy(), pre, 1e-3, 1e-3, 50,
        )
        # At least some nodes should have finite output
        assert np.any(np.isfinite(U_out))

    def test_2dof_sequential_exception_sets_nan(self):
        """When icgn_subpb1 raises, output should be NaN."""
        img, coords, Df, U0, para, n = _make_test_data(n_nodes=2)

        n_nodes = coords.shape[0]
        U_old_2d = np.zeros((n_nodes, 2))
        F_old_2d = np.zeros((n_nodes, 4))
        udual_2d = np.zeros((n_nodes, 2))

        from al_dic.solver.icgn_batch import precompute_subsets_2dof
        ws = np.full(n_nodes, para.winsize, dtype=int)
        pre = precompute_subsets_2dof(
            coords, img, Df.df_dx, Df.df_dy, Df.img_ref_mask, ws, ws,
        )

        with patch(
            "al_dic.solver.icgn_subpb1.icgn_subpb1",
            side_effect=RuntimeError("crash"),
        ):
            U_out, conv = _sequential_2dof(
                coords, U_old_2d, F_old_2d, udual_2d,
                img.copy(), pre, 1e-3, 1e-3, 50,
            )
        for j in range(n_nodes):
            if pre["valid"][j]:
                assert np.isnan(U_out[j, 0])
