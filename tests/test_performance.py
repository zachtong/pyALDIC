"""Performance regression tests — guard against unexpected slowdowns.

These tests use @pytest.mark.slow and verify wall-clock time stays
within reasonable bounds. They don't test correctness (covered elsewhere).
"""

import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from al_dic.core.data_structures import DICPara, ImageGradients
from al_dic.solver.icgn_batch import precompute_subsets_6dof
from al_dic.solver.local_icgn import local_icgn
from al_dic.utils.warp_mask import warp_mask


def _make_speckle(h, w, seed=42):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.standard_normal((h, w)), sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def _make_gradients(img):
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return dx, dy


def _make_interior_coords(h, w, step):
    margin = step
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


@pytest.mark.slow
class TestPerformance:
    def test_precompute_6dof_performance(self):
        """128x128 image + 64 nodes: precompute < 2s."""
        img = _make_speckle(128, 128)
        dx, dy = _make_gradients(img)
        mask = np.ones_like(img)
        coords = _make_interior_coords(128, 128, step=16)

        t0 = time.perf_counter()
        pre = precompute_subsets_6dof(coords, img, dx, dy, mask, winsize=20)
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, f"Precompute took {elapsed:.2f}s (limit: 2s)"
        assert pre["valid"].any()

    def test_local_icgn_performance(self):
        """Full local_icgn call: < 10s."""
        img = _make_speckle(128, 128)
        dx, dy = _make_gradients(img)
        mask = np.ones_like(img)
        Df = ImageGradients(df_dx=dx, df_dy=dy, img_ref_mask=mask, img_size=img.shape)
        coords = _make_interior_coords(128, 128, step=16)
        n = coords.shape[0]
        U0 = np.zeros(2 * n)
        para = DICPara(winsize=20, icgn_max_iter=20)

        t0 = time.perf_counter()
        U, F, t, conv, bad, holes = local_icgn(
            U0, coords, Df, img, img.copy(), para, tol=1e-3,
        )
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"local_icgn took {elapsed:.2f}s (limit: 10s)"

    def test_warp_mask_performance(self):
        """512x512 warp: < 1s."""
        h, w = 512, 512
        mask = np.zeros((h, w), dtype=np.float64)
        mask[50:462, 50:462] = 1.0
        u = np.full((h, w), 5.0, dtype=np.float64)
        v = np.full((h, w), 3.0, dtype=np.float64)

        t0 = time.perf_counter()
        result = warp_mask(mask, u, v)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"warp_mask took {elapsed:.2f}s (limit: 1s)"

    def test_strain_compute_performance(self):
        """Strain computation for ~49 nodes: < 1s."""
        from al_dic.strain.comp_def_grad import comp_def_grad
        from tests.conftest import make_mesh_for_image

        mesh = make_mesh_for_image(128, 128, step=16)
        n = mesh.coordinates_fem.shape[0]
        U = np.random.default_rng(42).standard_normal(2 * n) * 0.01

        t0 = time.perf_counter()
        F = comp_def_grad(
            U, mesh.coordinates_fem, mesh.elements_fem,
            rad=20.0,
        )
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"Strain compute took {elapsed:.2f}s (limit: 1s)"
        assert F.shape == (4 * n,)
