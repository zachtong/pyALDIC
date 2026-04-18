"""Integration tests for init_guess_mode='seed_propagation'.

C10a scope: single-ref (accumulative) two-frame sequence. Verifies that
the pipeline wires seed_propagation correctly without relying on FFT
for the initial guess and produces a sane displacement field.
"""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

from al_dic.core.data_structures import DICPara, GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.solver.seed_propagation import Seed, SeedSet


def _speckle(h: int = 192, w: int = 192, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.rand(h, w).astype(np.float64)
    img = gaussian_filter(img, sigma=3.0)
    return (img - img.min()) / (img.max() - img.min() + 1e-10)


def _make_para(
    h: int, w: int,
    seed_set: SeedSet | None,
    init_mode: str = "seed_propagation",
    **overrides,
) -> DICPara:
    roi = GridxyROIRange(gridx=(30, w - 30), gridy=(30, h - 30))
    defaults = dict(
        winstepsize=16,
        winsize=20,
        winsize_min=8,
        tol=1e-2,
        mu=1e-3,
        admm_max_iter=2,
        admm_tol=1e-2,
        gauss_pt_order=2,
        alpha=0.0,
        use_global_step=True,
        disp_smoothness=0.0,
        strain_smoothness=0.0,
        smoothness=0.0,
        method_to_compute_strain=3,
        strain_type=0,
        gridxy_roi_range=roi,
        img_size=(h, w),
        icgn_max_iter=50,
        reference_mode="accumulative",
        init_guess_mode=init_mode,
        seed_set=seed_set,
        size_of_fft_search_region=20,
    )
    defaults.update(overrides)
    return DICPara(**defaults)


class TestSeedPropagationSingleRef:
    """Two-frame accumulative: no ref switch, seed used once and reused."""

    def test_small_shift_recovers_displacement(self):
        h, w = 192, 192
        shift_x, shift_y = 1.5, 0.8
        ref = _speckle(h, w, seed=7)
        deformed = ndimage_shift(
            ref, [shift_y, shift_x], order=3, mode="reflect",
        )

        # Bootstrap mesh by running a 'previous' mode pipeline first
        # to discover node_idx for a center node. A pragmatic shortcut:
        # the mesh for this para will have a node near the image center.
        probe_para = _make_para(h, w, seed_set=None, init_mode="fft")
        masks = [np.ones((h, w)), np.ones((h, w))]
        probe_result = run_aldic(
            probe_para, [ref, deformed], masks, compute_strain=False,
        )
        coords = probe_result.dic_mesh.coordinates_fem
        # Pick the node closest to image center
        center = np.array([w / 2, h / 2])
        center_idx = int(np.argmin(np.linalg.norm(coords - center, axis=1)))

        # Now run with seed_propagation using that seed
        seed_set = SeedSet(
            seeds=(Seed(node_idx=center_idx, region_id=0),),
            ncc_threshold=0.3,
        )
        para = _make_para(h, w, seed_set=seed_set)
        result = run_aldic(
            para, [ref, deformed], masks, compute_strain=False,
        )

        assert result.result_disp[0] is not None
        U = result.result_disp[0].U
        n = result.dic_mesh.coordinates_fem.shape[0]
        assert U.shape == (2 * n,)
        # Ignore NaN nodes at the ROI edges; check median recovery
        u_vals = U[0::2]
        v_vals = U[1::2]
        u_med = np.nanmedian(u_vals)
        v_med = np.nanmedian(v_vals)
        np.testing.assert_allclose(u_med, shift_x, atol=0.3)
        np.testing.assert_allclose(v_med, shift_y, atol=0.3)

    def test_missing_seed_set_raises(self):
        h, w = 128, 128
        para = _make_para(h, w, seed_set=None)
        # validate_dicpara should refuse on construction
        ref = _speckle(h, w)
        deformed = ref.copy()
        masks = [np.ones((h, w)), np.ones((h, w))]
        with pytest.raises(ValueError, match="seed_set"):
            run_aldic(para, [ref, deformed], masks, compute_strain=False)