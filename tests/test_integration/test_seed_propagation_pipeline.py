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

from al_dic.core.data_structures import DICPara, FrameSchedule, GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.solver.seed_propagation import (
    Seed,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedSet,
)


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


class TestSeedPropagationRefSwitch:
    """C10b: ref switch triggers seed warp; pipeline keeps running."""

    def test_ref_switch_auto_warps_seeds(self):
        """4-frame sequence with from_every_n(2) → ref switches at frame 3.

        Images constructed as uniform shifts of Image 0:
          Image 0 = base speckle
          Image 1 = Image 0 + 1*(dx, dy)
          Image 2 = Image 0 + 2*(dx, dy)
          Image 3 = Image 0 + 3*(dx, dy)

        Schedule: refs = (0, 0, 2). At frame 3 the ref switches from
        Image 0 to Image 2; seeds placed on Image 0's mesh must be
        warped to Image 2's mesh. Frame 3 should recover displacement
        (dx, dy) — the Image 2 -> Image 3 motion.
        """
        h, w = 192, 192
        dx, dy = 1.2, 0.5
        ref0 = _speckle(h, w, seed=11)
        images = [ref0]
        for k in (1, 2, 3):
            images.append(
                ndimage_shift(
                    ref0, [k * dy, k * dx], order=3, mode="reflect",
                ),
            )
        masks = [np.ones((h, w)) for _ in images]

        # Probe run to discover node_idx near image center
        probe_para = _make_para(h, w, seed_set=None, init_mode="fft")
        probe_result = run_aldic(
            probe_para, images[:2], masks[:2], compute_strain=False,
        )
        coords = probe_result.dic_mesh.coordinates_fem
        center = np.array([w / 2, h / 2])
        seed_idx = int(np.argmin(np.linalg.norm(coords - center, axis=1)))

        seed_set = SeedSet(
            seeds=(Seed(node_idx=seed_idx, region_id=0),),
            ncc_threshold=0.3,
        )
        schedule = FrameSchedule.from_every_n(2, n_frames=4)
        # ref_indices should be (0, 0, 2) — verify the schedule actually
        # triggers a ref change, otherwise the test is not exercising warp.
        assert schedule.ref_indices == (0, 0, 2)

        para = _make_para(
            h, w, seed_set=seed_set,
            frame_schedule=schedule,
            reference_mode="incremental",  # frame_schedule overrides
        )
        result = run_aldic(
            para, images, masks, compute_strain=False,
        )

        assert result.result_disp[2] is not None, "Frame 3 after ref switch failed"
        U3 = result.result_disp[2].U
        # U3 is the displacement from Image 2 -> Image 3 (after ref switch)
        # Expected ≈ (dx, dy) uniformly
        u_med = np.nanmedian(U3[0::2])
        v_med = np.nanmedian(U3[1::2])
        np.testing.assert_allclose(u_med, dx, atol=0.35)
        np.testing.assert_allclose(v_med, dy, atol=0.35)


class TestSeedPropagationQualityGates:
    """Verify fail-loud behavior when a user picks a bad seed location.
    Any SeedPropagationError subclass is acceptable — the contract is
    'do not silently degrade to wrong output'. Unit tests in
    test_solver/test_seed_propagation.py exercise each exception type
    in isolation.
    """

    def test_seed_in_low_texture_region_raises(self):
        """Seed placed in a uniform zone triggers a SeedPropagationError.

        Ascending failure modes depending on setup:
          - NCC < threshold (SeedNCCBelowThreshold) — if search-window
            fits in image at the clamp-down search_radius.
          - Out-of-bounds (SeedPropagationError 'window out of image
            bounds') — if auto-expand pushes the window past the edge
            because uniform regions keep clipping the peak.

        Either is acceptable: the test confirms the pipeline propagates
        a typed error rather than silently continuing with garbage.
        """
        h, w = 192, 192
        # Textured left half, uniform right half.
        ref = _speckle(h, w, seed=13)
        ref[:, w // 2:] = 0.5  # kill all texture in right half
        deformed = ndimage_shift(
            ref, [0.0, 1.5], order=3, mode="reflect",
        )
        masks = [np.ones((h, w)), np.ones((h, w))]

        # Probe mesh to pick a node in the uniform right half
        probe_para = _make_para(h, w, seed_set=None, init_mode="fft")
        probe = run_aldic(
            probe_para, [ref, deformed], masks, compute_strain=False,
        )
        coords = probe.dic_mesh.coordinates_fem
        # Nodes with x > w/2 are in the uniform zone
        right_half = np.where(coords[:, 0] > w / 2 + 20)[0]
        bad_seed_idx = int(right_half[len(right_half) // 2])

        seed_set = SeedSet(
            seeds=(Seed(node_idx=bad_seed_idx, region_id=0),),
            ncc_threshold=0.85,  # strict — uniform region won't reach this
        )
        para = _make_para(h, w, seed_set=seed_set)

        with pytest.raises((SeedNCCBelowThreshold, SeedPropagationError)):
            run_aldic(
                para, [ref, deformed], masks, compute_strain=False,
            )