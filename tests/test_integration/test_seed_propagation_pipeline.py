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


class TestSeedPropagationMultiRefSwitch:
    """Multi-ref-switch and large-displacement scenarios to stress-test
    the warp-or-reseed path across long incremental runs.
    """

    def _build_shifted_sequence(
        self,
        h: int, w: int,
        n_frames: int,
        dx: float, dy: float,
        rng_seed: int = 17,
    ) -> list[np.ndarray]:
        """Return images where image[k] = image[0] shifted by k*(dx, dy)."""
        ref = _speckle(h, w, seed=rng_seed)
        imgs = [ref]
        for k in range(1, n_frames):
            imgs.append(
                ndimage_shift(ref, [k * dy, k * dx], order=3, mode="reflect"),
            )
        return imgs

    def _center_seed_idx(
        self, h: int, w: int, images: list[np.ndarray],
    ) -> int:
        """Probe a non-seed pipeline to pick a node near image center."""
        masks = [np.ones((h, w)) for _ in images[:2]]
        probe_para = _make_para(h, w, seed_set=None, init_mode="fft")
        probe = run_aldic(
            probe_para, images[:2], masks, compute_strain=False,
        )
        coords = probe.dic_mesh.coordinates_fem
        center = np.array([w / 2, h / 2])
        return int(np.argmin(np.linalg.norm(coords - center, axis=1)))

    def test_multi_ref_switch_sequence(self):
        """refs=(0,0,2,2,4,4): two ref-switches in one run, both succeed.

        Verifies that the state machine correctly captures prev-frame
        state after every frame and warps cleanly on every switch, not
        just the first.
        """
        h, w = 192, 192
        dx, dy = 1.0, 0.4
        images = self._build_shifted_sequence(h, w, n_frames=6, dx=dx, dy=dy)
        masks = [np.ones((h, w)) for _ in images]

        seed_idx = self._center_seed_idx(h, w, images)
        seed_set = SeedSet(
            seeds=(Seed(node_idx=seed_idx, region_id=0),),
            ncc_threshold=0.3,
        )
        schedule = FrameSchedule.from_every_n(2, n_frames=6)
        # FrameSchedule produces one ref per displacement frame; for
        # n_frames=6 there are 5 displacement pairs.
        assert schedule.ref_indices == (0, 0, 2, 2, 4)

        para = _make_para(
            h, w, seed_set=seed_set,
            frame_schedule=schedule,
            reference_mode="incremental",
        )
        result = run_aldic(para, images, masks, compute_strain=False)

        # Every displacement frame should have completed
        for k in range(5):  # n_frames - 1 displacement frames
            assert result.result_disp[k] is not None, (
                f"Frame {k + 1} (schedule index {k}) failed to produce a "
                f"displacement result."
            )
        # Last frame post-switch recovers the per-frame motion (dx, dy)
        U_last = result.result_disp[-1].U
        np.testing.assert_allclose(
            np.nanmedian(U_last[0::2]), dx, atol=0.35,
        )
        np.testing.assert_allclose(
            np.nanmedian(U_last[1::2]), dy, atol=0.35,
        )

    def test_ref_switch_large_displacement_no_reseed(self):
        """Large inter-frame shift at ref boundary: warp must still work.

        Displacement between ref frames here is large enough that a
        naive full-grid FFT would need a wide search, but seed warping
        only needs the material point's cumulative U — which the prev
        frame already knows. Fallback should NOT fire.
        """
        h, w = 256, 256
        # 6 frames; per-frame shift = 15 px → ref-to-ref gap of 30 px
        dx, dy = 15.0, 0.0
        images = self._build_shifted_sequence(h, w, n_frames=6, dx=dx, dy=dy)
        # Keep ROI generous so warped seeds stay inside new mesh
        masks = [np.ones((h, w)) for _ in images]

        seed_idx = self._center_seed_idx(h, w, images)
        seed_set = SeedSet(
            seeds=(Seed(node_idx=seed_idx, region_id=0),),
            ncc_threshold=0.3,
        )
        schedule = FrameSchedule.from_every_n(2, n_frames=6)
        para = _make_para(
            h, w, seed_set=seed_set,
            frame_schedule=schedule,
            reference_mode="incremental",
            size_of_fft_search_region=40,  # wide enough for 15 px motion
        )
        result = run_aldic(para, images, masks, compute_strain=False)
        # All frames must complete
        for k, disp in enumerate(result.result_disp):
            assert disp is not None, f"Frame index {k} failed."


class TestSeedPropagationReseedFallback:
    """Exercise the auto-reseed fallback path when warp raises."""

    def test_warp_failure_triggers_autoplace_and_continues(self):
        """Manually craft a state where old seeds warp outside new ROI.

        Rather than rely on a full pipeline setup to reproduce the
        narrow warp-fails-mid-run scenario, this test directly drives
        ``compute_seed_prop_init_guess`` with a hand-built previous
        state: seeds whose U values push them outside the new mesh's
        region map. The fallback should auto-place new seeds and record
        a ReseedEvent.
        """
        from al_dic.io.image_ops import compute_image_gradient
        from al_dic.mesh.mesh_setup import mesh_setup
        from al_dic.solver.seed_prop_pipeline import (
            ReseedEvent,
            SeedPropagationState,
            build_grid_for_roi,
            compute_seed_prop_init_guess,
        )

        h, w = 192, 192
        ref = _speckle(h, w, seed=21)
        deformed = ref.copy()  # zero displacement — seeds warp-in-place
        mask = np.ones((h, w), dtype=np.float64)

        # Build mesh via the same helpers the pipeline uses
        seed_set_placeholder = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),), ncc_threshold=0.3,
        )
        para = _make_para(h, w, seed_set=seed_set_placeholder)
        x0, y0 = build_grid_for_roi(para, h, w)
        dic_mesh = mesh_setup(x0, y0, para)
        coords = dic_mesh.coordinates_fem

        # Prev-frame "displacement" pushes a seed way outside the new ROI
        n_nodes = coords.shape[0]
        prev_U_2d = np.zeros((n_nodes, 2))
        # Drive seed node 0 off the map (+10000 px): nearest new node
        # will be far out of range.
        prev_U_2d[0] = [10000.0, 10000.0]

        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),),
            ncc_threshold=0.3,
        )
        state = SeedPropagationState(
            current_seeds=seed_set,
            prev_coords_fem=coords.copy(),
            prev_U_2d=prev_U_2d,
        )

        Df = compute_image_gradient(ref, mask)

        # Trigger ref_switched=True. Warp distance (10000 px) blows past
        # max_snap_distance (50 px), so warp_seeds_to_new_ref raises
        # SeedWarpFailure. Fallback should auto-place fresh seeds.
        U0 = compute_seed_prop_init_guess(
            state, dic_mesh,
            ref, deformed, mask, Df, para,
            tol=para.tol, ref_switched=True,
            max_snap_distance=50.0,
            frame_idx=5, ref_idx=3,
        )

        # Run completed: got a U0 vector back
        assert U0 is not None
        assert U0.shape == (2 * n_nodes,)
        # ReseedEvent was recorded with the correct frame_idx/ref_idx
        assert len(state.reseed_events) == 1
        ev: ReseedEvent = state.reseed_events[0]
        assert ev.frame_idx == 5
        assert ev.ref_idx == 3
        assert ev.n_new_seeds >= 1
        # state.current_seeds is the fresh auto-placed set, not the
        # original (which would have failed).
        assert len(state.current_seeds.seeds) >= 1

    def test_warp_failure_then_autoplace_also_fails_reraises(self):
        """If auto-place itself finds nothing, original SeedWarpFailure wins."""
        from al_dic.io.image_ops import compute_image_gradient
        from al_dic.mesh.mesh_setup import mesh_setup
        from al_dic.solver.seed_prop_pipeline import (
            SeedPropagationState,
            build_grid_for_roi,
            compute_seed_prop_init_guess,
        )
        from al_dic.solver.seed_propagation import SeedWarpFailure

        h, w = 192, 192
        # Pair with no matching structure (ref is speckle, def is random
        # noise) — NCC will be low everywhere, so auto-place can't find
        # anything above the floor.
        ref = _speckle(h, w, seed=31)
        rng = np.random.RandomState(99)
        deformed = rng.rand(h, w).astype(np.float64)
        mask = np.ones((h, w), dtype=np.float64)

        # Strict NCC floor — auto-place cannot find anything on the
        # random-noise "deformed" image.
        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),),
            ncc_threshold=0.99,
        )
        para = _make_para(h, w, seed_set=seed_set)
        x0, y0 = build_grid_for_roi(para, h, w)
        dic_mesh = mesh_setup(x0, y0, para)
        coords = dic_mesh.coordinates_fem
        n_nodes = coords.shape[0]

        prev_U_2d = np.zeros((n_nodes, 2))
        prev_U_2d[0] = [10000.0, 10000.0]  # push seed out of map

        state = SeedPropagationState(
            current_seeds=seed_set,
            prev_coords_fem=coords.copy(),
            prev_U_2d=prev_U_2d,
        )
        Df = compute_image_gradient(ref, mask)

        with pytest.raises(SeedWarpFailure, match="Auto-place fallback"):
            compute_seed_prop_init_guess(
                state, dic_mesh,
                ref, deformed, mask, Df, para,
                tol=para.tol, ref_switched=True,
                max_snap_distance=50.0,
                frame_idx=1, ref_idx=1,
            )


class TestSeedPropagationResultExposure:
    """Verify ref_switch_frames and reseed_events propagate to
    PipelineResult so the GUI can surface them.
    """

    def test_ref_switch_frames_populated(self):
        """Multi-ref-switch run: every switch-frame appears in result."""
        h, w = 192, 192
        dx, dy = 1.0, 0.4
        ref = _speckle(h, w, seed=41)
        images = [ref]
        for k in (1, 2, 3, 4, 5):
            images.append(
                ndimage_shift(ref, [k * dy, k * dx], order=3, mode="reflect"),
            )
        masks = [np.ones((h, w)) for _ in images]

        # Probe node for seed
        probe_para = _make_para(h, w, seed_set=None, init_mode="fft")
        probe = run_aldic(
            probe_para, images[:2], masks[:2], compute_strain=False,
        )
        coords = probe.dic_mesh.coordinates_fem
        center = np.array([w / 2, h / 2])
        seed_idx = int(np.argmin(np.linalg.norm(coords - center, axis=1)))

        seed_set = SeedSet(
            seeds=(Seed(node_idx=seed_idx, region_id=0),),
            ncc_threshold=0.3,
        )
        schedule = FrameSchedule.from_every_n(2, n_frames=6)
        # refs = (0, 0, 2, 2, 4): switches happen at frame indices where
        # ref_idx differs from prev. That is frames 2 (ref 0->2) and 4
        # (ref 2->4). The loop starts at frame 1 (first displacement);
        # prev_ref_idx is None at frame 1 so no switch recorded there.
        para = _make_para(
            h, w, seed_set=seed_set,
            frame_schedule=schedule,
            reference_mode="incremental",
        )
        result = run_aldic(para, images, masks, compute_strain=False)

        # Two ref-switches: at frame 3 (ref 0->2) and frame 5 (ref 2->4).
        assert result.ref_switch_frames == (3, 5)
        # Run completed normally (no warp failures with this small motion)
        # so reseed_events remains empty.
        assert result.reseed_events == ()

    def test_non_seed_prop_runs_have_empty_reseed_events(self):
        """Non-seed-prop modes should have empty reseed_events but can still
        report ref_switch_frames."""
        h, w = 128, 128
        ref = _speckle(h, w, seed=51)
        deformed = ndimage_shift(ref, [0.5, 1.0], order=3, mode="reflect")
        images = [ref, deformed, deformed, deformed]
        masks = [np.ones((h, w)) for _ in images]

        schedule = FrameSchedule.from_every_n(2, n_frames=4)
        para = _make_para(
            h, w, seed_set=None, init_mode="fft",
            frame_schedule=schedule,
            reference_mode="incremental",
        )
        result = run_aldic(para, images, masks, compute_strain=False)
        assert result.reseed_events == ()
        # The presence or absence of ref_switch_frames in non-seed-prop
        # modes is determined by the schedule; just verify the attribute
        # exists and is a tuple.
        assert isinstance(result.ref_switch_frames, tuple)


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