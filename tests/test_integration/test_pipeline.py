"""Integration tests for the run_aldic pipeline.

Tests the full pipeline with small synthetic images and meshes,
verifying end-to-end correctness of the AL-DIC workflow.
"""

import numpy as np
import pytest
from dataclasses import replace

from al_dic.core.data_structures import (
    DICMesh,
    DICPara,
    GridxyROIRange,
    PipelineResult,
)
from al_dic.core.data_structures import FrameSchedule
from al_dic.core.pipeline import (
    run_aldic,
    _auto_tune_beta,
    _restore_at_nodes,
    _compute_cumulative_displacements_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_speckle_pair(
    h: int = 64,
    w: int = 64,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reference/deformed image pair with known shift.

    Generates a random speckle pattern, then shifts it by (shift_x, shift_y)
    using Fourier phase shift for sub-pixel accuracy.
    """
    rng = np.random.RandomState(seed)
    ref = rng.rand(h, w).astype(np.float64)

    # Fourier shift
    freq = np.fft.fft2(ref)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    phase = np.exp(-2j * np.pi * (ky * shift_y + kx * shift_x))
    deformed = np.real(np.fft.ifft2(freq * phase))
    deformed = np.clip(deformed, 0, 1)

    return ref, deformed


def _make_simple_mesh(h: int = 64, w: int = 64, step: int = 16):
    """Create a uniform grid mesh for testing."""
    half_win = 20  # typical winsize/2
    xs = np.arange(half_win, w - half_win, step, dtype=np.float64)
    ys = np.arange(half_win, h - half_win, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    n_nodes = len(coords)

    # Build Q4 elements (no midside nodes → -1)
    ny, nx = len(ys), len(xs)
    elements = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n0 = iy * nx + ix
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n0 + nx
            elements.append([n0, n1, n2, n3, -1, -1, -1, -1])

    if not elements:
        elems = np.empty((0, 8), dtype=np.int64)
    else:
        elems = np.array(elements, dtype=np.int64)

    return DICMesh(coordinates_fem=coords, elements_fem=elems)


def _make_default_para(h: int = 64, w: int = 64, **overrides):
    """Create DIC parameters suitable for small synthetic tests."""
    roi = GridxyROIRange(gridx=(10, w - 10), gridy=(10, h - 10))
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
    )
    defaults.update(overrides)
    return DICPara(**defaults)


# ---------------------------------------------------------------------------
# Tests: pipeline basics
# ---------------------------------------------------------------------------


class TestRunALDICProviderEquivalence:
    """B1: run_aldic(ListFrameProvider) is byte-identical to run_aldic(list).

    Proves the frame-provider refactor adds zero numerical change -- the
    list path is wrapped in the same eager ListFrameProvider internally, and
    an externally-built provider yields the same PipelineResult.
    """

    def test_provider_matches_list_byte_for_byte(self):
        from al_dic.io.image_ops import ListFrameProvider

        ref, deformed = _make_speckle_pair(shift_x=0.5, shift_y=0.3)
        images = [ref, deformed]
        masks = [np.ones((64, 64)), np.ones((64, 64))]
        para = _make_default_para()

        res_list = run_aldic(para, images, masks)
        provider = ListFrameProvider(images, para.gridxy_roi_range)
        res_prov = run_aldic(para, provider, masks)

        assert len(res_prov.result_disp) == len(res_list.result_disp)
        for fr_l, fr_p in zip(res_list.result_disp, res_prov.result_disp):
            if fr_l is None or fr_p is None:
                assert fr_l is None and fr_p is None
                continue
            np.testing.assert_array_equal(fr_p.U, fr_l.U)


class TestRunALDICValidation:
    def test_too_few_images(self):
        """Should raise ValueError with < 2 images."""
        para = _make_default_para()
        with pytest.raises(ValueError, match="At least 2 images"):
            run_aldic(para, [np.zeros((64, 64))], [np.ones((64, 64))])

    def test_mask_length_mismatch(self):
        """Should raise ValueError when mask count != image count."""
        para = _make_default_para()
        images = [np.zeros((64, 64)), np.zeros((64, 64))]
        masks = [np.ones((64, 64))]
        with pytest.raises(ValueError, match="masks length"):
            run_aldic(para, images, masks)

    def test_auto_fft_search_when_no_mesh(self):
        """Pipeline should auto-generate mesh/U0 via FFT when not provided."""
        h, w = 128, 128
        ref, deformed = _make_speckle_pair(h, w, shift_x=2.0)
        para = _make_default_para(h, w)
        masks = [np.ones((h, w)), np.ones((h, w))]
        # No mesh, no U0 — pipeline should use integer_search
        result = run_aldic(para, [ref, deformed], masks, compute_strain=False)
        assert result.dic_mesh is not None
        assert len(result.result_disp) >= 1

    def test_stop_fn_returns_partial_immediately(self):
        """stop_fn True from the start stops cleanly and returns a result
        marked stopped_early — a user cancel is NOT an error, so run_aldic
        must not raise (the GUI keeps whatever completed instead of losing
        the whole run)."""
        h, w = 64, 64
        ref, deformed = _make_speckle_pair(h, w)
        para = _make_default_para(h, w)
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        result = run_aldic(
            para,
            [ref, deformed],
            [np.ones((h, w)), np.ones((h, w))],
            stop_fn=lambda: True,
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )
        assert result.stopped_early is True
        # Cancelled before the single pair completed -> no frames kept.
        assert len(result.result_disp) == 0

    def test_stop_fn_keeps_completed_frames(self):
        """Cancelling mid-run keeps the frames already computed instead of
        discarding the whole run (regression for the Cancel-discards-all bug;
        aligns behaviour with the Cancel button's 'frames are kept' promise)."""
        h, w = 96, 96
        ref, d1 = _make_speckle_pair(h, w, shift_x=0.4)
        _, d2 = _make_speckle_pair(h, w, shift_x=0.8)
        para = _make_default_para(h, w)
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        # Stop AFTER the first pair completes: the stop check runs once at the
        # top of each frame iteration, so return False on the frame-1 check
        # and True on the frame-2 check.
        calls = {"n": 0}

        def stop_after_first() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        result = run_aldic(
            para,
            [ref, d1, d2],
            [np.ones((h, w)), np.ones((h, w)), np.ones((h, w))],
            stop_fn=stop_after_first,
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )
        assert result.stopped_early is True
        assert result.stopped_at_frame == 3  # 1-based: stopped entering frame 3
        # Frame 1 completed and is kept; frame 2 was never computed.
        assert len(result.result_disp) == 1
        assert result.result_disp[0] is not None


class TestRunALDICZeroDisplacement:
    """Test pipeline with zero displacement (reference == deformed)."""

    def test_zero_disp_returns_pipeline_result(self):
        """Pipeline with identical images should return valid result."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        n_nodes = mesh.coordinates_fem.shape[0]
        U0 = np.zeros(2 * n_nodes)

        result = run_aldic(
            para,
            [ref, ref],  # identical images
            [np.ones((h, w)), np.ones((h, w))],
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )

        assert isinstance(result, PipelineResult)
        assert len(result.result_disp) == 1
        assert result.result_disp[0].U is not None

    def test_zero_disp_small_displacement(self):
        """Zero-displacement case should produce near-zero U."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        n_nodes = mesh.coordinates_fem.shape[0]
        U0 = np.zeros(2 * n_nodes)

        result = run_aldic(
            para,
            [ref, ref],
            [np.ones((h, w)), np.ones((h, w))],
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )

        U = result.result_disp[0].U
        # With identical images, displacement should be ~0
        assert np.nanmax(np.abs(U)) < 1.0


class TestRunALDICWithStrain:
    """Test pipeline with strain computation enabled."""

    def test_strain_computed(self):
        """Pipeline should produce StrainResult when compute_strain=True."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        n_nodes = mesh.coordinates_fem.shape[0]
        U0 = np.zeros(2 * n_nodes)

        result = run_aldic(
            para,
            [ref, ref],
            [np.ones((h, w)), np.ones((h, w))],
            mesh=mesh,
            U0=U0,
            compute_strain=True,
        )

        assert len(result.result_strain) == 1
        sr = result.result_strain[0]
        assert sr.strain_exx is not None
        assert sr.strain_exy is not None
        assert sr.strain_eyy is not None
        assert sr.strain_principal_max is not None

    def test_no_strain_when_disabled(self):
        """compute_strain=False should produce empty strain list."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        result = run_aldic(
            para,
            [ref, ref],
            [np.ones((h, w)), np.ones((h, w))],
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )

        assert len(result.result_strain) == 0


class TestRunALDICMultiFrame:
    """Test multi-frame processing."""

    def test_two_deformed_frames(self):
        """Pipeline should handle 3 images (ref + 2 deformed)."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        result = run_aldic(
            para,
            [ref, ref, ref],
            [np.ones((h, w))] * 3,
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )

        assert len(result.result_disp) == 2


class TestRefCacheLRU:
    """B2: ref_cache is a bounded LRU. Capacity must not change DIC results --
    an evicted reference bundle is recomputed identically on the next miss.
    """

    @staticmethod
    def _run(reference_mode, capacity, monkeypatch):
        import al_dic.core.pipeline as pipeline_mod
        monkeypatch.setattr(pipeline_mod, "_REF_BUNDLE_CACHE_SIZE", capacity)
        h, w = 64, 64
        base, f1 = _make_speckle_pair(shift_x=0.2, shift_y=0.1, seed=0)
        _, f2 = _make_speckle_pair(shift_x=0.4, shift_y=0.2, seed=0)
        _, f3 = _make_speckle_pair(shift_x=0.6, shift_y=0.3, seed=0)
        frames = [base, f1, f2, f3]
        masks = [np.ones((h, w))] * 4
        para = _make_default_para(
            h, w, use_global_step=False, reference_mode=reference_mode
        )
        return run_aldic(para, frames, masks, compute_strain=False)

    def _assert_capacity_invariant(self, mode, monkeypatch):
        # capacity=1 forces eviction + recompute at every reference switch;
        # capacity=1000 never evicts. Results must be byte-identical.
        res_small = self._run(mode, 1, monkeypatch)
        res_large = self._run(mode, 1000, monkeypatch)
        assert len(res_small.result_disp) == len(res_large.result_disp)
        for a, b in zip(res_small.result_disp, res_large.result_disp):
            if a is None or b is None:
                assert a is None and b is None
                continue
            np.testing.assert_array_equal(a.U, b.U)

    def test_incremental_capacity_invariant(self, monkeypatch):
        self._assert_capacity_invariant("incremental", monkeypatch)

    def test_accumulative_capacity_invariant(self, monkeypatch):
        self._assert_capacity_invariant("accumulative", monkeypatch)

    @staticmethod
    def _run_all_caps(reference_mode, cap, monkeypatch):
        # Bound ALL three reference-keyed caches (ref bundle + the two heavy
        # solver precompute caches) to the same cap. use_global_step=True so
        # subpb1_precompute_cache is exercised (it is only used under global
        # step); icgn_ctx_cache is exercised every frame regardless.
        import al_dic.core.pipeline as pipeline_mod
        monkeypatch.setattr(pipeline_mod, "_REF_BUNDLE_CACHE_SIZE", cap)
        monkeypatch.setattr(pipeline_mod, "_PRECOMPUTE_CACHE_SIZE", cap)
        h, w = 64, 64
        base, f1 = _make_speckle_pair(shift_x=0.2, shift_y=0.1, seed=0)
        _, f2 = _make_speckle_pair(shift_x=0.4, shift_y=0.2, seed=0)
        _, f3 = _make_speckle_pair(shift_x=0.6, shift_y=0.3, seed=0)
        frames = [base, f1, f2, f3]
        masks = [np.ones((h, w))] * 4
        para = _make_default_para(
            h, w, use_global_step=True, reference_mode=reference_mode
        )
        return run_aldic(para, frames, masks, compute_strain=False)

    def test_precompute_caches_capacity_invariant_incremental(self, monkeypatch):
        # cap=1 forces evict+recompute of the 6-DOF/2-DOF precompute every
        # reference switch; cap=1000 never evicts. Byte-identical required.
        small = self._run_all_caps("incremental", 1, monkeypatch)
        large = self._run_all_caps("incremental", 1000, monkeypatch)
        assert len(small.result_disp) == len(large.result_disp)
        for a, b in zip(small.result_disp, large.result_disp):
            if a is None or b is None:
                assert a is None and b is None
                continue
            np.testing.assert_array_equal(a.U, b.U)


class TestRunALDICAccumulative:
    """Test accumulative reference mode."""

    def test_accumulative_mode(self):
        """Accumulative mode: U_accum should equal U."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False, reference_mode="accumulative")
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        result = run_aldic(
            para,
            [ref, ref],
            [np.ones((h, w))] * 2,
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )

        disp = result.result_disp[0]
        assert disp.U_accum is not None
        np.testing.assert_array_equal(disp.U, disp.U_accum)


class TestRunALDICProgressCallback:
    """Test progress and stop callbacks."""

    def test_progress_fn_called(self):
        """Progress callback should be invoked."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        calls = []
        run_aldic(
            para,
            [ref, ref],
            [np.ones((h, w))] * 2,
            progress_fn=lambda f, m: calls.append((f, m)),
            mesh=mesh,
            U0=U0,
            compute_strain=False,
        )

        assert len(calls) > 0
        # Last call should be completion
        assert calls[-1][0] == 1.0

    @pytest.mark.parametrize("compute_strain", [False, True])
    @pytest.mark.parametrize("n_frames", [2, 3, 5])
    def test_progress_stays_in_range_and_never_goes_back(self, n_frames, compute_strain):
        """The frame loop split its budget over n_frames - 2 pairs instead of
        n_frames - 1, so the last frame ran past 100 % (103 % for 28 images,
        ~190 % for 3); strains and assembly then reported lower values."""
        h, w = 64, 64
        ref = np.random.RandomState(42).rand(h, w).astype(np.float64)
        para = _make_default_para(h, w, use_global_step=False)
        mesh = _make_simple_mesh(h, w, step=16)
        U0 = np.zeros(2 * mesh.coordinates_fem.shape[0])

        calls = []
        run_aldic(
            para,
            [ref] * n_frames,
            [np.ones((h, w))] * n_frames,
            progress_fn=lambda f, m: calls.append((f, m)),
            mesh=mesh,
            U0=U0,
            compute_strain=compute_strain,
        )

        fractions = [f for f, _ in calls]
        assert min(fractions) >= 0.0
        assert max(fractions) <= 1.0, calls
        assert all(b >= a for a, b in zip(fractions, fractions[1:])), calls
        assert fractions[-1] == 1.0


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestRestoreAtNodes:
    def test_restore_F(self):
        """Should copy 4 components per node from source."""
        n = 5
        target = np.zeros(4 * n)
        source = np.ones(4 * n)
        indices = np.array([1, 3], dtype=np.int64)

        result = _restore_at_nodes(target, source, indices, 4)

        assert result[4] == 1.0  # node 1, F11
        assert result[5] == 1.0  # node 1, F21
        assert result[6] == 1.0  # node 1, F12
        assert result[7] == 1.0  # node 1, F22
        assert result[0] == 0.0  # node 0 unchanged
        assert result[8] == 0.0  # node 2 unchanged

    def test_restore_U(self):
        """Should copy 2 components per node from source."""
        n = 5
        target = np.zeros(2 * n)
        source = np.arange(2 * n, dtype=np.float64)
        indices = np.array([2], dtype=np.int64)

        result = _restore_at_nodes(target, source, indices, 2)

        assert result[4] == 4.0  # node 2, u
        assert result[5] == 5.0  # node 2, v
        assert result[0] == 0.0  # node 0 unchanged

    def test_empty_indices(self):
        """Empty node indices should return target unchanged."""
        target = np.ones(10)
        source = np.zeros(10)
        indices = np.array([], dtype=np.int64)

        result = _restore_at_nodes(target, source, indices, 2)
        np.testing.assert_array_equal(result, target)

    def test_does_not_modify_input(self):
        """Input arrays should not be modified."""
        target = np.zeros(8)
        target_orig = target.copy()
        source = np.ones(8)
        _restore_at_nodes(target, source, np.array([0], dtype=np.int64), 4)
        np.testing.assert_array_equal(target, target_orig)


class TestCumulativeDisplacements:
    def test_accumulative_mode(self):
        """Accumulative mode: U_accum = U."""
        from al_dic.core.data_structures import FrameResult

        coords = np.array([[0, 0], [16, 0], [0, 16]], dtype=np.float64)
        elems = np.empty((0, 8), dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elems)

        U = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result_disp = [FrameResult(U=U)]
        result_fe_mesh = [mesh]

        schedule = FrameSchedule.from_mode("accumulative", 2)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_fe_mesh, 2, schedule,
        )

        np.testing.assert_array_equal(result[0].U_accum, U)

    def test_incremental_identity(self):
        """Incremental mode with zero displacement: U_accum = 0."""
        from al_dic.core.data_structures import FrameResult

        coords = np.array([[0, 0], [16, 0], [0, 16]], dtype=np.float64)
        elems = np.empty((0, 8), dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elems)

        U = np.zeros(6)
        result_disp = [FrameResult(U=U)]
        result_fe_mesh = [mesh]

        schedule = FrameSchedule.from_mode("incremental", 2)
        result = _compute_cumulative_displacements_tree(
            result_disp, result_fe_mesh, 2, schedule,
        )

        np.testing.assert_allclose(result[0].U_accum, 0.0, atol=1e-10)
