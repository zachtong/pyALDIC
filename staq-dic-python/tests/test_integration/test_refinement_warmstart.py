"""Regression tests for the refinement + warm-start + canonical-mesh bugs.

Three bugs were observed when adaptive refinement was wired into the
pipeline:

1. **Accuracy degradation** — accumulative + refined mode silently lost
   sibling-reuse warm-start, so every frame fell back to FFT init.  On
   long sequences with growing displacement (e.g. bubble 30→150→30) the
   FFT auto-retry could not always recover and later frames degraded.
2. **Performance regression** — same root cause; ~25 frames × 1-6 FFT
   retries per frame instead of one FFT for the whole sequence.
3. **Overlay crash on incremental + refined** — ``PipelineResult.dic_mesh``
   was set to the *last* frame's mesh, but ``U_accum`` lives on the
   frame-0 reference mesh.  GUI overlay rendering then raised
   ``ValueError: different number of values and points``.

These tests pin the contracts that the fix must satisfy:

- **test_a (warmstart_call_count)**: in accumulative + refined mode,
  ``integer_search`` must be called at most once per reference frame
  across the whole sequence (sibling reuse must survive refinement).
- **test_b (canonical mesh shape)**: in incremental + refined mode,
  ``result.dic_mesh.coordinates_fem`` must agree with every
  ``result_disp[i].U_accum`` length, because the cumulative
  displacement transform always lives on the frame-0 mesh.
- **test_c (accumulative canonical mesh shape)**: same invariant for
  accumulative + refined, where the mesh stays the same across frames
  but the bug still appears if ``dic_mesh`` is set from the loop
  variable instead of frame 0's snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

from staq_dic.core import pipeline as pipeline_module
from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from staq_dic.mesh.refinement import RefinementPolicy


def _make_speckle(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic gaussian-blurred speckle (matches existing tests)."""
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.random((h, w)), sigma=1.5)


def _shifted(base: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return ``base`` translated by (dx, dy) px (cubic resample)."""
    return nd_shift(base, [dy, dx], order=3, mode="nearest")


def _make_mask_with_hole(h: int, w: int, radius: int = 18) -> np.ndarray:
    """Solid mask with a circular hole at the centre — triggers refinement."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.ones((h, w), dtype=np.float64)
    mask[(xx - w // 2) ** 2 + (yy - h // 2) ** 2 < radius ** 2] = 0.0
    return mask


def _make_growing_hole_masks(
    h: int, w: int, radii: list[int]
) -> list[np.ndarray]:
    """Per-frame masks with a centred hole that grows over the sequence.

    Different masks per frame are required to expose the
    ``PipelineResult.dic_mesh`` bug: when masks differ, mesh trimming
    produces a different node count per frame, so the last-frame loop
    variable diverges from the frame-0 snapshot.
    """
    return [_make_mask_with_hole(h, w, radius=r) for r in radii]


def _make_para(
    h: int,
    w: int,
    *,
    reference_mode: str,
    margin: int = 16,
):
    """Common DICPara for these tests.

    ``init_fft_search_method=1`` forces the direct-FFT path so that
    ``integer_search`` (rather than ``integer_search_pyramid``) is used —
    that's the function we monkeypatch for the call-count assertion.
    """
    return dicpara_default(
        winstepsize=16,
        winsize=32,
        winsize_min=4,
        size_of_fft_search_region=20,
        reference_mode=reference_mode,
        init_fft_search_method=1,
        gridxy_roi_range=GridxyROIRange(
            gridx=(margin, w - margin),
            gridy=(margin, h - margin),
        ),
    )


def _make_refinement_policy() -> RefinementPolicy:
    return RefinementPolicy(
        pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
    )


# ---------------------------------------------------------------------------
# Bug 1+2: warm-start must survive refinement in accumulative mode
# ---------------------------------------------------------------------------
class TestRefinementWarmStart:
    def test_accumulative_refined_calls_integer_search_at_most_once(
        self, monkeypatch
    ):
        """In accumulative + refined mode, ``integer_search`` must be
        called at most once for the whole sequence.

        Sibling reuse should warm-start every frame after the first; the
        bug was that the refinement guard cleared ``current_U0`` and
        forced an FFT search per frame.
        """
        h, w = 128, 128
        base = _make_speckle(h, w, seed=11)
        # Small, monotone translations: stay well inside the FFT search
        # window so the auto-retry loop never fires (would inflate the
        # call count and confuse the test).
        frames = [
            base,
            _shifted(base, 1.0, 0.0),
            _shifted(base, 2.0, 0.0),
            _shifted(base, 3.0, 0.0),
        ]
        mask = _make_mask_with_hole(h, w)
        masks = [mask] * len(frames)
        para = _make_para(h, w, reference_mode="accumulative")
        policy = _make_refinement_policy()

        call_count = {"n": 0}
        original = pipeline_module.integer_search

        def counting_integer_search(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            pipeline_module, "integer_search", counting_integer_search
        )

        result = run_aldic(
            para, frames, masks,
            compute_strain=False, refinement_policy=policy,
        )

        # Sanity: the run actually completed.
        assert len(result.result_disp) == len(frames) - 1
        assert all(r is not None for r in result.result_disp)

        # Bug present: 4+ FFT calls (one per frame, possibly with
        # retries).  After fix: exactly 1 (frame 1 only).
        assert call_count["n"] <= 1, (
            f"integer_search called {call_count['n']} times — sibling "
            f"reuse is broken under refinement (expected <= 1 for "
            f"{len(frames)} accumulative-mode frames)."
        )


# ---------------------------------------------------------------------------
# Bug 3: PipelineResult.dic_mesh must be the canonical (frame-0) mesh
# ---------------------------------------------------------------------------
class TestCanonicalDicMesh:
    def test_incremental_refined_dic_mesh_matches_u_accum(self):
        """In incremental + refined mode with per-frame masks, every
        frame has its own (differently sized) mesh.  ``result.dic_mesh``
        must still be the frame-0 mesh because that is the reference
        for ``U_accum`` produced by the cumulative transform.

        The growing hole simulates the bubble dataset, which is the
        case that exposed this bug in the GUI.
        """
        h, w = 128, 128
        base = _make_speckle(h, w, seed=22)
        frames = [
            base,
            _shifted(base, 0.7, 0.0),
            _shifted(base, 1.4, 0.0),
        ]
        # Growing-hole masks → frame-0 mesh has more nodes than later
        # frames, so a stale loop-variable assignment is detectable.
        masks = _make_growing_hole_masks(h, w, [16, 22, 28])
        para = _make_para(h, w, reference_mode="incremental")
        policy = _make_refinement_policy()

        result = run_aldic(
            para, frames, masks,
            compute_strain=False, refinement_policy=policy,
        )

        n_canon = result.dic_mesh.coordinates_fem.shape[0]
        assert n_canon > 0, "canonical mesh should not be empty"

        # The frame-0 mesh snapshot is the source of U_accum.
        frame0_mesh = result.result_fe_mesh_each_frame[0]
        assert frame0_mesh.coordinates_fem.shape == (n_canon, 2), (
            "result.dic_mesh must be the frame-0 mesh, not the last "
            "frame's refined mesh"
        )

        # Every U_accum entry must be sized for the canonical mesh.
        for i, frame_res in enumerate(result.result_disp):
            assert frame_res.U_accum is not None, (
                f"frame {i}: U_accum is None"
            )
            assert frame_res.U_accum.shape[0] == 2 * n_canon, (
                f"frame {i}: U_accum has length {frame_res.U_accum.shape[0]} "
                f"but canonical mesh has {n_canon} nodes (expected "
                f"{2 * n_canon})"
            )

    def test_accumulative_refined_dic_mesh_matches_u_accum(self):
        """Same invariant for accumulative + refined.  In accumulative
        mode all frames share ref=0, but per-frame masks (like the
        bubble dataset) still produce different refined meshes — the
        loop variable diverges from the frame-0 snapshot just as in
        incremental mode.
        """
        h, w = 128, 128
        base = _make_speckle(h, w, seed=33)
        frames = [
            base,
            _shifted(base, 1.0, 0.0),
            _shifted(base, 2.0, 0.0),
        ]
        masks = _make_growing_hole_masks(h, w, [16, 22, 28])
        para = _make_para(h, w, reference_mode="accumulative")
        policy = _make_refinement_policy()

        result = run_aldic(
            para, frames, masks,
            compute_strain=False, refinement_policy=policy,
        )

        n_canon = result.dic_mesh.coordinates_fem.shape[0]
        assert n_canon > 0
        for i, frame_res in enumerate(result.result_disp):
            assert frame_res.U_accum is not None
            assert frame_res.U_accum.shape[0] == 2 * n_canon, (
                f"frame {i}: U_accum / dic_mesh shape mismatch "
                f"({frame_res.U_accum.shape[0]} vs 2*{n_canon})"
            )
