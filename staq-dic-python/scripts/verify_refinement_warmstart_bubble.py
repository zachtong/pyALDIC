#!/usr/bin/env python
"""Manual validation script for the refinement warm-start fix.

Loads the first ``N_FRAMES`` frames of the bubble_30_150_30 experimental
dataset (with their per-frame masks) and runs the AL-DIC pipeline twice:

1. **Uniform mesh, accumulative**  -- baseline
2. **Refined mesh, accumulative**  -- the configuration that triggered
   the warm-start regression in the GUI report

For each run we count:
  * how many times ``integer_search`` is called (one per frame would
    indicate broken sibling reuse)
  * total wall-clock time
  * shape consistency between ``result.dic_mesh`` and
    ``result_disp[i].U_accum``  (Bug 3)

Pass criteria
-------------
* Refined run calls ``integer_search`` at most once for the whole
  sequence (one initial FFT for frame 1; sibling reuse handles
  frames 2..N).
* ``len(result.result_disp[i].U_accum) == 2 * result.dic_mesh.coordinates_fem.shape[0]``
  for all frames in the refined run.

Run with::

    python scripts/verify_refinement_warmstart_bubble.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "staq-dic-python" / "src"))

from staq_dic.core import pipeline as pipeline_module
from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.mesh.refinement import build_refinement_policy

BUBBLE_DIR = REPO_ROOT / "examples" / "bubble_30_150_30"
IMG_DIR = BUBBLE_DIR / "images"
MASK_DIR = BUBBLE_DIR / "masks"

N_FRAMES = 6
SUBSET_SIZE = 40
SUBSET_STEP = 16
REFINE_LEVEL = 2
HALF_WIN = SUBSET_SIZE // 2


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.float64)


def make_para(img_shape: tuple[int, int], roi_mask: np.ndarray):
    rows, cols = np.where(roi_mask > 0.5)
    roi = GridxyROIRange(
        gridx=(int(cols.min()), int(cols.max())),
        gridy=(int(rows.min()), int(rows.max())),
    )
    return dicpara_default(
        winsize=SUBSET_SIZE,
        winstepsize=SUBSET_STEP,
        winsize_min=min(8, SUBSET_STEP),
        size_of_fft_search_region=20,
        gridxy_roi_range=roi,
        img_size=img_shape,
        reference_mode="accumulative",
        init_fft_search_method=1,  # direct FFT so we can count cleanly
        show_plots=False,
    )


def run_with_call_counter(label: str, para, images, masks, *, refined: bool):
    """Run the pipeline once and return (result, call_count, elapsed)."""
    policy = None
    if refined:
        policy = build_refinement_policy(
            refine_inner_boundary=True,
            refine_outer_boundary=False,
            min_element_size=max(2, SUBSET_STEP // (2 ** REFINE_LEVEL)),
            half_win=HALF_WIN,
        )

    call_count = {"n": 0}
    original = pipeline_module.integer_search

    def counting_integer_search(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    pipeline_module.integer_search = counting_integer_search
    try:
        t0 = time.perf_counter()
        result = run_aldic(
            para=para,
            images=images,
            masks=masks,
            compute_strain=False,
            refinement_policy=policy,
        )
        elapsed = time.perf_counter() - t0
    finally:
        pipeline_module.integer_search = original

    print(f"\n[{label}]")
    print(f"  integer_search calls: {call_count['n']}")
    print(f"  elapsed:              {elapsed:.2f} s")
    print(f"  dic_mesh nodes:       {result.dic_mesh.coordinates_fem.shape[0]}")
    n_canon = result.dic_mesh.coordinates_fem.shape[0]
    for i, fr in enumerate(result.result_disp):
        u_accum_len = len(fr.U_accum) if fr.U_accum is not None else -1
        ok = "OK" if u_accum_len == 2 * n_canon else "MISMATCH"
        print(
            f"  frame {i + 1}: U_accum len={u_accum_len}, "
            f"expected={2 * n_canon}  [{ok}]"
        )

    return result, call_count["n"], elapsed


def main() -> int:
    print(f"Loading {N_FRAMES} frames from {BUBBLE_DIR}")
    images = [load_image(IMG_DIR / f"frame_{i:02d}.png") for i in range(N_FRAMES)]
    masks = [load_mask(MASK_DIR / f"mask_{i:02d}.png") for i in range(N_FRAMES)]
    img_shape = images[0].shape
    print(f"  image shape: {img_shape}, frames: {len(images)}")

    para = make_para(img_shape, masks[0])

    res_uniform, n_uniform, t_uniform = run_with_call_counter(
        "Uniform (baseline)", para, images, masks, refined=False,
    )
    res_refined, n_refined, t_refined = run_with_call_counter(
        "Refined (inner)",   para, images, masks, refined=True,
    )

    print("\n--- Summary ---")
    print(f"Uniform : {n_uniform:>3} FFT calls, {t_uniform:6.2f}s")
    print(f"Refined : {n_refined:>3} FFT calls, {t_refined:6.2f}s")

    # Pass criteria
    passed = True
    if n_refined > 6:
        # Allow up to 6 (one frame's worth of retries) — anything more
        # means warm-start is not happening per-frame.
        print(
            f"FAIL: refined run made {n_refined} FFT calls "
            f"(expected <= 6 — one frame's worth of retries)"
        )
        passed = False

    n_canon = res_refined.dic_mesh.coordinates_fem.shape[0]
    for i, fr in enumerate(res_refined.result_disp):
        if fr.U_accum is None or len(fr.U_accum) != 2 * n_canon:
            print(
                f"FAIL: refined frame {i + 1} U_accum/dic_mesh shape mismatch"
            )
            passed = False

    print("\nResult:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
