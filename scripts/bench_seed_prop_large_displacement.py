"""End-to-end pipeline benchmark: fft vs seed_propagation on 100 px motion.

Validates the ~20x whole-frame speedup claim from project memory
'project_seed_propagation_design.md'. Scratch benchmark
(tmp_bench_large_displacement.py) measured only the init-guess stage;
this runs the full run_aldic pipeline on both configs.

Run:
    python scripts/bench_seed_prop_large_displacement.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tests"))

from conftest import apply_displacement_lagrangian, generate_speckle  # noqa: E402

from al_dic.core.data_structures import DICPara, GridxyROIRange  # noqa: E402
from al_dic.core.pipeline import run_aldic  # noqa: E402
from al_dic.solver.seed_propagation import Seed, SeedSet  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario setup: 512x512 speckle, 100 px rigid translation
# ---------------------------------------------------------------------------

IMG = 512
WINSIZE = 32
STEP = 8
MARGIN = 24
DX, DY = 100.0, 0.0
N_RUNS = 3

print("=" * 72)
print(f"Pipeline benchmark: fft vs seed_propagation")
print(f"  Image: {IMG}x{IMG} speckle, winsize={WINSIZE}, step={STEP}")
print(f"  Motion: (dx, dy) = ({DX}, {DY}) px (Lagrangian warp)")
print("=" * 72)

ref = generate_speckle(IMG, IMG, sigma=3.0, seed=42)


def u_func(x, y):
    return np.full_like(x, DX, dtype=np.float64)


def v_func(x, y):
    return np.full_like(y, DY, dtype=np.float64)


deformed = apply_displacement_lagrangian(ref, u_func, v_func)

# Shrink ROI from the right where material warps out of frame.
mask = np.zeros((IMG, IMG), dtype=np.float64)
mask[MARGIN : IMG - MARGIN, MARGIN : IMG - int(abs(DX)) - MARGIN] = 1.0

roi = GridxyROIRange(
    gridx=(MARGIN, IMG - int(abs(DX)) - MARGIN - WINSIZE),
    gridy=(MARGIN, IMG - 1 - MARGIN),
)

para_base = DICPara(
    winsize=WINSIZE,
    winstepsize=STEP,
    winsize_min=8,
    img_size=(IMG, IMG),
    gridxy_roi_range=roi,
    icgn_max_iter=50,
    tol=1e-2,
    admm_max_iter=2,
    admm_tol=1e-2,
    fft_auto_expand_search=True,
    reference_mode="accumulative",
    size_of_fft_search_region=20,
    img_ref_mask=mask,
)


def _rmse_vs_truth(U: np.ndarray) -> float:
    u_vals = U[0::2]
    v_vals = U[1::2]
    return float(
        np.sqrt(np.nanmean((u_vals - DX) ** 2 + (v_vals - DY) ** 2))
    )


def _n_converged(U: np.ndarray) -> int:
    finite = np.isfinite(U[0::2]) & np.isfinite(U[1::2])
    return int(finite.sum())


def _run_with_timing(para, label):
    print(f"\n--- {label} ---")
    # Warmup for Numba JIT
    _ = run_aldic(para, [ref, deformed], [mask, mask], compute_strain=False)
    times = []
    last_result = None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        last_result = run_aldic(
            para, [ref, deformed], [mask, mask], compute_strain=False,
        )
        times.append(time.perf_counter() - t0)
    t_min = min(times) * 1000.0
    t_mean = (sum(times) / len(times)) * 1000.0
    U = last_result.result_disp[0].U
    rmse = _rmse_vs_truth(U)
    n_ok = _n_converged(U)
    n_total = U.size // 2
    print(f"  run_aldic wall-clock: {t_min:.0f} ms (min) / {t_mean:.0f} ms (mean)")
    print(f"  converged nodes: {n_ok}/{n_total} ({100 * n_ok / n_total:.1f}%)")
    print(f"  RMSE vs ground truth: {rmse:.3f} px")
    return t_min, rmse, last_result


# ---------------------------------------------------------------------------
# Config A: FFT with auto-expand (user's typical config — search=20)
# ---------------------------------------------------------------------------

para_a = replace(para_base, init_guess_mode="fft")
t_fft, rmse_fft, result_a = _run_with_timing(
    para_a, "A: init_guess_mode='fft', search=20 + auto-expand",
)


# ---------------------------------------------------------------------------
# Config B: seed_propagation with one seed at ROI center
# ---------------------------------------------------------------------------

coords = result_a.dic_mesh.coordinates_fem
roi_cx = (roi.gridx[0] + roi.gridx[1]) / 2
roi_cy = (roi.gridy[0] + roi.gridy[1]) / 2
seed_idx = int(
    np.argmin(
        np.linalg.norm(coords - np.array([roi_cx, roi_cy]), axis=1),
    ),
)

seed_set = SeedSet(
    seeds=(Seed(node_idx=seed_idx, region_id=0),),
    ncc_threshold=0.5,
)
para_b = replace(
    para_base,
    init_guess_mode="seed_propagation",
    seed_set=seed_set,
    size_of_fft_search_region=120,
)
t_sp, rmse_sp, result_b = _run_with_timing(
    para_b,
    f"B: init_guess_mode='seed_propagation', seed at node {seed_idx}, "
    f"search=120",
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(
    f"{'config':<40} {'time (ms)':>12} {'RMSE (px)':>12}"
)
print(
    f"{'fft + auto-expand, search=20':<40} {t_fft:>12.0f} {rmse_fft:>12.3f}"
)
print(
    f"{'seed_propagation, 1 seed, search=120':<40} {t_sp:>12.0f} {rmse_sp:>12.3f}"
)
speedup = t_fft / t_sp if t_sp > 0 else float("nan")
print(f"\nwall-clock speedup = fft / seed_prop = {speedup:.1f}x")
print(
    f"memory claim ~20x {'[CONFIRMED]' if speedup >= 10.0 else '[WEAKER THAN CLAIMED]' if speedup >= 3.0 else '[NOT OBSERVED]'}"
)
print("=" * 72)
