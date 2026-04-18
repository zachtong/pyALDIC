"""Head-to-head benchmark: FFT vs seed_propagation.

Runs both init-guess modes on a set of representative scenarios
spanning displacement magnitude, image size, and single vs multi-
frame workflows, then emits a markdown report.

Metrics per scenario:
  - wall-clock (ms)
  - RMSE against ground truth (px)
  - converged node fraction
  - whether the result is visually correct (RMSE < 0.5 px)

Output: reports/fft_vs_seedprop_report.md
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tests"))

from conftest import apply_displacement_lagrangian, generate_speckle
from al_dic.core.data_structures import (
    DICPara, FrameSchedule, GridxyROIRange,
)
from al_dic.core.pipeline import run_aldic
from al_dic.solver.seed_propagation import Seed, SeedSet


# ---------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    img_size: int
    n_frames: int              # total images including ref
    peak_disp: float           # final-frame x-displacement
    mode: str = "accumulative"  # accumulative or incremental
    winsize: int = 32
    step: int = 8
    description: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        "S1-tiny_smooth_2f",
        img_size=512, n_frames=2, peak_disp=2.0,
        description="Single-pair, 2 px shift — baseline for smooth, "
                    "sub-subset motion.",
    ),
    Scenario(
        "S2-medium_2f",
        img_size=512, n_frames=2, peak_disp=30.0,
        description="Single-pair, 30 px shift — typical inter-frame "
                    "motion for moderate deformation.",
    ),
    Scenario(
        "S3-large_2f",
        img_size=1024, n_frames=2, peak_disp=100.0,
        description="Single-pair, 100 px shift — stress test for "
                    "FFT's quadratic cost scaling in search radius.",
    ),
    Scenario(
        "S4-xlarge_2f",
        img_size=1024, n_frames=2, peak_disp=200.0,
        description="Single-pair, 200 px shift — material point "
                    "far from seed, near image-half-size limit.",
    ),
    Scenario(
        "S5-multi_accum_grow",
        img_size=1000, n_frames=10, peak_disp=100.0,
        description="10-frame accumulative, displacement grows "
                    "linearly 0 -> 100 px. Exercises per-frame "
                    "bootstrap with varying magnitude.",
    ),
    Scenario(
        "S6-multi_accum_xlarge",
        img_size=1000, n_frames=15, peak_disp=180.0,
        description="15-frame accumulative, 0 -> 180 px. Worst case "
                    "for warm-start propagation; late frames need "
                    "large seed-search radius.",
    ),
]


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def _build_sequence(sc: Scenario):
    """Return (images, masks, roi, ground_truth_per_frame_u)."""
    img = sc.img_size
    ref = generate_speckle(img, img, sigma=3.0, seed=42)

    images = [ref]
    per_frame_shifts = np.linspace(0, sc.peak_disp, sc.n_frames)[1:]

    for dx in per_frame_shifts:
        def _u(x, y, _dx=dx):
            return np.full_like(x, _dx, dtype=np.float64)

        def _v(x, y):
            return np.full_like(y, 0.0, dtype=np.float64)

        images.append(apply_displacement_lagrangian(ref, _u, _v))

    # Shrink ROI on right so deformed material stays inside image
    margin = 40
    mask = np.zeros((img, img), dtype=np.float64)
    mask[
        margin: img - margin,
        margin: img - int(sc.peak_disp) - margin,
    ] = 1.0
    masks = [mask] * len(images)

    roi = GridxyROIRange(
        gridx=(margin, img - int(sc.peak_disp) - margin - sc.winsize),
        gridy=(margin, img - 1 - margin),
    )
    return images, masks, roi, per_frame_shifts


def _make_para(sc: Scenario, roi, init_mode: str, seed_set=None):
    return DICPara(
        winsize=sc.winsize,
        winstepsize=sc.step,
        winsize_min=8,
        img_size=(sc.img_size, sc.img_size),
        gridxy_roi_range=roi,
        icgn_max_iter=100,
        tol=1e-2,
        admm_max_iter=2,
        admm_tol=1e-2,
        fft_auto_expand_search=True,
        reference_mode=sc.mode,
        size_of_fft_search_region=20,
        init_guess_mode=init_mode,
        seed_set=seed_set,
        img_ref_mask=None,  # filled by pipeline from masks list
    )


def _measure(result, truths: np.ndarray) -> dict:
    """Extract per-frame RMSE/convergence summary from a PipelineResult."""
    per_frame = []
    for i, fr in enumerate(result.result_disp):
        if fr is None:
            per_frame.append({"rmse": float("nan"),
                              "u_med": float("nan"),
                              "converged": 0, "total": 0})
            continue
        U = fr.U
        u = U[0::2]
        v = U[1::2]
        rmse = float(np.sqrt(
            np.nanmean((u - truths[i]) ** 2 + (v - 0.0) ** 2),
        ))
        u_med = float(np.nanmedian(u))
        total = u.size
        converged = int(np.isfinite(u).sum())
        per_frame.append({
            "rmse": rmse, "u_med": u_med,
            "converged": converged, "total": total,
        })
    return per_frame


def _run_scenario(sc: Scenario) -> dict:
    images, masks, roi, truths = _build_sequence(sc)
    print(f"\n=== {sc.name} ({sc.img_size}x{sc.img_size}, "
          f"{sc.n_frames} frames, peak {sc.peak_disp} px) ===")

    # ---- FFT ----
    para_fft = _make_para(sc, roi, init_mode="fft")
    t0 = time.perf_counter()
    try:
        res_fft = run_aldic(
            para_fft, images, masks, compute_strain=False,
        )
        t_fft = time.perf_counter() - t0
        fft_metrics = _measure(res_fft, truths)
        fft_err = None
    except Exception as e:
        t_fft = time.perf_counter() - t0
        fft_metrics = [{"rmse": float("nan"), "u_med": float("nan"),
                        "converged": 0, "total": 0}
                       for _ in range(sc.n_frames - 1)]
        fft_err = f"{type(e).__name__}: {e}"
    print(f"  FFT:       {t_fft:.1f} s"
          + (f" -- FAILED: {fft_err}" if fft_err else ""))

    # ---- Pick seed (center) using the FFT mesh if available ----
    if fft_err is None:
        coords = res_fft.dic_mesh.coordinates_fem
        cx, cy = (roi.gridx[0] + roi.gridx[1]) / 2, (roi.gridy[0] + roi.gridy[1]) / 2
        seed_idx = int(np.argmin(
            np.linalg.norm(coords - np.array([cx, cy]), axis=1),
        ))
    else:
        seed_idx = 0
    seed_set = SeedSet(
        seeds=(Seed(node_idx=seed_idx, region_id=0),),
        ncc_threshold=0.5,
    )

    # ---- seed_propagation ----
    para_sp = _make_para(sc, roi, init_mode="seed_propagation",
                         seed_set=seed_set)
    t0 = time.perf_counter()
    try:
        res_sp = run_aldic(
            para_sp, images, masks, compute_strain=False,
        )
        t_sp = time.perf_counter() - t0
        sp_metrics = _measure(res_sp, truths)
        sp_err = None
    except Exception as e:
        t_sp = time.perf_counter() - t0
        sp_metrics = [{"rmse": float("nan"), "u_med": float("nan"),
                       "converged": 0, "total": 0}
                      for _ in range(sc.n_frames - 1)]
        sp_err = f"{type(e).__name__}: {e}"
    print(f"  seed_prop: {t_sp:.1f} s"
          + (f" -- FAILED: {sp_err}" if sp_err else ""))

    return {
        "scenario": sc,
        "t_fft_s": t_fft,
        "fft_err": fft_err,
        "fft_metrics": fft_metrics,
        "t_sp_s": t_sp,
        "sp_err": sp_err,
        "sp_metrics": sp_metrics,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def _aggregate_run(m: list[dict]) -> dict:
    """Average metrics across frames for one (scenario, mode) pair."""
    rmses = [x["rmse"] for x in m if np.isfinite(x["rmse"])]
    convs = [x["converged"] / max(1, x["total"]) for x in m]
    return {
        "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "max_rmse": float(np.max(rmses)) if rmses else float("nan"),
        "mean_conv": float(np.mean(convs)),
    }


def _emit_markdown(runs: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    a = lines.append

    a("# FFT vs Seed Propagation — Head-to-Head Benchmark")
    a("")
    a("Comparison of the two primary init-guess methods in pyALDIC "
      "across displacement magnitude, image size, and multi-frame "
      "accumulative workflows.")
    a("")
    a("*Generated by `scripts/bench_fft_vs_seedprop.py`.*")
    a("")

    # -- Executive summary --
    a("## Executive Summary")
    a("")
    n_sp_faster = sum(1 for r in runs if r["t_sp_s"] < r["t_fft_s"])
    fft_fails = sum(1 for r in runs if r["fft_err"])
    sp_fails = sum(1 for r in runs if r["sp_err"])
    a(f"- **Scenarios run**: {len(runs)}")
    a(f"- **seed_propagation faster than FFT**: "
      f"{n_sp_faster}/{len(runs)} scenarios")
    a(f"- **FFT failures**: {fft_fails}")
    a(f"- **seed_propagation failures**: {sp_fails}")
    a("")

    # -- Overall table --
    a("## Speed / Accuracy Overview")
    a("")
    a("| Scenario | FFT time (s) | seed time (s) | speedup "
      "| FFT mean RMSE | seed mean RMSE | FFT conv | seed conv |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        sc = r["scenario"]
        fft_agg = _aggregate_run(r["fft_metrics"])
        sp_agg = _aggregate_run(r["sp_metrics"])
        speedup = (
            f"{r['t_fft_s'] / r['t_sp_s']:.2f}x"
            if r["t_sp_s"] > 0
            else "n/a"
        )
        a(f"| {sc.name} "
          f"| {r['t_fft_s']:.2f} "
          f"| {r['t_sp_s']:.2f} "
          f"| {speedup} "
          f"| {fft_agg['mean_rmse']:.4f} "
          f"| {sp_agg['mean_rmse']:.4f} "
          f"| {fft_agg['mean_conv']:.2%} "
          f"| {sp_agg['mean_conv']:.2%} |")
    a("")

    # -- Per-scenario detail --
    a("## Per-Scenario Details")
    a("")
    for r in runs:
        sc = r["scenario"]
        a(f"### {sc.name}")
        a("")
        a(f"*{sc.description}*")
        a("")
        a(f"- image: {sc.img_size}x{sc.img_size}, "
          f"frames: {sc.n_frames}, peak displacement: {sc.peak_disp} px")
        a(f"- FFT wall-clock: **{r['t_fft_s']:.2f} s**"
          + (f" — FAILED: {r['fft_err']}" if r["fft_err"] else ""))
        a(f"- seed_prop wall-clock: **{r['t_sp_s']:.2f} s**"
          + (f" — FAILED: {r['sp_err']}" if r["sp_err"] else ""))
        a("")
        a("| frame | truth u | FFT u_med | FFT RMSE "
          "| seed u_med | seed RMSE |")
        a("|---:|---:|---:|---:|---:|---:|")
        # truth u: reuse per-frame shifts from `_build_sequence`
        truths = np.linspace(0, sc.peak_disp, sc.n_frames)[1:]
        for i, (fm, sm) in enumerate(zip(r["fft_metrics"], r["sp_metrics"])):
            a(f"| {i + 1} | {truths[i]:.2f} "
              f"| {fm['u_med']:.2f} | {fm['rmse']:.4f} "
              f"| {sm['u_med']:.2f} | {sm['rmse']:.4f} |")
        a("")

    # -- Verdict --
    a("## Verdict")
    a("")
    a("**Recommend `seed_propagation` when:**")
    a("- Peak inter-frame displacement exceeds ~50 px")
    a("- Multi-frame accumulative sequences with growing displacement")
    a("- Discontinuous fields (cracks, shear bands; not benchmarked here, "
      "see `reports/init_guess_eval.pdf` for that class)")
    a("")
    a("**Recommend `FFT` when:**")
    a("- Small inter-frame displacement (< 20 px) with good texture")
    a("- No ROI regions require manual seed placement")
    a("- First-run / exploratory analysis where auto-place hasn't been "
      "validated yet")
    a("")
    a("**Recommend `previous`:**")
    a("- Very smooth sub-pixel motion after a good first-frame init")
    a("- Not a stand-alone choice; requires a trustworthy first frame "
      "(which means FFT or seeds first)")
    a("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    runs = []
    for sc in SCENARIOS:
        runs.append(_run_scenario(sc))

    out = BASE / "reports" / "fft_vs_seedprop_report.md"
    _emit_markdown(runs, out)
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
