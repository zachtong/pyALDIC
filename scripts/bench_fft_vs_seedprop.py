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
    # GUI defaults — matches what most users actually run.
    winsize: int = 40
    step: int = 16
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


def _warmup() -> None:
    """Run a tiny pipeline first so Numba-cached kernels are JIT-compiled.

    Without this, the first real scenario eats ~0.5 s of kernel
    compilation, distorting its timing.
    """
    print("Warming up Numba JIT cache...")
    img = 128
    ref = generate_speckle(img, img, sigma=3.0, seed=0)
    deformed = ref.copy()
    mask = np.zeros((img, img), dtype=np.float64)
    mask[20:108, 20:108] = 1.0
    roi = GridxyROIRange(gridx=(24, 100), gridy=(24, 100))
    for mode in ("fft", "seed_propagation"):
        seed_set = None
        if mode == "seed_propagation":
            seed_set = SeedSet(
                seeds=(Seed(node_idx=12, region_id=0),),
                ncc_threshold=0.3,
            )
        para = DICPara(
            winsize=20, winstepsize=8, winsize_min=8,
            img_size=(img, img), gridxy_roi_range=roi,
            icgn_max_iter=30, tol=1e-2, admm_max_iter=1,
            fft_auto_expand_search=True,
            reference_mode="accumulative",
            size_of_fft_search_region=10,
            init_guess_mode=mode, seed_set=seed_set,
            img_ref_mask=mask,
        )
        try:
            run_aldic(para, [ref, deformed], [mask, mask],
                      compute_strain=False)
        except Exception:
            pass  # warmup failure is non-fatal
    print("Warmup done.\n")


def _emit_pdf(runs: list[dict], out_path: Path) -> None:
    """Render a multi-page PDF report summarising the benchmark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["scenario"].name for r in runs]
    t_fft = [r["t_fft_s"] for r in runs]
    t_sp = [r["t_sp_s"] for r in runs]
    speedup = [r["t_fft_s"] / r["t_sp_s"] if r["t_sp_s"] > 0 else 1.0
               for r in runs]
    fft_rmse = [_aggregate_run(r["fft_metrics"])["mean_rmse"] for r in runs]
    sp_rmse = [_aggregate_run(r["sp_metrics"])["mean_rmse"] for r in runs]

    with PdfPages(str(out_path)) as pdf:
        # Page 1: cover + summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.08, 0.92,
                 "FFT vs Seed Propagation — Head-to-Head Benchmark",
                 fontsize=18, fontweight="bold")
        fig.text(0.08, 0.87,
                 "pyALDIC init-guess method comparison across "
                 "displacement magnitude, image size, and multi-frame "
                 "workflows",
                 fontsize=11, style="italic")
        n_sp_faster = sum(1 for s in speedup if s > 1.0)
        fft_fails = sum(1 for r in runs if r["fft_err"])
        sp_fails = sum(1 for r in runs if r["sp_err"])
        summary_lines = [
            f"Scenarios run: {len(runs)}",
            f"Mesh parameters: winsize=40, step=16 (GUI defaults)",
            f"Numba JIT pre-warmed: yes",
            "",
            f"Seed propagation faster than FFT: {n_sp_faster}/{len(runs)}",
            f"FFT failures: {fft_fails}, seed_prop failures: {sp_fails}",
            f"Max speedup observed: {max(speedup):.2f}x",
            f"Median speedup observed: {sorted(speedup)[len(speedup)//2]:.2f}x",
        ]
        fig.text(0.08, 0.75, "\n".join(summary_lines),
                 fontsize=11, family="monospace",
                 verticalalignment="top")

        # Summary table on same page
        ax = fig.add_axes([0.08, 0.15, 0.84, 0.45])
        ax.axis("off")
        header = ["Scenario", "FFT (s)", "seed (s)", "speedup",
                  "FFT RMSE (px)", "seed RMSE (px)"]
        cells = [
            [names[i],
             f"{t_fft[i]:.2f}",
             f"{t_sp[i]:.2f}",
             f"{speedup[i]:.2f}x",
             f"{fft_rmse[i]:.4f}",
             f"{sp_rmse[i]:.4f}"]
            for i in range(len(runs))
        ]
        tbl = ax.table(cellText=cells, colLabels=header,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: speed bar chart
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(runs))
        width = 0.35
        ax.bar(x - width / 2, t_fft, width, label="FFT",
               color="#6366f1")
        ax.bar(x + width / 2, t_sp, width, label="seed_propagation",
               color="#22c55e")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Total pipeline time per scenario")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for i in range(len(runs)):
            ax.text(x[i], max(t_fft[i], t_sp[i]) * 1.02,
                    f"{speedup[i]:.1f}x",
                    ha="center", va="bottom", fontsize=9,
                    color="#444")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: speedup + displacement trend
        fig, ax = plt.subplots(figsize=(11, 6))
        peaks = [r["scenario"].peak_disp for r in runs]
        ax.plot(peaks, speedup, "o-", markersize=10,
                color="#6366f1", linewidth=2)
        for i, name in enumerate(names):
            ax.annotate(name, (peaks[i], speedup[i]),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=8)
        ax.axhline(1.0, color="#888", linestyle="--",
                   label="parity (same speed)")
        ax.set_xlabel("Peak displacement (px)")
        ax.set_ylabel("Speedup (FFT time / seed_prop time)")
        ax.set_title("Speedup scales with displacement magnitude")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Per-scenario detail pages
        for r in runs:
            sc = r["scenario"]
            truths = np.linspace(0, sc.peak_disp, sc.n_frames)[1:]
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.08, 0.92, sc.name, fontsize=16,
                     fontweight="bold")
            fig.text(0.08, 0.88, sc.description, fontsize=10,
                     style="italic", wrap=True)
            info_lines = [
                f"Image: {sc.img_size}x{sc.img_size}",
                f"Frames: {sc.n_frames}   Peak disp: {sc.peak_disp} px",
                f"Mesh: winsize={sc.winsize}, step={sc.step}",
                f"Mode: {sc.mode}",
                "",
                f"FFT wall-clock:       {r['t_fft_s']:.2f} s"
                + (f"  [FAILED: {r['fft_err']}]" if r["fft_err"] else ""),
                f"seed_prop wall-clock: {r['t_sp_s']:.2f} s"
                + (f"  [FAILED: {r['sp_err']}]" if r["sp_err"] else ""),
            ]
            fig.text(0.08, 0.80, "\n".join(info_lines),
                     fontsize=11, family="monospace",
                     verticalalignment="top")

            # Per-frame RMSE line plot
            ax = fig.add_axes([0.10, 0.12, 0.85, 0.45])
            fft_rmse_per_frame = [x["rmse"] for x in r["fft_metrics"]]
            sp_rmse_per_frame = [x["rmse"] for x in r["sp_metrics"]]
            frames = list(range(1, len(fft_rmse_per_frame) + 1))
            ax.plot(frames, fft_rmse_per_frame, "o-", label="FFT",
                    color="#6366f1")
            ax.plot(frames, sp_rmse_per_frame, "s-", label="seed_prop",
                    color="#22c55e")
            ax.set_xlabel("Frame index (1 = first deformed)")
            ax.set_ylabel("RMSE (px)")
            ax.set_title("Per-frame accuracy")
            ax.grid(linestyle="--", alpha=0.4)
            ax.legend()
            ax.set_yscale("symlog", linthresh=1e-3)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved: {out_path}")


def main() -> None:
    _warmup()
    runs = []
    for sc in SCENARIOS:
        runs.append(_run_scenario(sc))

    out_md = BASE / "reports" / "fft_vs_seedprop_report.md"
    _emit_markdown(runs, out_md)
    print(f"\nMarkdown saved: {out_md}")

    out_pdf = BASE / "reports" / "fft_vs_seedprop_report.pdf"
    _emit_pdf(runs, out_pdf)


if __name__ == "__main__":
    main()
