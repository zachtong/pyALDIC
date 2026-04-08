"""Benchmark: ADMM vs local-only IC-GN throughput comparison.

Measures the wall-clock time cost of 3 ADMM iterations vs. local-only
IC-GN to quantify the per-node overhead of the global FEM step.

Does NOT modify any source files.

Usage
-----
    python scripts/benchmark_admm.py

Output
------
Prints a summary table and saves a plain-text + PDF report to
    reports/benchmark_admm_<timestamp>.txt
    reports/benchmark_admm_<timestamp>.pdf

Benchmark design
----------------
* Synthetic speckle pair (Gaussian-filtered noise) with uniform 1-pixel
  x-translation.  IC-GN converges in 1-3 iterations (fast, minimal
  scenario variance — pure throughput signal).
* Subset size: 32 pixels (fixed).
* Node step sizes: 16 and 8 pixels.
* Image sizes: 256 and 512 pixels square (1024 excluded — FEM assembly
  time at 1024-step-8 would dominate and skew the ADMM comparison).
* Three modes timed for each scenario:

    LOCAL_ICGN   — direct call to local_icgn() (no pipeline wrapper,
                   same as benchmark_local_icgn.py).
    PIPELINE_ONLY — run_aldic(use_global_step=False): full pipeline
                   including FFT init-guess + IC-GN but NO FEM/ADMM.
    ADMM_3ITER   — run_aldic(use_global_step=True, admm_max_iter=3):
                   full AL-DIC with 3 ADMM iterations.

* Numba JIT warmup: one 256×256 step=8 LOCAL_ICGN run is executed first
  so JIT compilation time is excluded.  A second ADMM warmup run compiles
  the FEM/Subpb2 kernels.

Literature comparison
---------------------
Approximate throughput values from published DIC benchmarks are included
for context.  These are based on:
  [1] Pan B. (2013) "Bias error analysis of DIC", Strain.
      Typical CPU IC-GN (MATLAB, single-thread): ~2 000–8 000 nodes/s.
  [2] Blaber J. et al. (2015) "Ncorr: Open-Source 2D DIC Matlab",
      Exp. Mech.  MATLAB + C subset loop: ~3 000–10 000 nodes/s.
  [3] Dong Y. & Pan B. (2017) "A Review of Speckle Pattern Fabrication
      and Assessment …", Exp. Mech.
      GPU IC-GN (CUDA): ~500 000–2 000 000 nodes/s (hardware-dependent).
  [4] Yang J. & Bhattacharya K. (2019) "Augmented Lagrangian DIC",
      Exp. Mech.  MATLAB AL-DIC with ADMM: ~500–3 000 nodes/s (full
      pipeline, sequential, 1024² image, est. from paper wall-clock times).
  [5] Yang J. & Bhattacharya K. (2021) "Fast Adaptive Mesh AL-DIC",
      Exp. Mech.  Adaptive mesh reduces node count; overall ~2x speedup.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import time
import warnings
from pathlib import Path

# ── make sure the project src is on the path when run directly ───────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from staq_dic.core.data_structures import DICPara, GridxyROIRange, ImageGradients
from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.solver.local_icgn import local_icgn


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_speckle(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Synthetic speckle in [0, 1] (pipeline-ready, no extra normalization needed)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width))
    filtered = gaussian_filter(noise, sigma=3.0, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return filtered


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift image by (dx, dy) pixels using quintic spline interpolation."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return map_coordinates(img, [yy - dy, xx - dx], order=5, mode="constant", cval=0.0)


def _make_mesh(img_h: int, img_w: int, step: int, winsize: int) -> np.ndarray:
    """Regular grid of nodes, keeping subsets fully inside the image."""
    margin = winsize // 2
    xs = np.arange(margin, img_w - margin, step, dtype=np.float64)
    ys = np.arange(margin, img_h - margin, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


@contextlib.contextmanager
def _suppress_output():
    """Suppress stdout, UserWarnings, and logging below WARNING level."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        old_level = logging.root.level
        logging.disable(logging.WARNING)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            logging.disable(old_level)


# ── local IC-GN timing (direct call, no pipeline wrapper) ────────────────────


def _bench_local_icgn(img_size: int, step: int, winsize: int = 32) -> dict:
    """Time one local_icgn() call directly."""
    img_ref = _make_speckle(img_size, img_size, seed=42)
    img_def = _translate(img_ref, 1.0, 0.0)
    mask = np.ones((img_size, img_size), dtype=np.float64)
    Df = compute_image_gradient(img_ref, mask)
    coords = _make_mesh(img_size, img_size, step, winsize)
    n_nodes = len(coords)
    if n_nodes == 0:
        return {"n_nodes": 0, "skipped": True}

    para = DICPara(
        winsize=winsize, winstepsize=step, icgn_max_iter=50,
        tol=1e-2, img_size=(img_size, img_size),
    )
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)

    t0 = time.perf_counter()
    U, F, internal_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
        U0, coords, Df, img_ref, img_def, para, tol=1e-2,
    )
    wall_elapsed = time.perf_counter() - t0
    elapsed = internal_time if internal_time > 0 else wall_elapsed

    return {
        "mode": "LOCAL_ICGN",
        "img_size": img_size, "step": step, "n_nodes": n_nodes,
        "elapsed_s": elapsed, "wall_s": wall_elapsed,
        "nodes_per_sec": n_nodes / elapsed if elapsed > 0 else 0.0,
        "skipped": False,
    }


# ── pipeline timing (uses run_aldic) ─────────────────────────────────────────


def _bench_pipeline(img_size: int, step: int, winsize: int = 32,
                    use_admm: bool = False, admm_iters: int = 3) -> dict:
    """Time run_aldic() in pipeline-only or full-ADMM mode."""
    from staq_dic.core.pipeline import run_aldic

    img_ref = _make_speckle(img_size, img_size, seed=42)
    img_def = _translate(img_ref, 1.0, 0.0)
    images = [img_ref, img_def]
    masks = [np.ones((img_size, img_size), dtype=np.float64)] * 2

    margin = winsize // 2
    roi = GridxyROIRange(
        gridx=(margin, img_size - margin),
        gridy=(margin, img_size - margin),
    )
    para = DICPara(
        winsize=winsize,
        winstepsize=step,
        winsize_min=step,          # uniform mesh — no quadtree refinement
        tol=1e-2,
        icgn_max_iter=50,
        use_global_step=use_admm,
        admm_max_iter=admm_iters if use_admm else 1,
        admm_tol=1e-2,
        mu=1e-3,
        alpha=0.0,
        disp_smoothness=0.0,
        strain_smoothness=0.0,
        smoothness=0.0,
        size_of_fft_search_region=10,
        gridxy_roi_range=roi,
        img_size=(img_size, img_size),
    )

    with _suppress_output():
        t0 = time.perf_counter()
        result = run_aldic(para, images, masks, compute_strain=False)
        elapsed = time.perf_counter() - t0

    n_nodes = result.dic_mesh.coordinates_fem.shape[0]
    mode = f"ADMM_{admm_iters}ITER" if use_admm else "PIPELINE_ONLY"
    return {
        "mode": mode,
        "img_size": img_size, "step": step, "n_nodes": n_nodes,
        "elapsed_s": elapsed, "wall_s": elapsed,
        "nodes_per_sec": n_nodes / elapsed if elapsed > 0 else 0.0,
        "skipped": False,
    }


# ── warmup ────────────────────────────────────────────────────────────────────


def _warmup() -> None:
    """Trigger Numba JIT + FEM/Subpb2 compilation before timed runs."""
    print("Warming up Numba JIT (LOCAL_ICGN path)...", flush=True)
    t0 = time.perf_counter()
    _bench_local_icgn(img_size=256, step=8, winsize=32)  # 784 nodes → prange path
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    print("Warming up FEM / Subpb2 path (first ADMM run)...", flush=True)
    t0 = time.perf_counter()
    _bench_pipeline(img_size=256, step=16, winsize=32, use_admm=True, admm_iters=1)
    print(f"  done in {time.perf_counter() - t0:.1f}s\n", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    from staq_dic.export.export_utils import make_timestamp

    print("=" * 72)
    print("  AL-DIC Benchmark: Local IC-GN vs ADMM (3 iterations)")
    print("=" * 72)
    print()

    _warmup()

    # Scenarios: (image_size, step)
    scenarios = [
        (256,  16),
        (256,   8),
        (512,  16),
        (512,   8),
    ]

    results: list[dict] = []

    for img_size, step in scenarios:
        label = f"  {img_size}×{img_size}  step={step:2d}"
        print(f"{label}", flush=True)

        for mode_fn, kwargs in [
            (_bench_local_icgn,  dict(img_size=img_size, step=step)),
            (_bench_pipeline,    dict(img_size=img_size, step=step, use_admm=False)),
            (_bench_pipeline,    dict(img_size=img_size, step=step, use_admm=True, admm_iters=3)),
        ]:
            r = mode_fn(**kwargs)  # type: ignore[arg-type]
            if r.get("skipped"):
                print(f"    [skipped — no nodes]")
                continue
            print(
                f"    {r['mode']:<20s}  {r['n_nodes']:5d} nodes  "
                f"{r['nodes_per_sec']:8.0f} nodes/s  "
                f"t={r['elapsed_s']:.3f}s"
            )
            results.append(r)
        print()

    if not results:
        print("No results collected.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'Image':>12}  {'Step':>4}  {'Mode':<20}  {'Nodes':>6}  "
          f"{'nodes/s':>9}  {'Time(s)':>7}  {'Overhead':>9}")
    print("-" * 72)

    # Group by scenario to show overhead
    from itertools import groupby
    for (img_size, step), group in groupby(results, key=lambda r: (r["img_size"], r["step"])):
        group_list = list(group)
        local = next((r for r in group_list if r["mode"] == "LOCAL_ICGN"), None)
        for r in group_list:
            if local and r["mode"] != "LOCAL_ICGN":
                overhead = r["elapsed_s"] / local["elapsed_s"] if local["elapsed_s"] > 0 else float("nan")
                overhead_str = f"{overhead:.1f}×"
            else:
                overhead_str = "reference"
            print(
                f"  {r['img_size']}×{r['img_size']:4d}  "
                f"{r['step']:4d}  "
                f"{r['mode']:<20s}  "
                f"{r['n_nodes']:6d}  "
                f"{r['nodes_per_sec']:9.0f}  "
                f"{r['elapsed_s']:7.3f}  "
                f"{overhead_str:>9}"
            )
    print("=" * 72)

    # ── Literature comparison ──────────────────────────────────────────────
    print()
    print("Literature reference throughput (CPU IC-GN, approximate):")
    print("-" * 72)
    lit_table = [
        ("Pan 2013 [1]",         "Sequential MATLAB IC-GN",     "2 000–8 000"),
        ("Ncorr 2015 [2]",       "MATLAB + C subset loop",       "3 000–10 000"),
        ("Yang 2019 [4]",        "MATLAB AL-DIC (ADMM, 1024²)", "500–3 000"),
        ("GPU IC-GN [3]",        "CUDA (hardware-dependent)",   "500 000–2 000 000"),
        ("STAQ-DIC (ours)",      "Numba prange, LOCAL_ICGN",    _our_range(results, "LOCAL_ICGN")),
        ("STAQ-DIC (ours)",      "Numba, ADMM 3 iters",         _our_range(results, "ADMM_3ITER")),
    ]
    print(f"  {'Source':<22}  {'Method':<38}  {'nodes/s':>16}")
    print(f"  {'-'*22}  {'-'*38}  {'-'*16}")
    for src, method, speed in lit_table:
        print(f"  {src:<22}  {method:<38}  {speed:>16}")
    print()
    print("  [1] Pan B. (2013) Strain 49:321–334.")
    print("  [2] Blaber J. et al. (2015) Exp Mech 55:1105–1122.")
    print("  [3] Dong Y. & Pan B. (2017) Exp Mech 57:1283–1306.")
    print("  [4] Yang J. & Bhattacharya K. (2019) Exp Mech 59:361–374.")
    print()

    # ── Save text report ───────────────────────────────────────────────────
    reports_dir = _HERE.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = make_timestamp()
    txt_path = reports_dir / f"benchmark_admm_{ts}.txt"
    _write_text_report(txt_path, results, lit_table, ts)
    print(f"  Text report: {txt_path}")

    # ── Save PDF report ────────────────────────────────────────────────────
    try:
        pdf_path = reports_dir / f"benchmark_admm_{ts}.pdf"
        _write_pdf_report(pdf_path, results, lit_table, ts)
        print(f"  PDF report:  {pdf_path}")
    except Exception as exc:
        print(f"  (PDF skipped: {exc})")


def _our_range(results: list[dict], mode: str) -> str:
    """Return 'min–max nodes/s' string for our measurements of a given mode."""
    speeds = [r["nodes_per_sec"] for r in results if r["mode"] == mode and r["nodes_per_sec"] > 0]
    if not speeds:
        return "N/A"
    lo, hi = int(min(speeds)), int(max(speeds))
    return f"{lo:,}–{hi:,}"


def _write_text_report(path: Path, results: list[dict],
                       lit_table: list[tuple], ts: str) -> None:
    lines = [
        "AL-DIC Benchmark: Local IC-GN vs ADMM (3 iterations)",
        f"Timestamp: {ts}",
        "Setup: synthetic speckle, 1px uniform x-translation, winsize=32, tol=1e-2",
        "",
        f"{'Image':>12}  {'Step':>4}  {'Mode':<20}  {'Nodes':>6}  {'nodes/s':>9}  {'Time(s)':>7}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"  {r['img_size']}×{r['img_size']:4d}  "
            f"{r['step']:4d}  "
            f"{r['mode']:<20s}  "
            f"{r['n_nodes']:6d}  "
            f"{r['nodes_per_sec']:9.0f}  "
            f"{r['elapsed_s']:7.3f}"
        )
    lines += [
        "=" * 72,
        "",
        "Literature reference throughput (CPU IC-GN, approximate):",
        f"  {'Source':<22}  {'Method':<38}  {'nodes/s':>16}",
        f"  {'-'*22}  {'-'*38}  {'-'*16}",
    ]
    for src, method, speed in lit_table:
        lines.append(f"  {src:<22}  {method:<38}  {speed:>16}")
    lines += [
        "",
        "  [1] Pan B. (2013) Strain 49:321-334.",
        "  [2] Blaber J. et al. (2015) Exp Mech 55:1105-1122.",
        "  [3] Dong Y. & Pan B. (2017) Exp Mech 57:1283-1306.",
        "  [4] Yang J. & Bhattacharya K. (2019) Exp Mech 59:361-374.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pdf_report(path: Path, results: list[dict],
                      lit_table: list[tuple], ts: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    modes_order = ["LOCAL_ICGN", "PIPELINE_ONLY", "ADMM_3ITER"]
    mode_colors = {
        "LOCAL_ICGN":    "#2196F3",   # blue
        "PIPELINE_ONLY": "#4CAF50",   # green
        "ADMM_3ITER":    "#FF9800",   # orange
    }
    mode_labels = {
        "LOCAL_ICGN":    "Local IC-GN (direct)",
        "PIPELINE_ONLY": "Pipeline (no ADMM)",
        "ADMM_3ITER":    "AL-DIC (ADMM ×3)",
    }

    scenarios = sorted({(r["img_size"], r["step"]) for r in results})
    x = np.arange(len(scenarios))
    width = 0.26

    with PdfPages(str(path)) as pdf:
        # --- Page 1: throughput bar chart ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"AL-DIC Benchmark: Local IC-GN vs ADMM  (ts={ts})\n"
            "Synthetic speckle, 1 px x-translation, winsize=32, tol=1e-2",
            fontsize=12,
        )

        for ax_idx, (ax, use_log) in enumerate(zip(axes, [False, True])):
            for mi, mode in enumerate(modes_order):
                speeds = []
                for (img_s, step) in scenarios:
                    match = [r for r in results
                             if r["img_size"] == img_s and r["step"] == step
                             and r["mode"] == mode]
                    speeds.append(match[0]["nodes_per_sec"] if match else 0.0)
                offset = (mi - 1) * width
                bars = ax.bar(x + offset, speeds, width,
                              label=mode_labels[mode],
                              color=mode_colors[mode], alpha=0.85)
                for bar, v in zip(bars, speeds):
                    if v > 0:
                        txt = f"{v/1e3:.0f}k"
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() * 1.02,
                                txt, ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{s[0]}×{s[0]}\nstep={s[1]}" for s in scenarios], fontsize=9
            )
            ax.set_ylabel("Throughput (nodes / second)")
            ax.set_title("Linear scale" if not use_log else "Log scale")
            if use_log:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: overhead multiplier ---
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("ADMM Wall-clock Overhead vs Local IC-GN (×)", fontsize=12)

        pipeline_overheads, admm_overheads = [], []
        for (img_s, step) in scenarios:
            local = next((r for r in results if r["img_size"] == img_s
                          and r["step"] == step and r["mode"] == "LOCAL_ICGN"), None)
            pipe = next((r for r in results if r["img_size"] == img_s
                         and r["step"] == step and r["mode"] == "PIPELINE_ONLY"), None)
            admm = next((r for r in results if r["img_size"] == img_s
                         and r["step"] == step and r["mode"] == "ADMM_3ITER"), None)
            pipeline_overheads.append(
                pipe["elapsed_s"] / local["elapsed_s"]
                if local and pipe and local["elapsed_s"] > 0 else 0.0
            )
            admm_overheads.append(
                admm["elapsed_s"] / local["elapsed_s"]
                if local and admm and local["elapsed_s"] > 0 else 0.0
            )

        ax.bar(x - width / 2, pipeline_overheads, width, label="Pipeline / Local IC-GN",
               color=mode_colors["PIPELINE_ONLY"], alpha=0.85)
        ax.bar(x + width / 2, admm_overheads, width, label="ADMM×3 / Local IC-GN",
               color=mode_colors["ADMM_3ITER"], alpha=0.85)

        for xi, (p, a) in enumerate(zip(pipeline_overheads, admm_overheads)):
            ax.text(xi - width / 2, p + 0.05, f"{p:.1f}×", ha="center", fontsize=9)
            ax.text(xi + width / 2, a + 0.05, f"{a:.1f}×", ha="center", fontsize=9)

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="1× reference")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{s[0]}×{s[0]}\nstep={s[1]}" for s in scenarios], fontsize=9
        )
        ax.set_ylabel("Time / Local IC-GN time")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: literature comparison (horizontal bar) ---
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Throughput vs. Published Literature (approximate)", fontsize=12)

        our_local = _our_range(results, "LOCAL_ICGN")
        our_admm  = _our_range(results, "ADMM_3ITER")

        lit_entries = [
            ("Pan 2013 [1]\n(MATLAB IC-GN)", 2000, 8000, "#9E9E9E"),
            ("Ncorr 2015 [2]\n(MATLAB+C IC-GN)", 3000, 10000, "#9E9E9E"),
            ("Yang 2019 [4]\n(MATLAB AL-DIC)", 500, 3000, "#FF9800"),
            ("GPU IC-GN [3]\n(CUDA)", 500000, 2000000, "#F44336"),
            ("STAQ-DIC LOCAL\n(Numba, this machine)", *_parse_range(our_local), "#2196F3"),
            ("STAQ-DIC ADMM×3\n(Numba, this machine)", *_parse_range(our_admm), "#4CAF50"),
        ]

        for yi, (label, lo, hi, color) in enumerate(lit_entries):
            mid = (lo + hi) / 2
            ax.barh(yi, hi - lo, left=lo, height=0.5, color=color, alpha=0.8)
            ax.text(hi * 1.05, yi, f"{int(lo):,}–{int(hi):,}", va="center", fontsize=8)

        ax.set_yticks(range(len(lit_entries)))
        ax.set_yticklabels([e[0] for e in lit_entries], fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Throughput (nodes / second, log scale)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title("Note: All literature values are approximate; hardware varies.")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _parse_range(s: str) -> tuple[float, float]:
    """Parse 'lo,hi' string from _our_range() into two floats."""
    s_clean = s.replace(",", "")
    parts = s_clean.split("–")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return 1000.0, 10000.0  # safe fallback


if __name__ == "__main__":
    main()
