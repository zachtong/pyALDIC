"""Benchmark: local IC-GN solver throughput (nodes / second).

Measures pure local IC-GN performance -- no ADMM, no global FEM step.
Does NOT modify any source files.

Usage
-----
    python scripts/benchmark_local_icgn.py

Output
------
Prints a summary table and saves a plain-text report to
    reports/benchmark_local_icgn_<timestamp>.txt

Benchmark design
----------------
* Synthetic speckle pair (Gaussian-filtered noise) with uniform 1-pixel
  x-translation.  IC-GN converges in 1-3 iterations -- negligible scenario
  variance, pure throughput signal.
* Image sizes: 256, 512, 1024 pixels square.
* Node step sizes: 16, 8, 4 pixels  (= 2^4, 2^3, 2^2 -- power-of-2 required).
* Subset size: 32 pixels (fixed, typical real-world value).
* Numba JIT warmup: one small (64x64, step=16) run is executed first so that
  compilation time is not included in any reported measurement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── make sure the project src is on the path when run directly ───────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from staq_dic.core.data_structures import DICPara, ImageGradients
from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.solver.local_icgn import local_icgn


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_speckle(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Synthetic speckle in [20, 235]."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width))
    filtered = gaussian_filter(noise, sigma=3.0, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift image by (dx, dy) pixels using quintic spline interpolation."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return map_coordinates(
        img, [yy - dy, xx - dx], order=5, mode="constant", cval=0.0
    )


def _make_mesh(img_h: int, img_w: int, step: int, winsize: int) -> np.ndarray:
    """Regular grid of nodes, keeping subsets fully inside the image."""
    margin = winsize // 2
    xs = np.arange(margin, img_w - margin, step, dtype=np.float64)
    ys = np.arange(margin, img_h - margin, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _run_benchmark(
    img_size: int,
    step: int,
    winsize: int = 32,
    shift_x: float = 1.0,
    shift_y: float = 0.0,
    tol: float = 1e-2,
) -> dict:
    """Run one benchmark scenario and return metrics."""
    # --- build image pair ---
    img_ref = _make_speckle(img_size, img_size, seed=42)
    img_def = _translate(img_ref, shift_x, shift_y)

    # --- gradient + mask ---
    mask = np.ones((img_size, img_size), dtype=np.float64)
    Df = compute_image_gradient(img_ref, mask)

    # --- mesh ---
    coords = _make_mesh(img_size, img_size, step, winsize)
    n_nodes = len(coords)
    if n_nodes == 0:
        return {"n_nodes": 0, "skipped": True}

    # --- zero initial displacement ---
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)

    # --- DIC parameters ---
    para = DICPara(
        winsize=winsize,
        winstepsize=step,
        icgn_max_iter=50,
        tol=tol,
        img_size=(img_size, img_size),
    )

    # --- time the call ---
    wall_t0 = time.perf_counter()
    U, F, internal_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
        U0, coords, Df, img_ref, img_def, para, tol=tol,
    )
    wall_elapsed = time.perf_counter() - wall_t0

    # Use internal_time (measured inside the solver, excludes Python overhead)
    elapsed = internal_time if internal_time > 0 else wall_elapsed

    good_nodes = n_nodes - int(bad_pt_num) - int(mark_hole.sum())
    avg_iters = float(np.mean(conv_iter[conv_iter > 0])) if (conv_iter > 0).any() else 0.0

    # Displacement error vs ground truth
    u_out = U[0::2]
    u_gt = np.full(n_nodes, shift_x)
    valid = np.isfinite(u_out) & (conv_iter > 0) & (conv_iter <= 50)
    rmse = float(np.sqrt(np.mean((u_out[valid] - u_gt[valid]) ** 2))) if valid.any() else float("nan")

    return {
        "img_size": img_size,
        "step": step,
        "winsize": winsize,
        "n_nodes": n_nodes,
        "good_nodes": good_nodes,
        "bad_pt_num": int(bad_pt_num),
        "elapsed_s": elapsed,
        "wall_s": wall_elapsed,
        "nodes_per_sec": n_nodes / elapsed if elapsed > 0 else 0.0,
        "avg_iters": avg_iters,
        "rmse_px": rmse,
        "skipped": False,
    }


# ── Numba warmup ─────────────────────────────────────────────────────────────

def _warmup() -> None:
    """Trigger Numba JIT compilation before any timed run.

    The Numba prange path is activated only when N >= 50.
    Use 256×256 step=8 (784 nodes) to ensure the parallel kernel is compiled.
    """
    print("Warming up Numba JIT (first call compiles kernels)...", flush=True)
    t0 = time.perf_counter()
    _run_benchmark(img_size=256, step=8, winsize=32)
    elapsed = time.perf_counter() - t0
    print(f"  Warmup done in {elapsed:.1f}s\n", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from staq_dic.export.export_utils import make_timestamp

    print("=" * 68)
    print("  Local IC-GN Benchmark  (no ADMM, pure solver throughput)")
    print("=" * 68)
    print()

    _warmup()

    # Scenarios: (image_size, step)
    scenarios = [
        (256,  16),
        (256,   8),
        (256,   4),
        (512,  16),
        (512,   8),
        (512,   4),
        (1024, 16),
        (1024,  8),
        (1024,  4),
    ]

    results = []
    for img_size, step in scenarios:
        label = f"  {img_size}×{img_size}  step={step:2d}"
        print(f"{label}  running...", end="", flush=True)
        r = _run_benchmark(img_size=img_size, step=step)
        if r.get("skipped"):
            print("  [skipped — no nodes]")
            continue
        print(
            f"\r{label}  {r['n_nodes']:5d} nodes  "
            f"{r['nodes_per_sec']:8.0f} nodes/s  "
            f"RMSE={r['rmse_px']:.4f}px  "
            f"iters={r['avg_iters']:.1f}  "
            f"t={r['elapsed_s']:.3f}s"
        )
        results.append(r)

    if not results:
        print("No results collected.")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print(f"{'Image':>12}  {'Step':>4}  {'Nodes':>6}  "
          f"{'nodes/s':>9}  {'RMSE(px)':>9}  {'Iters':>5}  {'Time(s)':>7}")
    print("-" * 68)
    for r in results:
        print(
            f"  {r['img_size']}×{r['img_size']:4d}  "
            f"{r['step']:4d}  "
            f"{r['n_nodes']:6d}  "
            f"{r['nodes_per_sec']:9.0f}  "
            f"{r['rmse_px']:9.5f}  "
            f"{r['avg_iters']:5.1f}  "
            f"{r['elapsed_s']:7.3f}"
        )
    print("=" * 68)

    best = max(results, key=lambda r: r["nodes_per_sec"])
    worst = min(results, key=lambda r: r["nodes_per_sec"])
    print(f"\n  Peak:    {best['nodes_per_sec']:,.0f} nodes/s  "
          f"({best['img_size']}×{best['img_size']} step={best['step']})")
    print(f"  Lowest:  {worst['nodes_per_sec']:,.0f} nodes/s  "
          f"({worst['img_size']}×{worst['img_size']} step={worst['step']})")
    print()

    # ── Save text report ──────────────────────────────────────────────────
    reports_dir = _HERE.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = make_timestamp()
    report_path = reports_dir / f"benchmark_local_icgn_{ts}.txt"

    lines = [
        "Local IC-GN Benchmark Report",
        f"Timestamp: {ts}",
        "Setup: synthetic speckle, 1px uniform x-translation, winsize=32, tol=1e-2",
        "",
        f"{'Image':>12}  {'Step':>4}  {'Nodes':>6}  "
        f"{'nodes/s':>9}  {'RMSE(px)':>9}  {'Iters':>5}  {'Time(s)':>7}",
        "-" * 68,
    ]
    for r in results:
        lines.append(
            f"  {r['img_size']}×{r['img_size']:4d}  "
            f"{r['step']:4d}  "
            f"{r['n_nodes']:6d}  "
            f"{r['nodes_per_sec']:9.0f}  "
            f"{r['rmse_px']:9.5f}  "
            f"{r['avg_iters']:5.1f}  "
            f"{r['elapsed_s']:7.3f}"
        )
    lines += [
        "=" * 68,
        f"Peak:   {best['nodes_per_sec']:,.0f} nodes/s  "
        f"({best['img_size']}×{best['img_size']} step={best['step']})",
        f"Lowest: {worst['nodes_per_sec']:,.0f} nodes/s  "
        f"({worst['img_size']}×{worst['img_size']} step={worst['step']})",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
