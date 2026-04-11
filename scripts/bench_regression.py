"""Regression benchmark: verify window splitting causes no degradation on solid masks.

For solid (all-ones) masks, window splitting should have zero effect because there
are no mask boundaries. Any RMSE or timing difference indicates a regression.

Test matrix:
    - 3 mesh densities:  step=16 (coarse), step=8 (medium), step=4 (fine)
    - 3 displacement fields: affine, quadratic, translation
    - 2 reference modes: accumulative (3 frames), incremental (3 frames)
    = 18 test configurations

For each config, runs pipeline twice:
    - NEW: raw images + all-ones mask  (current production code)
    - OLD: img*mask as images + all-ones mask  (simulates pre-fix behavior)

Since mask=1 everywhere, both should give identical results.

Usage:
    python bench_regression.py
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import DICMesh, GridxyROIRange, merge_uv
from al_dic.core.pipeline import run_aldic

logging.getLogger("al_dic").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_speckle(h=256, w=256, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    f = gaussian_filter(noise, sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def apply_displacement_lagrangian(ref, u_func, v_func, n_iter=20):
    from scipy.ndimage import map_coordinates
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    warped = map_coordinates(ref, [Y.ravel(), X.ravel()], order=5, mode="nearest")
    return warped.reshape(h, w)


def make_mesh(h, w, step, margin=None):
    if margin is None:
        margin = step
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    ny, nx = len(ys), len(xs)
    elems = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n0 = iy * nx + ix
            elems.append([n0, n0+1, n0+nx+1, n0+nx, -1, -1, -1, -1])
    elems = np.array(elems, dtype=np.int64) if elems else np.empty((0, 8), dtype=np.int64)
    return DICMesh(coordinates_fem=coords, elements_fem=elems, x0=xs, y0=ys)


def compute_rmse(U, gt_u, gt_v):
    u, v = U[0::2], U[1::2]
    ok = np.isfinite(u) & np.isfinite(v)
    if not np.any(ok):
        return np.inf
    return float(np.sqrt(np.mean((u[ok]-gt_u[ok])**2 + (v[ok]-gt_v[ok])**2)))


# ---------------------------------------------------------------------------
# Displacement fields
# ---------------------------------------------------------------------------

FIELDS = {
    "affine": {
        # 2% biaxial stretch
        "u2": lambda x, y: 0.02 * (x - CX),
        "v2": lambda x, y: 0.02 * (y - CY),
        "u3": lambda x, y: 0.04 * (x - CX),
        "v3": lambda x, y: 0.04 * (y - CY),
    },
    "quadratic": {
        # Quadratic (barrel distortion)
        "u2": lambda x, y: 1e-4 * (x - CX)**2 - 5e-5 * (y - CY)**2,
        "v2": lambda x, y: 1e-4 * (x - CX) * (y - CY),
        "u3": lambda x, y: 2e-4 * (x - CX)**2 - 1e-4 * (y - CY)**2,
        "v3": lambda x, y: 2e-4 * (x - CX) * (y - CY),
    },
    "translation": {
        # Rigid translation
        "u2": lambda x, y: np.full_like(x, 1.5),
        "v2": lambda x, y: np.full_like(x, -1.0),
        "u3": lambda x, y: np.full_like(x, 3.0),
        "v3": lambda x, y: np.full_like(x, -2.0),
    },
}

STEPS = [16, 8, 4]
MODES = ["accumulative", "incremental"]


# ---------------------------------------------------------------------------
# Single config runner
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    field: str
    step: int
    mode: str
    n_nodes: int
    rmse_new_f2: float
    rmse_old_f2: float
    rmse_new_f3: float
    rmse_old_f3: float
    time_new: float
    time_old: float


def run_one(field_name: str, step: int, ref_mode: str, ref_img: np.ndarray) -> BenchResult:
    """Run one configuration, return comparison results."""
    fld = FIELDS[field_name]
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)

    # Winsize = 2 * step (standard overlap ratio)
    winsize = max(2 * step, 8)
    margin = step

    # Generate deformed images
    deformed2 = apply_displacement_lagrangian(ref_img, fld["u2"], fld["v2"])
    deformed3 = apply_displacement_lagrangian(ref_img, fld["u3"], fld["v3"])

    # Build mesh + ground truth
    mesh = make_mesh(IMG_H, IMG_W, step=step, margin=margin)
    node_x, node_y = mesh.coordinates_fem[:, 0], mesh.coordinates_fem[:, 1]
    gt_u2, gt_v2 = fld["u2"](node_x, node_y), fld["v2"](node_x, node_y)
    gt_u3, gt_v3 = fld["u3"](node_x, node_y), fld["v3"](node_x, node_y)
    U0 = merge_uv(gt_u2, gt_v2)

    para = dicpara_default(
        winsize=winsize, winstepsize=step, winsize_min=max(step // 2, 4),
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode=ref_mode,
        admm_max_iter=3, admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0, disp_smoothness=0.0, smoothness=0.0,
        show_plots=False, icgn_max_iter=50, tol=1e-2,
        mu=1e-3, gauss_pt_order=2, alpha=0.0,
    )

    images_raw = [ref_img, deformed2, deformed3]
    images_old = [ref_img * mask, deformed2 * mask, deformed3 * mask]
    masks_list = [mask, mask, mask]

    results = {}
    for label, imgs in [("new", images_raw), ("old", images_old)]:
        # Run twice, take second timing
        for trial in range(2):
            t0 = time.perf_counter()
            res = run_aldic(
                para, images=imgs, masks=masks_list,
                compute_strain=False, mesh=mesh, U0=U0.copy(),
            )
            elapsed = time.perf_counter() - t0

        # Frame 2 = result_disp[0], Frame 3 = result_disp[1]
        U_f2 = res.result_disp[0].U_accum if res.result_disp[0].U_accum is not None else res.result_disp[0].U
        U_f3 = res.result_disp[1].U_accum if res.result_disp[1].U_accum is not None else res.result_disp[1].U

        results[label] = {
            "rmse_f2": compute_rmse(U_f2, gt_u2, gt_v2),
            "rmse_f3": compute_rmse(U_f3, gt_u3, gt_v3),
            "time": elapsed,
        }

    return BenchResult(
        field=field_name, step=step, mode=ref_mode,
        n_nodes=mesh.coordinates_fem.shape[0],
        rmse_new_f2=results["new"]["rmse_f2"],
        rmse_old_f2=results["old"]["rmse_f2"],
        rmse_new_f3=results["new"]["rmse_f3"],
        rmse_old_f3=results["old"]["rmse_f3"],
        time_new=results["new"]["time"],
        time_old=results["old"]["time"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 78)
    print("  Regression Benchmark: Window Splitting on Solid Masks")
    print("  Goal: NEW and OLD must produce identical results (mask=1 everywhere)")
    print("=" * 78)

    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)

    all_results: list[BenchResult] = []
    total = len(FIELDS) * len(STEPS) * len(MODES)
    idx = 0

    for field_name in FIELDS:
        for step in STEPS:
            for mode in MODES:
                idx += 1
                tag = f"[{idx:>2}/{total}] {field_name:12s} step={step:>2} {mode:13s}"
                print(f"  {tag} ...", end="", flush=True)
                r = run_one(field_name, step, mode, ref)
                # Check regression
                delta_f2 = abs(r.rmse_new_f2 - r.rmse_old_f2)
                delta_f3 = abs(r.rmse_new_f3 - r.rmse_old_f3)
                max_delta = max(delta_f2, delta_f3)
                status = "OK" if max_delta < 0.01 else f"WARN delta={max_delta:.4f}"
                print(f"  RMSE f2={r.rmse_new_f2:.4f}/{r.rmse_old_f2:.4f}  "
                      f"f3={r.rmse_new_f3:.4f}/{r.rmse_old_f3:.4f}  "
                      f"t={r.time_new:.3f}/{r.time_old:.3f}s  [{status}]")
                all_results.append(r)

    # --- Summary table ---
    print("\n" + "=" * 78)
    print("  SUMMARY TABLE")
    print("=" * 78)
    print(f"  {'Field':<12} {'Step':>4} {'Mode':<13} {'Nodes':>5} "
          f"{'RMSE_NEW':>9} {'RMSE_OLD':>9} {'Delta':>8} {'t_NEW':>6} {'t_OLD':>6} {'Status':>6}")
    print("  " + "-" * 76)

    n_pass = 0
    n_warn = 0
    for r in all_results:
        # Use frame 3 as the primary metric (harder case)
        delta = abs(r.rmse_new_f3 - r.rmse_old_f3)
        status = "PASS" if delta < 0.01 else "WARN"
        if status == "PASS":
            n_pass += 1
        else:
            n_warn += 1
        print(f"  {r.field:<12} {r.step:>4} {r.mode:<13} {r.n_nodes:>5} "
              f"{r.rmse_new_f3:>9.4f} {r.rmse_old_f3:>9.4f} {delta:>8.5f} "
              f"{r.time_new:>6.3f} {r.time_old:>6.3f} {status:>6}")

    print(f"\n  Result: {n_pass} PASS, {n_warn} WARN out of {total} configs")
    if n_warn == 0:
        print("  CONCLUSION: No regression detected. Window splitting is safe.")
    else:
        print("  CONCLUSION: REGRESSION DETECTED — investigate WARN cases.")

    # --- Visualization ---
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "bench_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_results(all_results, out_dir)
    print(f"\n  Figures saved to {out_dir}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results: list[BenchResult], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Organize data
    fields = list(FIELDS.keys())
    steps = STEPS
    modes = MODES

    # =================================================================
    # Fig 1: RMSE parity (NEW vs OLD) — scatter plot
    # =================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle("Regression Check: NEW vs OLD RMSE (solid mask, should be identical)",
                  fontsize=13, fontweight="bold")

    for i, field in enumerate(fields):
        ax = axes1[i]
        subset = [r for r in results if r.field == field]
        new_vals = [r.rmse_new_f3 for r in subset]
        old_vals = [r.rmse_old_f3 for r in subset]

        # Color by step size
        colors = []
        markers = []
        for r in subset:
            c = {16: "#2196F3", 8: "#FF9800", 4: "#4CAF50"}[r.step]
            m = "o" if r.mode == "accumulative" else "^"
            colors.append(c)
            markers.append(m)

        for j, r in enumerate(subset):
            ax.scatter(old_vals[j], new_vals[j], c=colors[j], marker=markers[j],
                       s=80, edgecolors="black", linewidths=0.5, zorder=5)

        # Perfect parity line
        lo = min(min(new_vals), min(old_vals)) * 0.9
        hi = max(max(new_vals), max(old_vals)) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="parity")
        ax.set_xlabel("OLD RMSE [px]")
        ax.set_ylabel("NEW RMSE [px]")
        ax.set_title(f"{field}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="step=16"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800",
               markersize=8, label="step=8"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="step=4"),
        Line2D([0], [0], marker="o", color="gray", markersize=8, label="accum"),
        Line2D([0], [0], marker="^", color="gray", markersize=8, label="incr"),
    ]
    axes1[-1].legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig1.tight_layout()
    fig1.savefig(out_dir / "rmse_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # =================================================================
    # Fig 2: RMSE vs node count (scaling behavior)
    # =================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("RMSE vs Mesh Density (accumulative mode, frame 3)",
                  fontsize=13, fontweight="bold")

    for i, field in enumerate(fields):
        ax = axes2[i]
        subset = [r for r in results if r.field == field and r.mode == "accumulative"]
        nodes = [r.n_nodes for r in subset]
        new_rmse = [r.rmse_new_f3 for r in subset]
        old_rmse = [r.rmse_old_f3 for r in subset]

        ax.plot(nodes, new_rmse, "o-", color="#2196F3", label="NEW", markersize=8)
        ax.plot(nodes, old_rmse, "s--", color="#FF5722", label="OLD", markersize=8)
        ax.set_xlabel("Node count")
        ax.set_ylabel("RMSE [px]")
        ax.set_title(f"{field}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        # Annotate step sizes
        for j, r in enumerate(subset):
            ax.annotate(f"step={r.step}", (nodes[j], new_rmse[j]),
                        textcoords="offset points", xytext=(5, 8), fontsize=7)

    fig2.tight_layout()
    fig2.savefig(out_dir / "rmse_vs_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # =================================================================
    # Fig 3: Timing comparison
    # =================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle("Timing: NEW vs OLD (should be identical for solid mask)",
                  fontsize=13, fontweight="bold")

    labels = [f"{r.field[:4]}\ns{r.step}\n{r.mode[:3]}" for r in results]
    x = np.arange(len(results))
    bw = 0.35
    t_new = [r.time_new for r in results]
    t_old = [r.time_old for r in results]

    ax3.bar(x - bw/2, t_new, bw, label="NEW", color="#2196F3", alpha=0.85)
    ax3.bar(x + bw/2, t_old, bw, label="OLD", color="#FF5722", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=6)
    ax3.set_ylabel("Time [s]")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(out_dir / "timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # =================================================================
    # Fig 4: Delta heatmap (field × step × mode)
    # =================================================================
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    fig4.suptitle("RMSE Delta (|NEW - OLD|) — should be ~0 everywhere",
                  fontsize=13, fontweight="bold")

    row_labels = [f"{r.field} s{r.step} {r.mode[:3]}" for r in results]
    deltas_f2 = [abs(r.rmse_new_f2 - r.rmse_old_f2) for r in results]
    deltas_f3 = [abs(r.rmse_new_f3 - r.rmse_old_f3) for r in results]
    data = np.array([deltas_f2, deltas_f3]).T  # (n_configs, 2)

    im = ax4.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)
    ax4.set_yticks(range(len(results)))
    ax4.set_yticklabels(row_labels, fontsize=7)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["Frame 2 delta", "Frame 3 delta"])
    fig4.colorbar(im, ax=ax4, shrink=0.7, label="|RMSE_NEW - RMSE_OLD| [px]")

    # Annotate values
    for i in range(len(results)):
        for j in range(2):
            val = data[i, j]
            color = "white" if val > data.max() * 0.5 else "black"
            ax4.text(j, i, f"{val:.5f}", ha="center", va="center",
                     fontsize=7, color=color)

    fig4.tight_layout()
    fig4.savefig(out_dir / "delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    print(f"  Saved: rmse_parity.png, rmse_vs_density.png, "
          f"timing_comparison.png, delta_heatmap.png")


if __name__ == "__main__":
    run_benchmark()
