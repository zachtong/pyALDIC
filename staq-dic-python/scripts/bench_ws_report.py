"""Window splitting performance report: accuracy, speed, and full-field DIC results.

Standalone report for the current (window-splitting-enabled) pipeline on solid masks.
Generates:
    1. Full-field displacement maps + error maps for each config
    2. Summary table (RMSE, timing, node count)
    3. RMSE vs mesh density scaling curve

Test matrix:
    - 3 displacement fields: affine, quadratic, translation
    - 3 mesh densities:      step=16 (coarse), step=8 (medium), step=4 (fine)
    - 2 reference modes:     accumulative, incremental
    = 18 configs (full-field plots for accumulative only, 9 figures)

Usage:
    python bench_ws_report.py
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICMesh, GridxyROIRange, merge_uv
from staq_dic.core.pipeline import run_aldic

logging.getLogger("staq_dic").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0
N_TIMING_TRIALS = 3  # median of trials[1:] to skip JIT warmup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_speckle(h=256, w=256, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    f = gaussian_filter(noise, sigma=sigma, mode="nearest")
    f -= f.min()
    f /= f.max()
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
            elems.append([n0, n0 + 1, n0 + nx + 1, n0 + nx, -1, -1, -1, -1])
    elems = np.array(elems, dtype=np.int64) if elems else np.empty((0, 8), dtype=np.int64)
    return DICMesh(coordinates_fem=coords, elements_fem=elems, x0=xs, y0=ys), (nx, ny)


# ---------------------------------------------------------------------------
# Displacement fields
# ---------------------------------------------------------------------------

FIELDS = {
    "affine": {
        "label": "2% biaxial stretch",
        "u2": lambda x, y: 0.02 * (x - CX),
        "v2": lambda x, y: 0.02 * (y - CY),
        "u3": lambda x, y: 0.04 * (x - CX),
        "v3": lambda x, y: 0.04 * (y - CY),
    },
    "quadratic": {
        "label": "Barrel distortion",
        "u2": lambda x, y: 1e-4 * (x - CX) ** 2 - 5e-5 * (y - CY) ** 2,
        "v2": lambda x, y: 1e-4 * (x - CX) * (y - CY),
        "u3": lambda x, y: 2e-4 * (x - CX) ** 2 - 1e-4 * (y - CY) ** 2,
        "v3": lambda x, y: 2e-4 * (x - CX) * (y - CY),
    },
    "translation": {
        "label": "Rigid translation",
        "u2": lambda x, y: np.full_like(x, 1.5),
        "v2": lambda x, y: np.full_like(x, -1.0),
        "u3": lambda x, y: np.full_like(x, 3.0),
        "v3": lambda x, y: np.full_like(x, -2.0),
    },
}

STEPS = [16, 8, 4]
MODES = ["accumulative", "incremental"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    field: str
    step: int
    mode: str
    n_nodes: int
    nx: int
    ny: int
    # Frame 3 results (hardest frame — largest displacement)
    meas_u: np.ndarray  # (n_nodes,)
    meas_v: np.ndarray
    gt_u: np.ndarray
    gt_v: np.ndarray
    err_u: np.ndarray
    err_v: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    rmse_f2: float
    rmse_f3: float
    rmse_u_f3: float
    rmse_v_f3: float
    max_err_f3: float
    time_median: float


# ---------------------------------------------------------------------------
# Single config runner
# ---------------------------------------------------------------------------


def run_one(field_name: str, step: int, ref_mode: str,
            ref_img: np.ndarray) -> ConfigResult:
    fld = FIELDS[field_name]
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
    winsize = max(2 * step, 8)
    margin = step

    deformed2 = apply_displacement_lagrangian(ref_img, fld["u2"], fld["v2"])
    deformed3 = apply_displacement_lagrangian(ref_img, fld["u3"], fld["v3"])

    mesh, (nx, ny) = make_mesh(IMG_H, IMG_W, step=step, margin=margin)
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]
    gt_u2 = fld["u2"](node_x, node_y)
    gt_v2 = fld["v2"](node_x, node_y)
    gt_u3 = fld["u3"](node_x, node_y)
    gt_v3 = fld["v3"](node_x, node_y)
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

    images = [ref_img, deformed2, deformed3]
    masks_list = [mask, mask, mask]

    # Timing: N trials, take median of trials[1:]
    timings = []
    res = None
    for trial in range(N_TIMING_TRIALS):
        t0 = time.perf_counter()
        res = run_aldic(
            para, images=images, masks=masks_list,
            compute_strain=False, mesh=mesh, U0=U0.copy(),
        )
        timings.append(time.perf_counter() - t0)

    t_median = float(np.median(timings[1:])) if len(timings) > 1 else timings[0]

    # Extract frame 2 & 3 displacements
    U_f2 = res.result_disp[0].U_accum if res.result_disp[0].U_accum is not None else res.result_disp[0].U
    U_f3 = res.result_disp[1].U_accum if res.result_disp[1].U_accum is not None else res.result_disp[1].U

    meas_u3 = U_f3[0::2]
    meas_v3 = U_f3[1::2]
    err_u3 = meas_u3 - gt_u3
    err_v3 = meas_v3 - gt_v3

    ok2 = np.isfinite(U_f2[0::2]) & np.isfinite(U_f2[1::2])
    ok3 = np.isfinite(meas_u3) & np.isfinite(meas_v3)

    def _rmse(a, b, m):
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if np.any(m) else np.inf

    rmse_f2 = float(np.sqrt(np.mean(
        (U_f2[0::2][ok2] - gt_u2[ok2]) ** 2 + (U_f2[1::2][ok2] - gt_v2[ok2]) ** 2
    ))) if np.any(ok2) else np.inf

    rmse_f3 = float(np.sqrt(np.mean(
        (meas_u3[ok3] - gt_u3[ok3]) ** 2 + (meas_v3[ok3] - gt_v3[ok3]) ** 2
    ))) if np.any(ok3) else np.inf

    rmse_u = _rmse(meas_u3, gt_u3, ok3)
    rmse_v = _rmse(meas_v3, gt_v3, ok3)
    max_err = float(np.max(np.abs(np.concatenate([err_u3[ok3], err_v3[ok3]])))) if np.any(ok3) else np.inf

    return ConfigResult(
        field=field_name, step=step, mode=ref_mode,
        n_nodes=mesh.coordinates_fem.shape[0], nx=nx, ny=ny,
        meas_u=meas_u3, meas_v=meas_v3, gt_u=gt_u3, gt_v=gt_v3,
        err_u=err_u3, err_v=err_v3,
        node_x=node_x, node_y=node_y,
        rmse_f2=rmse_f2, rmse_f3=rmse_f3,
        rmse_u_f3=rmse_u, rmse_v_f3=rmse_v,
        max_err_f3=max_err, time_median=t_median,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_fullfield(results: list[ConfigResult], out_dir: Path):
    """Generate full-field displacement + error maps for accumulative configs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    accum = [r for r in results if r.mode == "accumulative"]

    for field_name in FIELDS:
        subset = [r for r in accum if r.field == field_name]
        subset.sort(key=lambda r: -r.step)  # coarse to fine

        fig, axes = plt.subplots(
            len(subset), 6, figsize=(24, 4 * len(subset)),
            squeeze=False,
        )
        fig.suptitle(
            f"Window Splitting DIC — {field_name} ({FIELDS[field_name]['label']})\n"
            f"Full-field results (frame 3, accumulative mode)",
            fontsize=14, fontweight="bold", y=0.98,
        )

        col_titles = [
            "GT u [px]", "Measured u [px]", "Error u [px]",
            "GT v [px]", "Measured v [px]", "Error v [px]",
        ]

        for row_idx, r in enumerate(subset):
            # Reshape to grid
            gt_u_2d = r.gt_u.reshape(r.ny, r.nx)
            gt_v_2d = r.gt_v.reshape(r.ny, r.nx)
            m_u_2d = r.meas_u.reshape(r.ny, r.nx)
            m_v_2d = r.meas_v.reshape(r.ny, r.nx)
            e_u_2d = r.err_u.reshape(r.ny, r.nx)
            e_v_2d = r.err_v.reshape(r.ny, r.nx)

            xs = r.node_x.reshape(r.ny, r.nx)
            ys = r.node_y.reshape(r.ny, r.nx)

            # Displacement colormap limits (shared for GT + measured)
            u_lim = max(abs(np.nanmin(gt_u_2d)), abs(np.nanmax(gt_u_2d)), 0.01)
            v_lim = max(abs(np.nanmin(gt_v_2d)), abs(np.nanmax(gt_v_2d)), 0.01)

            # Error colormap limit
            e_lim = max(
                np.nanmax(np.abs(e_u_2d[np.isfinite(e_u_2d)])),
                np.nanmax(np.abs(e_v_2d[np.isfinite(e_v_2d)])),
                0.001,
            )

            panels = [
                (gt_u_2d, u_lim, "RdBu_r"),
                (m_u_2d, u_lim, "RdBu_r"),
                (e_u_2d, e_lim, "seismic"),
                (gt_v_2d, v_lim, "RdBu_r"),
                (m_v_2d, v_lim, "RdBu_r"),
                (e_v_2d, e_lim, "seismic"),
            ]

            for col_idx, (data, lim, cmap) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                im = ax.pcolormesh(
                    xs, ys, data, cmap=cmap,
                    vmin=-lim, vmax=lim, shading="nearest",
                )
                ax.set_aspect("equal")
                ax.invert_yaxis()
                fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"step={r.step}\n({r.nx}×{r.ny}={r.n_nodes} nodes)",
                        fontsize=9,
                    )
                ax.tick_params(labelsize=7)

            # Add RMSE annotation on error panels
            for col_idx, (rmse_val, comp) in [(2, (r.rmse_u_f3, "u")), (5, (r.rmse_v_f3, "v"))]:
                ax = axes[row_idx, col_idx]
                ax.text(
                    0.02, 0.95,
                    f"RMSE={rmse_val:.4f} px\nmax|err|={r.max_err_f3:.4f} px",
                    transform=ax.transAxes, fontsize=7,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = out_dir / f"fullfield_{field_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {fname.name}")


def plot_summary(results: list[ConfigResult], out_dir: Path):
    """Summary figures: table, RMSE scaling, timing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = list(FIELDS.keys())
    steps = STEPS

    # =================================================================
    # Fig 1: Summary table as figure
    # =================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.axis("off")
    fig1.suptitle(
        "Window Splitting Performance Report — Summary Table",
        fontsize=14, fontweight="bold",
    )

    col_labels = [
        "Field", "Step", "Mode", "Nodes",
        "RMSE_f2\n[px]", "RMSE_f3\n[px]", "RMSE_u\n[px]", "RMSE_v\n[px]",
        "Max|err|\n[px]", "Time\n[s]",
    ]
    table_data = []
    cell_colors = []

    for r in sorted(results, key=lambda x: (x.field, -x.step, x.mode)):
        row = [
            r.field, str(r.step), r.mode[:5],
            str(r.n_nodes),
            f"{r.rmse_f2:.4f}", f"{r.rmse_f3:.4f}",
            f"{r.rmse_u_f3:.4f}", f"{r.rmse_v_f3:.4f}",
            f"{r.max_err_f3:.4f}", f"{r.time_median:.3f}",
        ]
        table_data.append(row)

        # Color-code RMSE: green < 0.05, yellow 0.05-0.2, orange > 0.2
        rmse = r.rmse_f3
        if rmse < 0.05:
            c = "#c8e6c9"  # green
        elif rmse < 0.2:
            c = "#fff9c4"  # yellow
        else:
            c = "#ffe0b2"  # orange
        cell_colors.append([c] * len(col_labels))

    tbl = ax1.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig1.tight_layout()
    fig1.savefig(out_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("    Saved summary_table.png")

    # =================================================================
    # Fig 2: RMSE vs mesh density (scaling behavior)
    # =================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(
        "RMSE vs Mesh Density — Window Splitting Enabled",
        fontsize=13, fontweight="bold",
    )

    for i, fld in enumerate(fields):
        ax = axes2[i]
        for mode, ls, marker in [("accumulative", "-", "o"), ("incremental", "--", "s")]:
            subset = sorted(
                [r for r in results if r.field == fld and r.mode == mode],
                key=lambda r: r.n_nodes,
            )
            nodes = [r.n_nodes for r in subset]
            rmse_u = [r.rmse_u_f3 for r in subset]
            rmse_v = [r.rmse_v_f3 for r in subset]
            rmse_t = [r.rmse_f3 for r in subset]

            ax.plot(nodes, rmse_t, f"{marker}{ls}", label=f"total ({mode[:5]})", linewidth=2, markersize=7)
            ax.plot(nodes, rmse_u, f"{marker}{ls}", label=f"u ({mode[:5]})", linewidth=1, alpha=0.6, markersize=5)
            ax.plot(nodes, rmse_v, f"{marker}{ls}", label=f"v ({mode[:5]})", linewidth=1, alpha=0.6, markersize=5)

        ax.set_xlabel("Node count", fontsize=10)
        ax.set_ylabel("RMSE [px]", fontsize=10)
        ax.set_title(f"{fld} ({FIELDS[fld]['label']})", fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")

        # Annotate step sizes
        for r in [rr for rr in results if rr.field == fld and rr.mode == "accumulative"]:
            ax.annotate(
                f"s{r.step}", (r.n_nodes, r.rmse_f3),
                textcoords="offset points", xytext=(5, 8), fontsize=7,
            )

    fig2.tight_layout()
    fig2.savefig(out_dir / "rmse_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("    Saved rmse_scaling.png")

    # =================================================================
    # Fig 3: Timing vs node count
    # =================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle(
        "Pipeline Timing vs Node Count — Window Splitting Enabled",
        fontsize=13, fontweight="bold",
    )

    colors = {"affine": "#2196F3", "quadratic": "#4CAF50", "translation": "#FF9800"}
    for mode, ls, marker in [("accumulative", "-", "o"), ("incremental", "--", "s")]:
        for fld in fields:
            subset = sorted(
                [r for r in results if r.field == fld and r.mode == mode],
                key=lambda r: r.n_nodes,
            )
            nodes = [r.n_nodes for r in subset]
            times = [r.time_median for r in subset]
            ax3.plot(
                nodes, times, f"{marker}{ls}",
                color=colors[fld], linewidth=2, markersize=7,
                label=f"{fld} ({mode[:5]})",
            )

    ax3.set_xlabel("Node count", fontsize=10)
    ax3.set_ylabel("Time [s]", fontsize=10)
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(alpha=0.3)
    ax3.set_xscale("log")

    fig3.tight_layout()
    fig3.savefig(out_dir / "timing_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("    Saved timing_scaling.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("  Window Splitting Performance Report")
    print("  Pipeline with masked-subset IC-GN (production code)")
    print("=" * 78)

    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)

    all_results: list[ConfigResult] = []
    total = len(FIELDS) * len(STEPS) * len(MODES)
    idx = 0

    for field_name in FIELDS:
        for step in STEPS:
            for mode in MODES:
                idx += 1
                tag = f"[{idx:>2}/{total}] {field_name:12s} step={step:>2} {mode:13s}"
                print(f"  {tag} ...", end="", flush=True)
                r = run_one(field_name, step, mode, ref)
                print(
                    f"  RMSE={r.rmse_f3:.4f} px  "
                    f"(u={r.rmse_u_f3:.4f}, v={r.rmse_v_f3:.4f})  "
                    f"max|err|={r.max_err_f3:.4f}  "
                    f"t={r.time_median:.3f}s  "
                    f"[{r.n_nodes} nodes]"
                )
                all_results.append(r)

    # --- Console summary ---
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'Field':<12} {'Step':>4} {'Mode':<6} {'Nodes':>5} "
          f"{'RMSE_f3':>8} {'RMSE_u':>7} {'RMSE_v':>7} {'Max|e|':>7} {'Time':>6}")
    print("  " + "-" * 72)
    for r in sorted(all_results, key=lambda x: (x.field, -x.step, x.mode)):
        print(f"  {r.field:<12} {r.step:>4} {r.mode[:5]:<6} {r.n_nodes:>5} "
              f"{r.rmse_f3:>8.4f} {r.rmse_u_f3:>7.4f} {r.rmse_v_f3:>7.4f} "
              f"{r.max_err_f3:>7.4f} {r.time_median:>6.3f}")

    # --- Figures ---
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "bench_ws_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating figures in {out_dir} ...")
    plot_fullfield(all_results, out_dir)
    plot_summary(all_results, out_dir)

    print(f"\n  All figures saved to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
