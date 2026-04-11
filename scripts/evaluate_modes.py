#!/usr/bin/env python
"""Evaluate accumulative vs incremental vs skip-3 key-frame tracking.

Generates a 10-frame progressive deformation sequence for 5 deformation
types, runs the full AL-DIC pipeline with 3 tracking strategies, and
compares per-frame RMSE.

Deformation types (Lagrangian, peak at frame 10):
    1. Translation:  +0.5 px/frame  →  4.5 px
    2. Affine:       +0.5%/frame    → ~5 px at edge
    3. Quadratic:    +0.5 px/frame  → ~7 px at edge
    4. Rotation:     +0.5 deg/frame → ~10 px at r=127
    5. Combined:     affine + quadratic + rotation (reduced amplitudes)

Tracking strategies:
    - Accumulative:  ref_indices = (0,0,...,0)
    - Incremental:   ref_indices = (0,1,...,8)
    - Skip-3:        ref_indices = (0,0,0,3,3,3,6,6,6)

Usage:
    python scripts/evaluate_modes.py [--output-dir reports/mode_comparison]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

# Add project root to path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ / "src") not in sys.path:
    sys.path.insert(0, str(_PROJ / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICPara,
    FrameSchedule,
    GridxyROIRange,
    merge_uv,
)
from al_dic.core.pipeline import run_aldic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0
N_FRAMES = 10  # total frames (1 ref + 9 deformed)
STEP = 16
MARGIN = 16


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42):
    """Synthetic speckle pattern."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement_lagrangian(
    ref: np.ndarray,
    u_func,
    v_func,
    n_iter: int = 20,
) -> np.ndarray:
    """Generate deformed image using Lagrangian displacement convention.

    DIC solves for u(X) where X is the reference coordinate:
        x = X + u(X)

    To build g(x) = f(X), we need to invert x = X + u(X) for each
    deformed pixel x.  Fixed-point iteration:
        X_{k+1} = x - u(X_k)

    converges for |du/dX| < 1 (satisfied for typical DIC deformations).
    """
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Fixed-point iteration to find reference coords X for each deformed pixel x
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)

    coords = np.array([Y.ravel(), X.ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def make_mesh_for_image(h, w, step, margin):
    """Simple regular Q4 mesh."""
    from al_dic.core.data_structures import DICMesh

    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    ny, nx = len(ys), len(xs)
    elements = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n0 = iy * nx + ix
            elements.append([n0, n0 + 1, n0 + nx + 1, n0 + nx, -1, -1, -1, -1])
    elems = np.array(elements, dtype=np.int64) if elements else np.empty((0, 8), dtype=np.int64)
    return DICMesh(coordinates_fem=coords, elements_fem=elems, x0=xs, y0=ys)


# ---------------------------------------------------------------------------
# Deformation definitions
# ---------------------------------------------------------------------------


def _deformation_funcs(deform_type: str, frame_k: int):
    """Return (u_func, v_func) for the k-th deformed frame (k >= 1).

    Each function takes (x_grid, y_grid) reference coordinate grids and
    returns Lagrangian displacement fields of the same shape.

    Amplitude design: peak displacement at frame 10 stays within ~10 px
    so IC-GN (winsize=32) can track reliably with the initial guess.
    """
    if deform_type == "translation":
        # +0.5 px/frame → 4.5 px total at frame 10
        tx = 0.5 * frame_k
        return (
            lambda x, y, _tx=tx: np.full_like(x, _tx),
            lambda x, y: np.zeros_like(x),
        )

    if deform_type == "affine":
        # +0.5% strain/frame → ~4.5% at frame 10, peak ~5 px at mesh edge
        eps = 0.005 * frame_k
        return (
            lambda x, y, _e=eps: _e * (x - CX),
            lambda x, y, _e=eps: _e * (y - CY),
        )

    if deform_type == "quadratic":
        # +0.5 px peak/frame → ~4.5 px peak at frame 10
        # At mesh edge (x=16 or 240): u ≈ 4.5 * ((113/90)^2) ≈ 7.1 px
        amp = 0.5 * frame_k
        R = 90.0
        return (
            lambda x, y, _a=amp, _R=R: _a * ((x - CX) / _R) ** 2,
            lambda x, y, _a=amp, _R=R: _a * ((y - CY) / _R) ** 2,
        )

    if deform_type == "rotation":
        # +0.5 deg/frame → ~4.5 deg at frame 10, peak ~10 px at r=127
        theta = np.radians(0.5 * frame_k)
        ct, st = np.cos(theta), np.sin(theta)
        return (
            lambda x, y, _ct=ct, _st=st: (x - CX) * (_ct - 1) - (y - CY) * _st,
            lambda x, y, _ct=ct, _st=st: (x - CX) * _st + (y - CY) * (_ct - 1),
        )

    if deform_type == "combined":
        # Affine + quadratic + rotation, each at ~1/3 of standalone amplitude
        eps = 0.002 * frame_k
        amp = 0.2 * frame_k
        R = 90.0
        theta = np.radians(0.2 * frame_k)
        ct, st = np.cos(theta), np.sin(theta)

        def u_func(x, y, _e=eps, _a=amp, _R=R, _ct=ct, _st=st):
            u_aff = _e * (x - CX)
            u_quad = _a * ((x - CX) / _R) ** 2
            u_rot = (x - CX) * (_ct - 1) - (y - CY) * _st
            return u_aff + u_quad + u_rot

        def v_func(x, y, _e=eps, _a=amp, _R=R, _ct=ct, _st=st):
            v_aff = _e * (y - CY)
            v_quad = _a * ((y - CY) / _R) ** 2
            v_rot = (x - CX) * _st + (y - CY) * (_ct - 1)
            return v_aff + v_quad + v_rot

        return u_func, v_func

    raise ValueError(f"Unknown deformation type: {deform_type}")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


STRATEGIES = {
    "accumulative": FrameSchedule.from_mode("accumulative", N_FRAMES),
    "incremental": FrameSchedule.from_mode("incremental", N_FRAMES),
    "skip3": FrameSchedule(ref_indices=(0, 0, 0, 3, 3, 3, 6, 6, 6)),
}

DEFORM_TYPES = ["translation", "affine", "quadratic", "rotation", "combined"]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def compute_disp_rmse(
    U: NDArray, coords: NDArray, gt_u: NDArray, gt_v: NDArray,
    edge_margin: int = 32,
) -> tuple[float, float]:
    """RMSE on interior nodes (excluding edge-margin pixels from image border).

    Using edge exclusion instead of a circular mask avoids IC-GN window
    contamination from mask-zeroed pixels near mask boundaries.
    """
    u_comp, v_comp = U[0::2], U[1::2]
    x, y = coords[:, 0], coords[:, 1]
    interior = (
        (x >= edge_margin) & (x <= IMG_W - 1 - edge_margin) &
        (y >= edge_margin) & (y <= IMG_H - 1 - edge_margin)
    )
    valid = interior & np.isfinite(u_comp) & np.isfinite(v_comp)
    if not np.any(valid):
        return np.inf, np.inf
    err_u = u_comp[valid] - gt_u[valid]
    err_v = v_comp[valid] - gt_v[valid]
    return float(np.sqrt(np.mean(err_u**2))), float(np.sqrt(np.mean(err_v**2)))


def run_evaluation(
    deform_type: str,
    strategy_name: str,
    schedule: FrameSchedule,
    ref_img: NDArray,
    mesh,
) -> dict:
    """Run pipeline for one deformation type + one strategy.

    Returns dict with:
        rmse_u: list[float] per deformed frame
        rmse_v: list[float] per deformed frame
        rmse_total: list[float] per deformed frame
        elapsed: float (seconds)
    """
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]

    # Generate images using Lagrangian convention
    # u_func(X, Y) is the true displacement at reference coord (X, Y).
    # apply_displacement_lagrangian inverts x = X + u(X) via fixed-point
    # iteration so that g(x) = f(X) exactly.
    imgs = [ref_img]
    gt_u_list, gt_v_list = [], []
    for k in range(1, N_FRAMES):
        u_func, v_func = _deformation_funcs(deform_type, k)
        imgs.append(apply_displacement_lagrangian(ref_img, u_func, v_func))
        gt_u_list.append(u_func(node_x, node_y))
        gt_v_list.append(v_func(node_x, node_y))

    # Use full-image mask for pipeline — circular masks zero out image pixels
    # near the boundary, which corrupts IC-GN subsets and propagates through ADMM.
    full_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
    masks_list = [full_mask] * N_FRAMES

    # Initial guess for first pair
    u0_func, v0_func = _deformation_funcs(deform_type, 1)
    U0 = merge_uv(u0_func(node_x, node_y), v0_func(node_x, node_y))

    para = dicpara_default(
        winsize=32,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="incremental",
        frame_schedule=schedule,
        admm_max_iter=3,
        admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0,
        disp_smoothness=0.0,
        smoothness=0.0,
        show_plots=False,
        icgn_max_iter=50,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        alpha=0.0,
    )

    t0 = time.perf_counter()
    result = run_aldic(para, imgs, masks_list, mesh=mesh, U0=U0, compute_strain=False)
    elapsed = time.perf_counter() - t0

    rmse_u, rmse_v, rmse_total = [], [], []
    coords = result.dic_mesh.coordinates_fem

    # Store per-frame full-field data for visualization
    field_data: list[dict] = []

    for i in range(N_FRAMES - 1):
        if i < len(result.result_disp):
            fr = result.result_disp[i]
            U = fr.U_accum if fr.U_accum is not None else fr.U
            ru, rv = compute_disp_rmse(U, coords, gt_u_list[i], gt_v_list[i])
            field_data.append(dict(
                u_comp=U[0::2].copy(),
                v_comp=U[1::2].copy(),
                gt_u=gt_u_list[i],
                gt_v=gt_v_list[i],
            ))
        else:
            ru, rv = np.inf, np.inf
            field_data.append(None)
        rmse_u.append(ru)
        rmse_v.append(rv)
        rmse_total.append(np.sqrt(ru**2 + rv**2))

    return dict(
        rmse_u=rmse_u, rmse_v=rmse_v, rmse_total=rmse_total,
        elapsed=elapsed,
        coords=coords, field_data=field_data,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_field_maps(all_results: dict, output_dir: Path) -> None:
    """Generate full-field displacement contour maps for each deformation × strategy.

    For each deformation type, produces a figure showing:
        - Row 0: Ground truth (u, v)
        - Rows 1-3: Each strategy's computed (u, v) and error magnitude
    Frames shown: 3, 6, 9 (early, mid, late).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
    except ImportError:
        print("matplotlib not available — skipping field maps")
        return

    field_dir = output_dir / "field_maps"
    field_dir.mkdir(parents=True, exist_ok=True)

    show_frames = [2, 5, 8]  # 0-based indices → frames 3, 6, 9
    strategy_names = list(STRATEGIES.keys())

    for deform_type in DEFORM_TYPES:
        first_strat = strategy_names[0]
        sample = all_results[deform_type][first_strat]
        coords = sample["coords"]
        x_nodes = coords[:, 0]
        y_nodes = coords[:, 1]

        # Build Delaunay triangulation once for this coordinate set
        tri = Triangulation(x_nodes, y_nodes)

        for frame_idx in show_frames:
            frame_num = frame_idx + 2  # display label

            # --- Figure layout: 4 rows × 3 cols ---
            # Row 0: GT u, GT v, (empty)
            # Rows 1-3: strategy u, strategy v, error magnitude
            n_rows = 1 + len(strategy_names)
            fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.2 * n_rows))

            # Collect global ranges for consistent colorbar
            gt_data = sample["field_data"][frame_idx]
            if gt_data is None:
                continue
            gt_u = gt_data["gt_u"]
            gt_v = gt_data["gt_v"]

            u_min, u_max = float(np.nanmin(gt_u)), float(np.nanmax(gt_u))
            v_min, v_max = float(np.nanmin(gt_v)), float(np.nanmax(gt_v))
            # Expand range to include computed values
            for sn in strategy_names:
                fd = all_results[deform_type][sn]["field_data"][frame_idx]
                if fd is None:
                    continue
                u_min = min(u_min, float(np.nanmin(fd["u_comp"])))
                u_max = max(u_max, float(np.nanmax(fd["u_comp"])))
                v_min = min(v_min, float(np.nanmin(fd["v_comp"])))
                v_max = max(v_max, float(np.nanmax(fd["v_comp"])))

            # Ensure a minimum range for uniform fields (e.g. pure translation)
            u_pad = max(abs(u_max - u_min) * 0.05, 0.1)
            v_pad = max(abs(v_max - v_min) * 0.05, 0.1)
            if u_max - u_min < 1e-6:
                u_min -= u_pad
                u_max += u_pad
            if v_max - v_min < 1e-6:
                v_min -= v_pad
                v_max += v_pad

            # Explicit shared levels for consistent colorbars across all rows
            u_levels = np.linspace(u_min, u_max, 21)
            v_levels = np.linspace(v_min, v_max, 21)

            # --- Row 0: Ground truth ---
            ax_gt_u = axes[0, 0]
            tc = ax_gt_u.tricontourf(tri, gt_u, levels=u_levels, cmap="RdBu_r",
                                     extend="both")
            fig.colorbar(tc, ax=ax_gt_u, shrink=0.8)
            ax_gt_u.set_title(f"GT u (frame {frame_num})")
            ax_gt_u.set_aspect("equal")
            ax_gt_u.invert_yaxis()

            ax_gt_v = axes[0, 1]
            tc = ax_gt_v.tricontourf(tri, gt_v, levels=v_levels, cmap="RdBu_r",
                                     extend="both")
            fig.colorbar(tc, ax=ax_gt_v, shrink=0.8)
            ax_gt_v.set_title(f"GT v (frame {frame_num})")
            ax_gt_v.set_aspect("equal")
            ax_gt_v.invert_yaxis()

            axes[0, 2].axis("off")
            axes[0, 2].text(0.5, 0.5, f"{deform_type}\nframe {frame_num}",
                            ha="center", va="center", fontsize=14,
                            transform=axes[0, 2].transAxes)

            # --- Rows 1+: each strategy ---
            err_max_global = 0.0
            for j, sn in enumerate(strategy_names):
                fd = all_results[deform_type][sn]["field_data"][frame_idx]
                if fd is None:
                    for c in range(3):
                        axes[j + 1, c].axis("off")
                    continue
                err = np.sqrt((fd["u_comp"] - gt_u)**2 + (fd["v_comp"] - gt_v)**2)
                err_max_global = max(err_max_global, float(np.nanmax(err)))

            for j, sn in enumerate(strategy_names):
                fd = all_results[deform_type][sn]["field_data"][frame_idx]
                if fd is None:
                    for c in range(3):
                        axes[j + 1, c].axis("off")
                    continue

                row = j + 1

                # u component
                ax_u = axes[row, 0]
                tc = ax_u.tricontourf(tri, fd["u_comp"], levels=u_levels,
                                      cmap="RdBu_r", extend="both")
                fig.colorbar(tc, ax=ax_u, shrink=0.8)
                ax_u.set_title(f"{sn} u")
                ax_u.set_aspect("equal")
                ax_u.invert_yaxis()

                # v component
                ax_v = axes[row, 1]
                tc = ax_v.tricontourf(tri, fd["v_comp"], levels=v_levels,
                                      cmap="RdBu_r", extend="both")
                fig.colorbar(tc, ax=ax_v, shrink=0.8)
                ax_v.set_title(f"{sn} v")
                ax_v.set_aspect("equal")
                ax_v.invert_yaxis()

                # Error magnitude
                err = np.sqrt((fd["u_comp"] - gt_u)**2 + (fd["v_comp"] - gt_v)**2)
                ax_e = axes[row, 2]
                err_ceil = max(err_max_global, 0.01)
                tc = ax_e.tricontourf(tri, err, levels=20, cmap="hot_r",
                                      vmin=0, vmax=err_ceil)
                fig.colorbar(tc, ax=ax_e, shrink=0.8)
                # RMSE on interior nodes only (matching the table)
                interior = (
                    (x_nodes >= 32) & (x_nodes <= IMG_W - 33) &
                    (y_nodes >= 32) & (y_nodes <= IMG_H - 33) &
                    np.isfinite(fd["u_comp"]) & np.isfinite(fd["v_comp"])
                )
                rmse_val = float(np.sqrt(np.mean(err[interior]**2))) if np.any(interior) else 0.0
                ax_e.set_title(f"{sn} |error| (RMSE={rmse_val:.3f})")
                ax_e.set_aspect("equal")
                ax_e.invert_yaxis()

            fig.suptitle(f"{deform_type} — frame {frame_num}", fontsize=14, y=1.01)
            fig.tight_layout()
            fname = field_dir / f"field_{deform_type}_frame{frame_num:02d}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")


def plot_results(all_results: dict, output_dir: Path) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(2, N_FRAMES + 1)

    for deform_type in DEFORM_TYPES:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for strategy_name in STRATEGIES:
            data = all_results[deform_type][strategy_name]
            ax.plot(frames, data["rmse_total"], "o-", label=strategy_name, markersize=4)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Total RMSE (px)")
        ax.set_title(f"RMSE growth — {deform_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"rmse_{deform_type}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir / f'rmse_{deform_type}.png'}")

    # Summary bar chart: mean RMSE across frames for each strategy × deformation
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(DEFORM_TYPES))
    width = 0.25
    for j, strategy_name in enumerate(STRATEGIES):
        means = [
            np.mean(all_results[dt][strategy_name]["rmse_total"])
            for dt in DEFORM_TYPES
        ]
        ax.bar(x + j * width, means, width, label=strategy_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(DEFORM_TYPES, rotation=15, ha="right")
    ax.set_ylabel("Mean RMSE (px)")
    ax.set_title("Mean RMSE by deformation type and strategy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'summary_bar.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate DIC tracking modes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/mode_comparison",
        help="Directory for output plots and tables",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AL-DIC Mode Comparison Evaluation")
    print(f"  {N_FRAMES} frames, {len(DEFORM_TYPES)} deformation types, "
          f"{len(STRATEGIES)} strategies")
    print("=" * 70)

    ref_img = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)
    mesh = make_mesh_for_image(IMG_H, IMG_W, STEP, MARGIN)

    all_results: dict[str, dict[str, dict]] = {}

    for deform_type in DEFORM_TYPES:
        print(f"\n--- Deformation: {deform_type} ---")
        all_results[deform_type] = {}

        for strategy_name, schedule in STRATEGIES.items():
            print(f"  Strategy: {strategy_name} ... ", end="", flush=True)
            data = run_evaluation(
                deform_type, strategy_name, schedule, ref_img, mesh,
            )
            all_results[deform_type][strategy_name] = data
            mean_rmse = np.mean(data["rmse_total"])
            print(f"done ({data['elapsed']:.1f}s, mean RMSE={mean_rmse:.4f} px)")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Mean Total RMSE (px)")
    print("=" * 70)
    header = f"{'Deformation':<15}" + "".join(
        f"  {s:<14}" for s in STRATEGIES
    )
    print(header)
    print("-" * len(header))
    for dt in DEFORM_TYPES:
        row = f"{dt:<15}"
        for sn in STRATEGIES:
            mean_rmse = np.mean(all_results[dt][sn]["rmse_total"])
            row += f"  {mean_rmse:<14.4f}"
        print(row)

    # Per-frame RMSE table
    print("\n" + "=" * 70)
    print("PER-FRAME TOTAL RMSE (px)")
    print("=" * 70)
    for dt in DEFORM_TYPES:
        print(f"\n  {dt}:")
        header = f"  {'Frame':<8}" + "".join(f"  {s:<14}" for s in STRATEGIES)
        print(header)
        for i in range(N_FRAMES - 1):
            row = f"  {i + 2:<8}"
            for sn in STRATEGIES:
                val = all_results[dt][sn]["rmse_total"][i]
                row += f"  {val:<14.4f}"
            print(row)

    # Save CSV
    csv_path = output_dir / "rmse_summary.csv"
    with open(csv_path, "w") as f:
        f.write("deformation,strategy,frame,rmse_u,rmse_v,rmse_total\n")
        for dt in DEFORM_TYPES:
            for sn in STRATEGIES:
                data = all_results[dt][sn]
                for i in range(N_FRAMES - 1):
                    f.write(
                        f"{dt},{sn},{i + 2},"
                        f"{data['rmse_u'][i]:.6f},"
                        f"{data['rmse_v'][i]:.6f},"
                        f"{data['rmse_total'][i]:.6f}\n"
                    )
    print(f"\nCSV saved to {csv_path}")

    # Generate plots
    print("\nGenerating RMSE plots...")
    plot_results(all_results, output_dir)

    print("\nGenerating full-field displacement maps...")
    plot_field_maps(all_results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
