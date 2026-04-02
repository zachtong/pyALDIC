#!/usr/bin/env python
"""Pipeline validation: translation, affine, quadratic deformation.

Runs three deformation types through the pipeline, captures intermediate
results (FFT → Local ICGN → ADMM), and generates a PDF report with
displacement heatmaps and error analysis.

Usage:
    python scripts/report_pipeline_validation.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

# ---------------------------------------------------------------------------
# STAQ-DIC imports
# ---------------------------------------------------------------------------
from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICPara, GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.io.image_ops import compute_image_gradient, normalize_images
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.solver.init_disp import init_disp
from staq_dic.solver.integer_search import integer_search
from staq_dic.solver.local_icgn import local_icgn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 512, 512
STEP = 16
SEED = 42
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_PATH = REPORT_DIR / "pipeline_validation.pdf"


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42) -> NDArray:
    """Synthetic speckle in [20, 235] range."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement_lagrangian(
    ref_image: NDArray, u_func, v_func, n_iter: int = 20,
) -> NDArray:
    """Generate deformed image via Lagrangian inversion (order=5 spline)."""
    h, w = ref_image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    coords = np.array([Y.ravel(), X.ravel()])
    warped = map_coordinates(ref_image, coords, order=5, mode="nearest")
    return warped.reshape(h, w)


# ---------------------------------------------------------------------------
# Deformation cases
# ---------------------------------------------------------------------------
CX, CY = IMG_W / 2.0, IMG_H / 2.0

CASES = [
    {
        "name": "Translation",
        "u_func": lambda x, y: np.full_like(x, 3.7),
        "v_func": lambda x, y: np.full_like(y, -2.3),
        "desc": "u = 3.7 px,  v = -2.3 px  (constant)",
        # du/dx, du/dy, dv/dx, dv/dy
        "F_gt": (0.0, 0.0, 0.0, 0.0),
    },
    {
        "name": "Affine",
        "u_func": lambda x, y: 0.020 * (x - CX) + 0.005 * (y - CY),
        "v_func": lambda x, y: -0.005 * (x - CX) + 0.015 * (y - CY),
        "desc": "du/dx=0.020, du/dy=0.005, dv/dx=-0.005, dv/dy=0.015",
        "F_gt": (0.020, 0.005, -0.005, 0.015),
    },
    {
        "name": "Quadratic",
        "u_func": lambda x, y: (
            1.5e-5 * (x - CX) ** 2 - 1.0e-5 * (y - CY) ** 2
            + 0.015 * (x - CX) + 1.0
        ),
        "v_func": lambda x, y: (
            -1.0e-5 * (x - CX) ** 2 + 2.0e-5 * (y - CY) ** 2
            + 0.010 * (y - CY) - 0.5
        ),
        "desc": "u = 1.5e-5*x^2 - 1e-5*y^2 + 0.015*x + 1\n"
                "v = -1e-5*x^2 + 2e-5*y^2 + 0.01*y - 0.5",
        "F_gt": None,  # spatially varying
    },
]


# ---------------------------------------------------------------------------
# DICPara construction
# ---------------------------------------------------------------------------
def make_para() -> DICPara:
    return dicpara_default(
        winsize=32,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(
            gridx=(STEP, IMG_W - 1 - STEP),
            gridy=(STEP, IMG_H - 1 - STEP),
        ),
        reference_mode="accumulative",
        show_plots=False,
        # ADMM settings
        admm_max_iter=20,
        admm_tol=1e-4,
    )


# ---------------------------------------------------------------------------
# Pipeline stages (manual calls to capture intermediates)
# ---------------------------------------------------------------------------
def run_fft_stage(f_norm, g_norm, para):
    """FFT integer search + init_disp → (U_fft, mesh, x0, y0)."""
    x0, y0, u_grid, v_grid, info = integer_search(f_norm, g_norm, para)
    U_fft = init_disp(u_grid, v_grid, info["cc_max"], x0, y0)
    mesh = mesh_setup(x0, y0, para)
    return U_fft, mesh, x0, y0


def run_icgn_stage(U_fft, mesh, Df, f_norm, g_norm, para):
    """Local IC-GN 6-DOF → U_icgn."""
    U_icgn, F_icgn, _, _, _, _ = local_icgn(
        U_fft.copy(), mesh.coordinates_fem, Df, f_norm, g_norm, para, para.tol,
    )
    return U_icgn


def run_admm_stage(ref_raw, def_raw, mask, para):
    """Full pipeline (FFT + ICGN + ADMM) → U_admm."""
    result = run_aldic(
        para, [ref_raw, def_raw], [mask, mask], compute_strain=False,
    )
    return result.result_disp[0].U, result.dic_mesh


# ---------------------------------------------------------------------------
# Ground truth at mesh nodes
# ---------------------------------------------------------------------------
def gt_at_nodes(u_func, v_func, coords):
    """Compute GT displacement at mesh node coordinates."""
    u_gt = u_func(coords[:, 0], coords[:, 1])
    v_gt = v_func(coords[:, 0], coords[:, 1])
    return u_gt, v_gt


def compute_rmse(U, u_gt, v_gt, edge_margin=48, coords=None, img_size=None):
    """RMSE on interior nodes (excluding edge_margin from borders)."""
    u = U[0::2]
    v = U[1::2]
    h, w = img_size
    x, y = coords[:, 0], coords[:, 1]
    interior = (
        (x >= edge_margin) & (x <= w - 1 - edge_margin)
        & (y >= edge_margin) & (y <= h - 1 - edge_margin)
    )
    valid = interior & np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        return np.inf, np.inf
    eu = u[valid] - u_gt[valid]
    ev = v[valid] - v_gt[valid]
    return float(np.sqrt(np.mean(eu**2))), float(np.sqrt(np.mean(ev**2)))


# ---------------------------------------------------------------------------
# Reshape helper: interleaved (2*N,) → 2D grids for imshow
# ---------------------------------------------------------------------------
def to_grid(vals_1d, nx, ny):
    """Reshape (nx*ny,) → (ny, nx) suitable for imshow (rows=y, cols=x).

    mesh_setup uses indexing='ij': node_idx = ix * ny + iy.
    So vals_1d.reshape(nx, ny)[ix, iy] has x varying along axis-0.
    Transpose to get (ny, nx) with rows=y, cols=x.
    """
    return vals_1d.reshape(nx, ny).T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def add_heatmap(ax, data, title, cmap, vmin=None, vmax=None, extent=None):
    """Single imshow panel with colorbar."""
    im = ax.imshow(
        data, cmap=cmap, origin="upper", aspect="equal",
        vmin=vmin, vmax=vmax, extent=extent,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    return im


def plot_case_page(pdf, case_name, desc, x0, y0,
                   u_gt, v_gt, results, rmse_table):
    """One page per deformation type: 3 cols × 4 rows.

    Rows: u-displacement, u-error, v-displacement, v-error.
    Cols: FFT, Local ICGN, ADMM.
    """
    nx, ny = len(x0), len(y0)
    ext = [x0[0], x0[-1], y0[-1], y0[0]]  # imshow extent (left,right,bottom,top)

    # Ground truth grids
    u_gt_2d = to_grid(u_gt, nx, ny)
    v_gt_2d = to_grid(v_gt, nx, ny)

    # Prepare grids for each stage
    stage_names = ["FFT", "Local ICGN", "ADMM"]
    u_grids, v_grids, eu_grids, ev_grids = [], [], [], []
    for sname in stage_names:
        U = results[sname]
        u_2d = to_grid(U[0::2], nx, ny)
        v_2d = to_grid(U[1::2], nx, ny)
        u_grids.append(u_2d)
        v_grids.append(v_2d)
        eu_grids.append(u_2d - u_gt_2d)
        ev_grids.append(v_2d - v_gt_2d)

    # Common color ranges (across stages, per component)
    all_u = np.concatenate([u_gt_2d.ravel()] + [g.ravel() for g in u_grids])
    all_v = np.concatenate([v_gt_2d.ravel()] + [g.ravel() for g in v_grids])
    all_u = all_u[np.isfinite(all_u)]
    all_v = all_v[np.isfinite(all_v)]
    u_vmin, u_vmax = np.percentile(all_u, [1, 99])
    v_vmin, v_vmax = np.percentile(all_v, [1, 99])

    # Error range (symmetric around 0)
    all_eu = np.concatenate([g.ravel() for g in eu_grids])
    all_ev = np.concatenate([g.ravel() for g in ev_grids])
    all_eu = all_eu[np.isfinite(all_eu)]
    all_ev = all_ev[np.isfinite(all_ev)]
    e_abs_max = max(
        np.percentile(np.abs(all_eu), 99) if len(all_eu) > 0 else 0.1,
        np.percentile(np.abs(all_ev), 99) if len(all_ev) > 0 else 0.1,
        0.01,  # minimum range to avoid degenerate colorbar
    )

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    fig.suptitle(
        f"{case_name} Deformation\n{desc}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Column 0: Ground Truth
    add_heatmap(axes[0, 0], u_gt_2d, "GT  u (px)", "viridis",
                u_vmin, u_vmax, ext)
    add_heatmap(axes[2, 0], v_gt_2d, "GT  v (px)", "viridis",
                v_vmin, v_vmax, ext)
    # Leave error slots blank for GT column
    for row in [1, 3]:
        axes[row, 0].axis("off")

    # Columns 1-3: FFT, ICGN, ADMM
    for col, sname in enumerate(stage_names, start=1):
        rmse_u, rmse_v = rmse_table[sname]
        add_heatmap(axes[0, col], u_grids[col - 1],
                    f"{sname}  u (px)", "viridis", u_vmin, u_vmax, ext)
        add_heatmap(axes[1, col], eu_grids[col - 1],
                    f"{sname}  u error  [RMSE={rmse_u:.4f}]",
                    "RdBu_r", -e_abs_max, e_abs_max, ext)
        add_heatmap(axes[2, col], v_grids[col - 1],
                    f"{sname}  v (px)", "viridis", v_vmin, v_vmax, ext)
        add_heatmap(axes[3, col], ev_grids[col - 1],
                    f"{sname}  v error  [RMSE={rmse_v:.4f}]",
                    "RdBu_r", -e_abs_max, e_abs_max, ext)

    # Row labels
    for row, label in enumerate(["u disp", "u error", "v disp", "v error"]):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def plot_summary_page(pdf, all_rmse):
    """Final summary table of RMSE values across all cases and stages."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    ax.set_title("Pipeline Validation — RMSE Summary (px)", fontsize=14,
                 fontweight="bold", pad=20)

    col_labels = ["Deformation", "Component",
                  "FFT", "Local ICGN", "ADMM",
                  "ICGN/FFT", "ADMM/FFT"]
    rows = []
    for case_name, rmse_dict in all_rmse.items():
        for comp, comp_idx in [("u", 0), ("v", 1)]:
            fft_r = rmse_dict["FFT"][comp_idx]
            icgn_r = rmse_dict["Local ICGN"][comp_idx]
            admm_r = rmse_dict["ADMM"][comp_idx]
            ratio_icgn = icgn_r / fft_r if fft_r > 0 else 0.0
            ratio_admm = admm_r / fft_r if fft_r > 0 else 0.0
            rows.append([
                case_name, comp,
                f"{fft_r:.5f}", f"{icgn_r:.5f}", f"{admm_r:.5f}",
                f"{ratio_icgn:.3f}", f"{ratio_admm:.3f}",
            ])

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i, row in enumerate(rows, start=1):
        color = "#ecf0f1" if (i - 1) // 2 % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    para = make_para()

    # Generate reference image
    print(f"Generating {IMG_H}x{IMG_W} speckle image (seed={SEED})...")
    ref_raw = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=SEED)
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)

    all_rmse: dict[str, dict] = {}

    with PdfPages(str(REPORT_PATH)) as pdf:
        for case in CASES:
            case_name = case["name"]
            u_func, v_func = case["u_func"], case["v_func"]
            print(f"\n{'='*60}")
            print(f"  {case_name}: {case['desc']}")
            print(f"{'='*60}")

            # --- Generate deformed image ---
            t0 = time.perf_counter()
            def_raw = apply_displacement_lagrangian(ref_raw, u_func, v_func)
            print(f"  Image generation: {time.perf_counter() - t0:.2f}s")

            # --- Normalize (replicate pipeline Section 2b) ---
            img_norm, clamped_roi = normalize_images(
                [ref_raw, def_raw], para.gridxy_roi_range,
            )
            para_local = replace(
                para, gridxy_roi_range=clamped_roi, img_size=(IMG_H, IMG_W),
            )
            f_norm = img_norm[0] * mask
            g_norm = img_norm[1] * mask

            # --- Stage 1: FFT ---
            t0 = time.perf_counter()
            U_fft, mesh, x0, y0 = run_fft_stage(f_norm, g_norm, para_local)
            t_fft = time.perf_counter() - t0
            print(f"  FFT:  {t_fft:.3f}s  ({mesh.coordinates_fem.shape[0]} nodes)")

            # --- Stage 2: Local ICGN ---
            Df = compute_image_gradient(f_norm, mask, img_raw=img_norm[0])
            t0 = time.perf_counter()
            U_icgn = run_icgn_stage(
                U_fft, mesh, Df, img_norm[0], g_norm, para_local,
            )
            t_icgn = time.perf_counter() - t0
            print(f"  ICGN: {t_icgn:.3f}s")

            # --- Stage 3: Full pipeline (ADMM) ---
            t0 = time.perf_counter()
            U_admm, admm_mesh = run_admm_stage(
                ref_raw, def_raw, mask, para_local,
            )
            t_admm = time.perf_counter() - t0
            print(f"  ADMM: {t_admm:.3f}s")

            # --- Ground truth at mesh nodes ---
            coords = mesh.coordinates_fem
            u_gt, v_gt = gt_at_nodes(u_func, v_func, coords)

            # --- RMSE ---
            results = {"FFT": U_fft, "Local ICGN": U_icgn, "ADMM": U_admm}
            rmse_table = {}
            for sname, U in results.items():
                ru, rv = compute_rmse(
                    U, u_gt, v_gt,
                    edge_margin=48, coords=coords, img_size=(IMG_H, IMG_W),
                )
                rmse_table[sname] = (ru, rv)
                print(f"  {sname:12s}  RMSE_u={ru:.5f}  RMSE_v={rv:.5f}")

            all_rmse[case_name] = rmse_table

            # --- Plot ---
            plot_case_page(
                pdf, case_name, case["desc"],
                x0, y0, u_gt, v_gt, results, rmse_table,
            )

        # --- Summary page ---
        plot_summary_page(pdf, all_rmse)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
