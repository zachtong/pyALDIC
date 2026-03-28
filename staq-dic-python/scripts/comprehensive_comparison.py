"""Comprehensive Local DIC vs AL-DIC comparison across multiple deformation fields.

Tests:
  1. Small affine  (max ~3px)
  2. Large affine  (max ~15px)
  3. Small quadratic (max ~5px)
  4. Large quadratic (max ~20px)
  5. Combined affine + quadratic (max ~15px)
  6. Pure rotation (~2 degrees)

For each case, outputs full-field plots showing:
  - Ground truth, FFT, Local DIC, AL-DIC displacement fields
  - Error maps and histograms

Uses Lagrangian image generation for correct ground truth.
"""

import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RectBivariateSpline
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.io.image_ops import compute_image_gradient, normalize_images
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.solver.integer_search import integer_search
from staq_dic.solver.init_disp import init_disp
from staq_dic.solver.local_icgn import local_icgn
from staq_dic.solver.subpb1_solver import precompute_subpb1, subpb1_solver
from staq_dic.solver.subpb2_solver import precompute_subpb2, subpb2_solver
from staq_dic.solver.outlier_detection import fill_nan_rbf
from staq_dic.strain.nodal_strain_fem import global_nodal_strain_fem

OUT_DIR = Path("outputs/comprehensive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W = 1024, 1024
STEP = 4
WS = 16
FFT_SEARCH = 34


# ============================================================
# Image generation
# ============================================================

def make_speckle(h, w, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((h, w)), sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def apply_displacement_lagrangian(ref, u_func, v_func, n_iter=20):
    """Correct Lagrangian image generation via fixed-point inversion."""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        u_at = u_func(X, Y)
        v_at = v_func(X, Y)
        X_new = xx - u_at
        Y_new = yy - v_at
        if max(np.max(np.abs(X_new - X)), np.max(np.abs(Y_new - Y))) < 1e-10:
            break
        X, Y = X_new, Y_new
    coords = np.array([Y.ravel(), X.ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


# ============================================================
# Test case definitions
# ============================================================

def define_test_cases():
    """Return list of test cases, each with u_func, v_func, label, description."""
    cx, cy = W / 2.0, H / 2.0
    cases = []

    # Case 1: Small affine (max ~3px)
    # u = 0.003*x + 0.002*y (relative to center), v = -0.001*x + 0.004*y
    cases.append(dict(
        label="small_affine",
        title="Small Affine (max ~3px)",
        u_func=lambda x, y: 0.003 * (x - cx) + 0.002 * (y - cy),
        v_func=lambda x, y: -0.001 * (x - cx) + 0.004 * (y - cy),
    ))

    # Case 2: Large affine (max ~15px)
    cases.append(dict(
        label="large_affine",
        title="Large Affine (max ~15px)",
        u_func=lambda x, y: 0.020 * (x - cx) + 0.008 * (y - cy),
        v_func=lambda x, y: -0.005 * (x - cx) + 0.015 * (y - cy),
    ))

    # Case 3: Small quadratic (max ~5px)
    cases.append(dict(
        label="small_quadratic",
        title="Small Quadratic (max ~5px)",
        u_func=lambda x, y: 2.5 * ((x - cx) / cx) ** 2 + 2.5 * ((y - cy) / cy) ** 2,
        v_func=lambda x, y: 2.0 * ((x - cx) / cx) * ((y - cy) / cy),
    ))

    # Case 4: Large quadratic (max ~20px)
    cases.append(dict(
        label="large_quadratic",
        title="Large Quadratic (max ~20px)",
        u_func=lambda x, y: 10.0 * (((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2),
        v_func=lambda x, y: 8.0 * ((x - cx) / cx) * ((y - cy) / cy),
    ))

    # Case 5: Combined affine + quadratic (max ~15px)
    cases.append(dict(
        label="combined",
        title="Affine + Quadratic (max ~15px)",
        u_func=lambda x, y: (0.01 * (x - cx)
                              + 5.0 * ((x - cx) / cx) ** 2
                              + 3.0 * ((y - cy) / cy) ** 2),
        v_func=lambda x, y: (0.005 * (y - cy)
                              + 4.0 * ((x - cx) / cx) * ((y - cy) / cy)),
    ))

    # Case 6: Pure rotation ~2 degrees
    theta = 2.0 * np.pi / 180.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cases.append(dict(
        label="rotation_2deg",
        title="Pure Rotation 2° (max ~18px)",
        u_func=lambda x, y: (cos_t - 1) * (x - cx) - sin_t * (y - cy),
        v_func=lambda x, y: sin_t * (x - cx) + (cos_t - 1) * (y - cy),
    ))

    return cases


# ============================================================
# DIC pipeline runner
# ============================================================

def run_dic_pipeline(ref, g_img, case_label):
    """Run FFT -> Local IC-GN -> AL-DIC. Return all results."""
    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
        size_of_fft_search_region=FFT_SEARCH, show_plots=False,
    )
    imgs, clamped = normalize_images([ref, g_img], roi)
    para = replace(para, gridxy_roi_range=clamped)
    f_mask = np.ones((H, W))
    Df = compute_image_gradient(imgs[0], f_mask)
    f_img, g_img_n = imgs[0], imgs[1]

    # FFT
    t0 = time.perf_counter()
    x0, y0, u_grid, v_grid, fft_info = integer_search(f_img, g_img_n, para)
    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    t_fft = time.perf_counter() - t0

    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]

    # Local IC-GN
    t0 = time.perf_counter()
    U_local, F_local, local_time, conv_iter, bad_count, mark_hole = local_icgn(
        U_fft.copy(), coords, Df, f_img, g_img_n, para, para.tol,
    )
    t_local = time.perf_counter() - t0

    if np.isnan(U_local).any():
        U_local = fill_nan_rbf(U_local, coords, n_components=2)
        F_local = fill_nan_rbf(F_local, coords, n_components=4)

    # AL-DIC: S5 + ADMM x3
    mu_val = para.mu
    beta_val = 1e-3 * STEP ** 2 * mu_val
    alpha = 0.0

    t0 = time.perf_counter()
    subpb1_pre = precompute_subpb1(coords, Df, f_img, para)
    subpb2_cache = precompute_subpb2(dic_mesh, para.gauss_pt_order, beta_val, mu_val, alpha)

    # S5
    U_s2 = subpb2_solver(
        dic_mesh, para.gauss_pt_order, beta_val, mu_val,
        U_local, F_local, np.zeros(4 * n_nodes), np.zeros(2 * n_nodes),
        alpha, STEP, precomputed=subpb2_cache,
    )
    F_s2 = global_nodal_strain_fem(dic_mesh, para, U_s2)

    U_s1 = U_local.copy()
    F_s1 = F_local.copy()
    grad_dual = F_s2 - F_s1
    disp_dual = U_s2 - U_s1
    winsize_list = np.full((n_nodes, 2), para.winsize, dtype=np.float64)
    para2 = replace(para, winsize_list=winsize_list)

    admm_rmse = []
    for it in range(1, 4):
        U_s1_new, _, _, _ = subpb1_solver(
            U_s2, F_s2, disp_dual, grad_dual,
            coords, Df, f_img, g_img_n,
            mu_val, beta_val, para2, para.tol,
            precomputed=subpb1_pre,
        )
        F_s1_new = F_s2.copy()
        U_s2_new = subpb2_solver(
            dic_mesh, para.gauss_pt_order, beta_val, mu_val,
            U_s1_new, F_s1_new, grad_dual, disp_dual,
            alpha, STEP, precomputed=subpb2_cache,
        )
        F_s2_new = global_nodal_strain_fem(dic_mesh, para2, U_s2_new)
        grad_dual = F_s2_new - F_s1_new
        disp_dual = U_s2_new - U_s1_new
        U_s1, F_s1 = U_s1_new, F_s1_new
        U_s2, F_s2 = U_s2_new, F_s2_new

    t_aldic = time.perf_counter() - t0

    # Convergence info
    ci = np.asarray(conv_iter).ravel()
    good = (ci >= 1) & (ci <= para.icgn_max_iter)
    mean_iter = np.mean(ci[good]) if good.any() else 0
    max_iter_used = int(np.max(ci[good])) if good.any() else 0

    return dict(
        coords=coords,
        u_fft=U_fft[0::2], v_fft=U_fft[1::2],
        u_local=U_local[0::2], v_local=U_local[1::2],
        u_aldic=U_s2[0::2], v_aldic=U_s2[1::2],
        t_fft=t_fft, t_local=t_local, t_aldic=t_aldic,
        mean_iter=mean_iter, max_iter=max_iter_used,
        bad_count=bad_count,
    )


# ============================================================
# Plotting
# ============================================================

def plot_full_comparison(coords, u_gt, v_gt, res, case_title, fname):
    """Full-field comparison: GT, FFT, Local, ALDIC for u and v + error maps."""
    fig, axes = plt.subplots(4, 5, figsize=(26, 20))
    fig.suptitle(case_title, fontsize=15, fontweight="bold", y=0.98)

    s = 0.3  # scatter point size

    def _scatter(ax, vals, title, cmap="RdBu_r", symmetric=False, vlim=None):
        if vlim is None:
            p2, p98 = np.nanpercentile(vals, [2, 98])
        else:
            p2, p98 = -vlim, vlim
        if symmetric:
            vlim_abs = max(abs(p2), abs(p98), 1e-6)
            norm = TwoSlopeNorm(vmin=-vlim_abs, vcenter=0, vmax=vlim_abs)
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=s,
                            cmap=cmap, norm=norm, rasterized=True)
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=s,
                            cmap=cmap, vmin=p2, vmax=p98, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)

    # Compute errors
    eu_fft = res["u_fft"] - u_gt
    ev_fft = res["v_fft"] - v_gt
    eu_local = res["u_local"] - u_gt
    ev_local = res["v_local"] - v_gt
    eu_aldic = res["u_aldic"] - u_gt
    ev_aldic = res["v_aldic"] - v_gt

    mag_fft = np.sqrt(eu_fft ** 2 + ev_fft ** 2)
    mag_local = np.sqrt(eu_local ** 2 + ev_local ** 2)
    mag_aldic = np.sqrt(eu_aldic ** 2 + ev_aldic ** 2)

    rmse_fft = np.sqrt(np.nanmean(mag_fft ** 2))
    rmse_local = np.sqrt(np.nanmean(mag_local ** 2))
    rmse_aldic = np.sqrt(np.nanmean(mag_aldic ** 2))

    # Row 0: u-displacement (GT, FFT, Local, ALDIC, ALDIC error)
    _scatter(axes[0, 0], u_gt, "u Ground Truth (px)")
    _scatter(axes[0, 1], res["u_fft"], "u FFT")
    _scatter(axes[0, 2], res["u_local"], "u Local DIC")
    _scatter(axes[0, 3], res["u_aldic"], "u AL-DIC")

    err_vlim = max(np.nanpercentile(np.abs(eu_local), 98),
                   np.nanpercentile(np.abs(eu_aldic), 98), 0.01)
    _scatter(axes[0, 4], eu_aldic,
             f"u AL-DIC error (RMSE={np.sqrt(np.nanmean(eu_aldic**2)):.4f})",
             symmetric=True, vlim=err_vlim)

    # Row 1: v-displacement (GT, FFT, Local, ALDIC, ALDIC error)
    _scatter(axes[1, 0], v_gt, "v Ground Truth (px)")
    _scatter(axes[1, 1], res["v_fft"], "v FFT")
    _scatter(axes[1, 2], res["v_local"], "v Local DIC")
    _scatter(axes[1, 3], res["v_aldic"], "v AL-DIC")

    err_vlim_v = max(np.nanpercentile(np.abs(ev_local), 98),
                     np.nanpercentile(np.abs(ev_aldic), 98), 0.01)
    _scatter(axes[1, 4], ev_aldic,
             f"v AL-DIC error (RMSE={np.sqrt(np.nanmean(ev_aldic**2)):.4f})",
             symmetric=True, vlim=err_vlim_v)

    # Row 2: Error magnitude maps (FFT, Local, ALDIC) + Local error u + v
    mag_vlim = max(np.nanpercentile(mag_fft, 98),
                   np.nanpercentile(mag_local, 98),
                   np.nanpercentile(mag_aldic, 98), 0.01)
    _scatter(axes[2, 0], mag_fft,
             f"|err| FFT (RMSE={rmse_fft:.4f})", cmap="hot_r", vlim=mag_vlim)
    _scatter(axes[2, 1], mag_local,
             f"|err| Local (RMSE={rmse_local:.4f})", cmap="hot_r", vlim=mag_vlim)
    _scatter(axes[2, 2], mag_aldic,
             f"|err| ALDIC (RMSE={rmse_aldic:.4f})", cmap="hot_r", vlim=mag_vlim)

    _scatter(axes[2, 3], eu_local,
             f"u Local error (RMSE={np.sqrt(np.nanmean(eu_local**2)):.4f})",
             symmetric=True, vlim=err_vlim)
    _scatter(axes[2, 4], ev_local,
             f"v Local error (RMSE={np.sqrt(np.nanmean(ev_local**2)):.4f})",
             symmetric=True, vlim=err_vlim_v)

    # Row 3: Histograms + summary table
    ax = axes[3, 0]
    bins = np.linspace(0, max(mag_fft.max(), mag_local.max(), mag_aldic.max()), 80)
    ax.hist(mag_fft, bins=bins, alpha=0.5, label=f"FFT ({rmse_fft:.4f})", color="C2")
    ax.hist(mag_local, bins=bins, alpha=0.5, label=f"Local ({rmse_local:.4f})", color="C0")
    ax.hist(mag_aldic, bins=bins, alpha=0.5, label=f"ALDIC ({rmse_aldic:.4f})", color="C1")
    ax.set_xlabel("|error| (px)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution (RMSE)")
    ax.legend(fontsize=7)

    ax = axes[3, 1]
    pcts = [50, 75, 90, 95, 99]
    fft_p = np.nanpercentile(mag_fft, pcts)
    loc_p = np.nanpercentile(mag_local, pcts)
    ald_p = np.nanpercentile(mag_aldic, pcts)
    x = np.arange(len(pcts))
    ax.bar(x - 0.25, fft_p, 0.25, label="FFT", color="C2", alpha=0.7)
    ax.bar(x, loc_p, 0.25, label="Local", color="C0", alpha=0.7)
    ax.bar(x + 0.25, ald_p, 0.25, label="ALDIC", color="C1", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in pcts], fontsize=8)
    ax.set_ylabel("|error| (px)")
    ax.set_title("Error percentiles")
    ax.legend(fontsize=7)

    # Summary text
    ax = axes[3, 2]
    ax.axis("off")
    improve_pct = ((rmse_local - rmse_aldic) / rmse_local * 100
                   if rmse_local > 1e-10 else 0)
    summary = (
        f"RMSE (px):\n"
        f"  FFT:   {rmse_fft:.4f}\n"
        f"  Local: {rmse_local:.4f}\n"
        f"  ALDIC: {rmse_aldic:.4f}\n\n"
        f"Max error (px):\n"
        f"  FFT:   {mag_fft.max():.4f}\n"
        f"  Local: {mag_local.max():.4f}\n"
        f"  ALDIC: {mag_aldic.max():.4f}\n\n"
        f"ALDIC improvement: {improve_pct:.1f}%\n\n"
        f"Timing:\n"
        f"  FFT:   {res['t_fft']:.2f}s\n"
        f"  IC-GN: {res['t_local']:.2f}s\n"
        f"  ALDIC: {res['t_aldic']:.2f}s\n"
        f"  Total: {res['t_fft']+res['t_local']+res['t_aldic']:.2f}s\n\n"
        f"IC-GN iters: mean={res['mean_iter']:.1f}, max={res['max_iter']}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Local-ALDIC improvement map
    _scatter(axes[3, 3], mag_local - mag_aldic,
             "Local−ALDIC |err| (>0 = ALDIC better)",
             symmetric=True)

    axes[3, 4].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(all_results, fname):
    """Summary bar chart comparing all cases."""
    n = len(all_results)
    labels = [r["label"] for r in all_results]
    rmse_fft = [r["rmse_fft"] for r in all_results]
    rmse_local = [r["rmse_local"] for r in all_results]
    rmse_aldic = [r["rmse_aldic"] for r in all_results]
    max_local = [r["max_local"] for r in all_results]
    max_aldic = [r["max_aldic"] for r in all_results]
    t_local = [r["t_local_total"] for r in all_results]
    t_aldic = [r["t_aldic_total"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Comprehensive Comparison: Local DIC vs AL-DIC", fontsize=13)

    x = np.arange(n)
    w = 0.25

    # RMSE comparison
    ax = axes[0]
    ax.bar(x - w, rmse_fft, w, label="FFT", color="C2", alpha=0.7)
    ax.bar(x, rmse_local, w, label="Local DIC", color="C0", alpha=0.7)
    ax.bar(x + w, rmse_aldic, w, label="AL-DIC", color="C1", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (px)")
    ax.set_title("Displacement RMSE")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Max error comparison
    ax = axes[1]
    ax.bar(x - 0.15, max_local, 0.3, label="Local DIC", color="C0", alpha=0.7)
    ax.bar(x + 0.15, max_aldic, 0.3, label="AL-DIC", color="C1", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Max |error| (px)")
    ax.set_title("Max Displacement Error")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Timing comparison
    ax = axes[2]
    ax.bar(x - 0.15, t_local, 0.3, label="Local DIC", color="C0", alpha=0.7)
    ax.bar(x + 0.15, t_aldic, 0.3, label="AL-DIC", color="C1", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title("Computation Time")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ref = make_speckle(H, W, sigma=3.0, seed=42)
    cx, cy = W / 2.0, H / 2.0
    cases = define_test_cases()

    all_results = []

    for ci, case in enumerate(cases):
        label = case["label"]
        title = case["title"]
        u_func = case["u_func"]
        v_func = case["v_func"]

        print(f"\n{'=' * 70}")
        print(f"  Case {ci+1}/{len(cases)}: {title}")
        print(f"{'=' * 70}")

        # Check displacement range
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        u_field = u_func(xx, yy)
        v_field = v_func(xx, yy)
        print(f"  u range: [{u_field.min():.2f}, {u_field.max():.2f}] px")
        print(f"  v range: [{v_field.min():.2f}, {v_field.max():.2f}] px")
        max_disp = max(abs(u_field).max(), abs(v_field).max())
        print(f"  Max |disp|: {max_disp:.2f} px")

        if max_disp > FFT_SEARCH - 2:
            print(f"  WARNING: max disp ({max_disp:.1f}) near FFT search range ({FFT_SEARCH})")

        # Generate deformed image
        t0 = time.perf_counter()
        g_img = apply_displacement_lagrangian(ref, u_func, v_func)
        t_gen = time.perf_counter() - t0
        print(f"  Image generation: {t_gen:.2f}s")

        # Run DIC pipeline
        res = run_dic_pipeline(ref, g_img, label)
        coords = res["coords"]

        # Ground truth at nodes
        u_gt = u_func(coords[:, 0], coords[:, 1])
        v_gt = v_func(coords[:, 0], coords[:, 1])

        # Metrics
        mag_fft = np.sqrt((res["u_fft"] - u_gt) ** 2 + (res["v_fft"] - v_gt) ** 2)
        mag_local = np.sqrt((res["u_local"] - u_gt) ** 2 + (res["v_local"] - v_gt) ** 2)
        mag_aldic = np.sqrt((res["u_aldic"] - u_gt) ** 2 + (res["v_aldic"] - v_gt) ** 2)

        rmse_fft = np.sqrt(np.nanmean(mag_fft ** 2))
        rmse_local = np.sqrt(np.nanmean(mag_local ** 2))
        rmse_aldic = np.sqrt(np.nanmean(mag_aldic ** 2))
        improve = (rmse_local - rmse_aldic) / rmse_local * 100 if rmse_local > 1e-10 else 0

        print(f"\n  Results:")
        print(f"    {'':12s} {'RMSE':>10s} {'Max':>10s} {'Time':>8s}")
        print(f"    {'FFT':12s} {rmse_fft:10.4f} {mag_fft.max():10.4f} {res['t_fft']:8.2f}s")
        print(f"    {'Local DIC':12s} {rmse_local:10.4f} {mag_local.max():10.4f} {res['t_local']:8.2f}s")
        print(f"    {'AL-DIC':12s} {rmse_aldic:10.4f} {mag_aldic.max():10.4f} {res['t_aldic']:8.2f}s")
        print(f"    ALDIC improvement: {improve:.1f}% RMSE reduction")
        print(f"    IC-GN: mean={res['mean_iter']:.1f} iter, max={res['max_iter']}")

        # Plot
        plot_full_comparison(
            coords, u_gt, v_gt, res, title,
            OUT_DIR / f"case{ci+1}_{label}.png",
        )
        print(f"  Plot saved: {OUT_DIR / f'case{ci+1}_{label}.png'}")

        all_results.append(dict(
            label=title,
            rmse_fft=rmse_fft, rmse_local=rmse_local, rmse_aldic=rmse_aldic,
            max_local=mag_local.max(), max_aldic=mag_aldic.max(),
            t_local_total=res["t_fft"] + res["t_local"],
            t_aldic_total=res["t_fft"] + res["t_local"] + res["t_aldic"],
            improve=improve,
        ))

    # Summary
    print(f"\n\n{'=' * 80}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Case':<30s} {'FFT':>8s} {'Local':>8s} {'ALDIC':>8s} {'Improve':>8s} {'Time(L)':>8s} {'Time(A)':>8s}")
    print("-" * 80)
    for r in all_results:
        print(f"  {r['label']:<28s} {r['rmse_fft']:8.4f} {r['rmse_local']:8.4f} "
              f"{r['rmse_aldic']:8.4f} {r['improve']:7.1f}% "
              f"{r['t_local_total']:7.1f}s {r['t_aldic_total']:7.1f}s")

    # Summary plot
    plot_summary_table(all_results, OUT_DIR / "summary_comparison.png")
    print(f"\nSummary plot: {OUT_DIR / 'summary_comparison.png'}")
    print(f"All outputs: {OUT_DIR.resolve()}")
