"""Compare Local DIC vs AL-DIC on a 20px-amplitude quadratic field.

Single 2-frame test:
  ref  → deformed  (quadratic field, max |u|~20px)

Compares:
  1. FFT initial guess accuracy
  2. Local IC-GN only (no ADMM)
  3. AL-DIC (IC-GN + ADMM x3)

Reports timing, accuracy, convergence, and generates full-field plots.

NOTE: Uses Lagrangian image generation (fixed-point inversion) so that
u_field(X,Y) IS the correct ground truth at reference point (X,Y).
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

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.io.image_ops import compute_image_gradient, normalize_images
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.solver.integer_search import integer_search
from al_dic.solver.init_disp import init_disp
from al_dic.solver.local_icgn import local_icgn
from al_dic.solver.subpb1_solver import precompute_subpb1, subpb1_solver
from al_dic.solver.subpb2_solver import precompute_subpb2, subpb2_solver
from al_dic.utils.outlier_detection import detect_bad_points, fill_nan_idw
from al_dic.strain.nodal_strain_fem import global_nodal_strain_fem

OUT_DIR = Path("reports/local_vs_aldic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W = 1024, 1024
STEP = 4
WS = 16
FFT_SEARCH = 30  # ±30px to cover 20px max displacement


# ============================================================
# Helpers
# ============================================================

def make_speckle(h, w, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    f = gaussian_filter(noise, sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def apply_displacement_lagrangian(ref, u_field, v_field, n_iter=20):
    """Create deformed image using correct Lagrangian mapping.

    For each deformed pixel (x,y), finds reference pixel (X,Y) such that
    X + u(X,Y) = x, Y + v(X,Y) = y, then sets g(x,y) = f(X,Y).
    Uses fixed-point iteration to invert the displacement mapping.
    """
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    spl_u = RectBivariateSpline(np.arange(h), np.arange(w), u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(np.arange(h), np.arange(w), v_field, kx=3, ky=3)

    # Fixed-point iteration: X_{n+1} = x - u(X_n, Y_n)
    X = xx.copy()
    Y = yy.copy()
    for _ in range(n_iter):
        u_at = spl_u.ev(Y.ravel(), X.ravel()).reshape(h, w)
        v_at = spl_v.ev(Y.ravel(), X.ravel()).reshape(h, w)
        X_new = xx - u_at
        Y_new = yy - v_at
        if max(np.max(np.abs(X_new - X)), np.max(np.abs(Y_new - Y))) < 1e-10:
            break
        X, Y = X_new, Y_new

    coords = np.array([Y.ravel(), X.ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def gt_at_nodes(coords, u_field, v_field):
    h, w = u_field.shape
    spl_u = RectBivariateSpline(np.arange(h), np.arange(w), u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(np.arange(h), np.arange(w), v_field, kx=3, ky=3)
    return spl_u.ev(coords[:, 1], coords[:, 0]), spl_v.ev(coords[:, 1], coords[:, 0])


def compute_metrics(u_est, v_est, u_gt, v_gt, label=""):
    eu, ev = u_est - u_gt, v_est - v_gt
    mag = np.sqrt(eu**2 + ev**2)
    rmse = np.sqrt(np.nanmean(eu**2 + ev**2))
    maxe = np.nanmax(mag)
    u_rmse = np.sqrt(np.nanmean(eu**2))
    v_rmse = np.sqrt(np.nanmean(ev**2))
    u_max = np.nanmax(np.abs(eu))
    v_max = np.nanmax(np.abs(ev))
    return dict(rmse=rmse, maxe=maxe, u_rmse=u_rmse, v_rmse=v_rmse,
                u_max=u_max, v_max=v_max, eu=eu, ev=ev)


def plot_comparison(coords, u_gt, v_gt, local_m, aldic_m, fname):
    """6-panel comparison: GT, local result, AL-DIC result, local error, AL-DIC error."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle("Local DIC vs AL-DIC — 20px Quadratic Field", fontsize=13)

    # Row 1: u component
    for ax, vals, title in [
        (axes[0, 0], u_gt, "u ground truth (px)"),
        (axes[0, 1], local_m["eu"], f"Local u error (RMSE={local_m['u_rmse']:.4f})"),
        (axes[0, 2], aldic_m["eu"], f"AL-DIC u error (RMSE={aldic_m['u_rmse']:.4f})"),
    ]:
        vlim = max(abs(np.nanpercentile(vals, 2)), abs(np.nanpercentile(vals, 98)))
        if vlim == 0: vlim = 0.01
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim) if "error" in title else None
        vmin = np.nanpercentile(vals, 2) if norm is None else None
        vmax = np.nanpercentile(vals, 98) if norm is None else None
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=0.3, cmap="RdBu_r",
                        norm=norm, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, shrink=0.7)

    # Row 1 col 4: error magnitude comparison
    ax = axes[0, 3]
    local_mag = np.sqrt(local_m["eu"]**2 + local_m["ev"]**2)
    aldic_mag = np.sqrt(aldic_m["eu"]**2 + aldic_m["ev"]**2)
    vlim = np.nanpercentile(np.concatenate([local_mag, aldic_mag]), 98)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=local_mag - aldic_mag, s=0.3,
                    cmap="RdBu_r", vmin=-vlim/2, vmax=vlim/2, rasterized=True)
    ax.set_title("Local−ALDIC |error| (>0 = ALDIC better)", fontsize=9)
    ax.set_aspect("equal"); ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, shrink=0.7)

    # Row 2: v component
    for ax, vals, title in [
        (axes[1, 0], v_gt, "v ground truth (px)"),
        (axes[1, 1], local_m["ev"], f"Local v error (RMSE={local_m['v_rmse']:.4f})"),
        (axes[1, 2], aldic_m["ev"], f"AL-DIC v error (RMSE={aldic_m['v_rmse']:.4f})"),
    ]:
        vlim = max(abs(np.nanpercentile(vals, 2)), abs(np.nanpercentile(vals, 98)))
        if vlim == 0: vlim = 0.01
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim) if "error" in title else None
        vmin = np.nanpercentile(vals, 2) if norm is None else None
        vmax = np.nanpercentile(vals, 98) if norm is None else None
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=0.3, cmap="RdBu_r",
                        norm=norm, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, shrink=0.7)

    # Row 2 col 4: local |error|
    ax = axes[1, 3]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=local_mag, s=0.3,
                    cmap="hot_r", vmin=0, vmax=vlim, rasterized=True)
    ax.set_title(f"Local |error| (max={local_m['maxe']:.4f})", fontsize=9)
    ax.set_aspect("equal"); ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, shrink=0.7)

    # Row 3: strain comparison (dudx and dvdy)
    # Skip for now — just show AL-DIC |error|
    ax = axes[2, 0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=aldic_mag, s=0.3,
                    cmap="hot_r", vmin=0, vmax=vlim, rasterized=True)
    ax.set_title(f"AL-DIC |error| (max={aldic_m['maxe']:.4f})", fontsize=9)
    ax.set_aspect("equal"); ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, shrink=0.7)

    # Row 3 cols 1-3: histogram of errors
    ax = axes[2, 1]
    bins = np.linspace(0, max(local_m['maxe'], aldic_m['maxe']), 100)
    ax.hist(local_mag, bins=bins, alpha=0.6, label=f"Local (RMSE={local_m['rmse']:.4f})", color="C0")
    ax.hist(aldic_mag, bins=bins, alpha=0.6, label=f"AL-DIC (RMSE={aldic_m['rmse']:.4f})", color="C1")
    ax.set_xlabel("|error| (px)"); ax.set_ylabel("Count")
    ax.set_title("Error distribution"); ax.legend(fontsize=8)

    ax = axes[2, 2]
    # Percentile comparison
    pcts = [50, 75, 90, 95, 99, 100]
    local_pcts = np.nanpercentile(local_mag, pcts)
    aldic_pcts = np.nanpercentile(aldic_mag, pcts)
    x = np.arange(len(pcts))
    ax.bar(x - 0.17, local_pcts, 0.34, label="Local", color="C0", alpha=0.7)
    ax.bar(x + 0.17, aldic_pcts, 0.34, label="AL-DIC", color="C1", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f"P{p}" for p in pcts], fontsize=8)
    ax.set_ylabel("|error| (px)"); ax.set_title("Error percentiles")
    ax.legend(fontsize=8)

    axes[2, 3].axis("off")

    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fname}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Local DIC vs AL-DIC: 20px Quadratic Field")
    print(f"Image: {H}x{W}, step={STEP}, ws={WS}, FFT search=±{FFT_SEARCH}px")
    print("=" * 70)

    # Generate speckle
    ref = make_speckle(H, W, sigma=3.0, seed=42)

    # Quadratic field: max amplitude ~20px
    # u = A * ((x-cx)^2 + (y-cy)^2) / R^2 * cos(theta) + translation
    # Simpler: u = scale * ((x-cx)/cx)^2 * cx, so max ~scale*cx at edges
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    xn = (xx - cx) / cx  # [-1, 1]
    yn = (yy - cy) / cy

    # u: parabolic in x, max ±20px at edges; v: saddle, max ±10px
    # u = 20 * xn^2 - 10  → u(edge) = 20*1 - 10 = 10, u(center) = -10 → span 20px
    # Actually: u = 10 * (xn^2 + yn^2) → max at corners = 10*(1+1) = 20, center = 0
    # v = 8 * xn * yn → max at corners = 8, center = 0
    u_field = 10.0 * (xn**2 + yn**2)  # [0, 20] at corners
    v_field = 8.0 * xn * yn            # [-8, 8] at corners

    u_min, u_max = u_field.min(), u_field.max()
    v_min, v_max = v_field.min(), v_field.max()
    print(f"\nDisplacement field:")
    print(f"  u: [{u_min:.2f}, {u_max:.2f}] (span {u_max-u_min:.2f}px)")
    print(f"  v: [{v_min:.2f}, {v_max:.2f}] (span {v_max-v_min:.2f}px)")

    # Analytical strain (du/dx, du/dy, dv/dx, dv/dy in pixel coordinates)
    dudx_field = 20.0 * xn / cx   # = 20*(x-cx)/cx^2
    dudy_field = 20.0 * yn / cy
    dvdx_field = 8.0 * yn / cx
    dvdy_field = 8.0 * xn / cy
    exx_field = dudx_field
    eyy_field = dvdy_field
    exy_field = 0.5 * (dudy_field + dvdx_field)
    print(f"  exx: [{exx_field.min():.6f}, {exx_field.max():.6f}]")
    print(f"  eyy: [{eyy_field.min():.6f}, {eyy_field.max():.6f}]")
    print(f"  exy: [{exy_field.min():.6f}, {exy_field.max():.6f}]")

    # Deformed image (Lagrangian: correct GT convention)
    t0 = time.perf_counter()
    g_raw = apply_displacement_lagrangian(ref, u_field, v_field)
    t_img = time.perf_counter() - t0
    print(f"  Image generation: {t_img:.3f}s (Lagrangian, fixed-point iteration)")

    # DIC parameters
    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
        size_of_fft_search_region=FFT_SEARCH,
        show_plots=False,
    )

    # Normalize images
    imgs, clamped = normalize_images([ref, g_raw], roi)
    para = replace(para, gridxy_roi_range=clamped)
    f_mask = np.ones((H, W))
    Df = compute_image_gradient(imgs[0], f_mask)
    f_img, g_img = imgs[0], imgs[1]

    # ============================================================
    # Step 1: FFT integer search
    # ============================================================
    print(f"\n{'='*70}")
    print("Step 1: FFT Integer Search")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    x0, y0, u_grid, v_grid, fft_info = integer_search(f_img, g_img, para)
    t_fft = time.perf_counter() - t0

    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)

    # Build mesh from FFT grid
    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]
    n_ele = dic_mesh.elements_fem.shape[0]
    print(f"  Mesh: {n_nodes} nodes, {n_ele} elements")
    print(f"  FFT time: {t_fft:.3f}s")

    # Ground truth
    u_gt, v_gt = gt_at_nodes(coords, u_field, v_field)

    u_fft, v_fft = U_fft[0::2], U_fft[1::2]
    valid = ~np.isnan(u_fft)
    fft_met = compute_metrics(u_fft[valid], v_fft[valid], u_gt[valid], v_gt[valid])
    print(f"  Valid nodes: {valid.sum()}/{n_nodes}")
    print(f"  RMSE = {fft_met['rmse']:.4f}px,  max = {fft_met['maxe']:.4f}px")
    print(f"  u_RMSE = {fft_met['u_rmse']:.4f},  v_RMSE = {fft_met['v_rmse']:.4f}")

    # ============================================================
    # Step 2: Local IC-GN (no ADMM)
    # ============================================================
    print(f"\n{'='*70}")
    print("Step 2: Local IC-GN (no ADMM)")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    U_local, F_local, local_time, conv_iter, bad_count, mark_hole = local_icgn(
        U_fft.copy(), coords, Df, f_img, g_img, para, para.tol,
    )
    t_local = time.perf_counter() - t0

    # Bad node stats
    nan_mask = np.isnan(U_local[0::2])
    n_nan = nan_mask.sum()
    if n_nan > 0:
        U_local = fill_nan_idw(U_local, coords, n_components=2)
        F_local = fill_nan_idw(F_local, coords, n_components=4)

    u_local, v_local = U_local[0::2], U_local[1::2]
    local_met = compute_metrics(u_local, v_local, u_gt, v_gt)

    # Convergence stats
    ci = np.asarray(conv_iter).ravel()
    good_mask = (ci >= 1) & (ci <= para.icgn_max_iter)
    if good_mask.any():
        mean_iter = np.mean(ci[good_mask])
        max_iter_used = int(np.max(ci[good_mask]))
    else:
        mean_iter = max_iter_used = 0

    print(f"  Time:        {t_local:.3f}s (IC-GN kernel: {local_time:.3f}s)")
    print(f"  NaN nodes:   {n_nan}, bad_count: {bad_count}")
    print(f"  IC-GN iters: mean={mean_iter:.1f}, max={max_iter_used}")
    print(f"  RMSE = {local_met['rmse']:.4f}px,  max = {local_met['maxe']:.4f}px")
    print(f"  u_RMSE = {local_met['u_rmse']:.4f},  v_RMSE = {local_met['v_rmse']:.4f}")

    # ============================================================
    # Step 3: AL-DIC (IC-GN + ADMM iterations)
    # ============================================================
    print(f"\n{'='*70}")
    print("Step 3: AL-DIC (IC-GN + S5 + ADMM x3)")
    print(f"{'='*70}")

    mu_val = para.mu
    beta_val = 1e-3 * STEP**2 * mu_val
    alpha = 0.0

    # Precompute
    t0 = time.perf_counter()
    subpb1_pre = precompute_subpb1(coords, Df, f_img, para)
    t_pre_sp1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    subpb2_cache = precompute_subpb2(dic_mesh, para.gauss_pt_order, beta_val, mu_val, alpha)
    t_pre_sp2 = time.perf_counter() - t0

    print(f"  Precompute subpb1: {t_pre_sp1:.3f}s")
    print(f"  Precompute subpb2: {t_pre_sp2:.3f}s")

    U_s1 = U_local.copy()
    F_s1 = F_local.copy()

    # S5: first subpb2 + strain
    t0 = time.perf_counter()
    U_s2 = subpb2_solver(
        dic_mesh, para.gauss_pt_order, beta_val, mu_val,
        U_s1, F_s1, np.zeros(4*n_nodes), np.zeros(2*n_nodes),
        alpha, STEP, precomputed=subpb2_cache,
    )
    F_s2 = global_nodal_strain_fem(dic_mesh, para, U_s2)
    t_s5 = time.perf_counter() - t0
    print(f"  S5 (first subpb2+strain): {t_s5:.3f}s")

    u_s5, v_s5 = U_s2[0::2], U_s2[1::2]
    s5_met = compute_metrics(u_s5, v_s5, u_gt, v_gt)
    print(f"  After S5: RMSE={s5_met['rmse']:.4f}px, max={s5_met['maxe']:.4f}px")

    # ADMM loop
    grad_dual = F_s2 - F_s1
    disp_dual = U_s2 - U_s1
    winsize_list = np.full((n_nodes, 2), para.winsize, dtype=np.float64)
    para2 = replace(para, winsize_list=winsize_list)

    admm_history = []
    t_admm_total = 0.0

    for it in range(1, 4):
        t0 = time.perf_counter()

        # Subproblem 1
        U_s1_new, _, _, _ = subpb1_solver(
            U_s2, F_s2, disp_dual, grad_dual,
            coords, Df, f_img, g_img,
            mu_val, beta_val, para2, para.tol,
            precomputed=subpb1_pre,
        )
        F_s1_new = F_s2.copy()

        # Subproblem 2
        U_s2_new = subpb2_solver(
            dic_mesh, para.gauss_pt_order, beta_val, mu_val,
            U_s1_new, F_s1_new, grad_dual, disp_dual,
            alpha, STEP, precomputed=subpb2_cache,
        )
        F_s2_new = global_nodal_strain_fem(dic_mesh, para2, U_s2_new)

        t_iter = time.perf_counter() - t0
        t_admm_total += t_iter

        # Convergence norms
        g_norm = np.sqrt(np.mean((F_s2_new - F_s1_new)**2))
        d_norm = np.sqrt(np.mean((U_s2_new - U_s1_new)**2))
        conv_norm = max(g_norm, d_norm)

        # Update
        grad_dual = F_s2_new - F_s1_new
        disp_dual = U_s2_new - U_s1_new
        U_s1, F_s1 = U_s1_new, F_s1_new
        U_s2, F_s2 = U_s2_new, F_s2_new

        u_it, v_it = U_s2[0::2], U_s2[1::2]
        it_met = compute_metrics(u_it, v_it, u_gt, v_gt)

        admm_history.append(dict(
            iter=it, t=t_iter, conv=conv_norm,
            rmse=it_met["rmse"], maxe=it_met["maxe"],
        ))
        print(f"  ADMM iter {it}: conv={conv_norm:.4e}, "
              f"RMSE={it_met['rmse']:.4f}px, max={it_met['maxe']:.4f}px  ({t_iter:.3f}s)")

    # Final AL-DIC result
    u_aldic, v_aldic = U_s2[0::2], U_s2[1::2]
    aldic_met = compute_metrics(u_aldic, v_aldic, u_gt, v_gt)

    # Strain accuracy (AL-DIC only, using FEM strain from last iteration)
    dudx_py = F_s2[0::4]
    dvdx_py = F_s2[1::4]
    dudy_py = F_s2[2::4]
    dvdy_py = F_s2[3::4]

    exx_gt_n = gt_at_nodes(coords, exx_field, np.zeros_like(exx_field))[0]
    eyy_gt_n = gt_at_nodes(coords, eyy_field, np.zeros_like(eyy_field))[0]
    exy_gt_n = gt_at_nodes(coords, exy_field, np.zeros_like(exy_field))[0]

    # Interior mask (trim ws boundary)
    inner = (
        (coords[:, 0] > WS + STEP) & (coords[:, 0] < W - WS - STEP) &
        (coords[:, 1] > WS + STEP) & (coords[:, 1] < H - WS - STEP)
    )

    exx_rmse = np.sqrt(np.nanmean((dudx_py[inner] - exx_gt_n[inner])**2))
    eyy_rmse = np.sqrt(np.nanmean((dvdy_py[inner] - eyy_gt_n[inner])**2))
    exy_rmse = np.sqrt(np.nanmean((0.5*(dudy_py+dvdx_py)[inner] - exy_gt_n[inner])**2))

    # ============================================================
    # Summary
    # ============================================================
    t_aldic_total = t_local + t_pre_sp1 + t_pre_sp2 + t_s5 + t_admm_total

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Step':<25s} {'Time':>8s} {'RMSE':>10s} {'Max':>10s}")
    print("-" * 60)
    print(f"  {'FFT initial guess':<23s} {t_fft:8.3f}s {fft_met['rmse']:10.4f}px {fft_met['maxe']:10.4f}px")
    print(f"  {'Local IC-GN only':<23s} {t_local:8.3f}s {local_met['rmse']:10.4f}px {local_met['maxe']:10.4f}px")
    print(f"  {'After S5 (subpb2)':<23s} {t_s5:8.3f}s {s5_met['rmse']:10.4f}px {s5_met['maxe']:10.4f}px")
    for h in admm_history:
        print(f"  {'ADMM iter '+str(h['iter']):<23s} {h['t']:8.3f}s {h['rmse']:10.4f}px {h['maxe']:10.4f}px")
    print("-" * 60)
    print(f"  {'Local DIC total':<23s} {t_fft+t_local:8.3f}s {local_met['rmse']:10.4f}px {local_met['maxe']:10.4f}px")
    print(f"  {'AL-DIC total':<23s} {t_fft+t_aldic_total:8.3f}s {aldic_met['rmse']:10.4f}px {aldic_met['maxe']:10.4f}px")
    print("-" * 60)
    improvement = (local_met['rmse'] - aldic_met['rmse']) / local_met['rmse'] * 100
    print(f"  AL-DIC improvement: {improvement:.1f}% RMSE reduction")

    print(f"\nStrain accuracy (interior nodes, AL-DIC):")
    print(f"  exx RMSE: {exx_rmse:.6f}")
    print(f"  eyy RMSE: {eyy_rmse:.6f}")
    print(f"  exy RMSE: {exy_rmse:.6f}")

    # ============================================================
    # Generate comparison plot
    # ============================================================
    print(f"\nGenerating comparison plots...")
    plot_comparison(coords, u_gt, v_gt, local_met, aldic_met,
                    OUT_DIR / "comparison_20px_quadratic.png")

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = ["FFT", "IC-GN", "S5"] + [f"ADMM{h['iter']}" for h in admm_history]
    rmses = [fft_met["rmse"], local_met["rmse"], s5_met["rmse"]] + [h["rmse"] for h in admm_history]
    maxes = [fft_met["maxe"], local_met["maxe"], s5_met["maxe"]] + [h["maxe"] for h in admm_history]

    axes[0].plot(steps, rmses, "o-", color="C0", label="RMSE")
    axes[0].plot(steps, maxes, "s--", color="C1", label="Max error")
    axes[0].set_ylabel("Error (px)"); axes[0].set_title("Accuracy convergence")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # ADMM convergence norms
    if admm_history:
        axes[1].semilogy(
            [h["iter"] for h in admm_history],
            [h["conv"] for h in admm_history],
            "o-", color="C2",
        )
        axes[1].set_xlabel("ADMM iteration"); axes[1].set_ylabel("Convergence norm")
        axes[1].set_title("ADMM convergence"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {OUT_DIR / 'convergence.png'}")

    print(f"\nAll outputs: {OUT_DIR.resolve()}")
