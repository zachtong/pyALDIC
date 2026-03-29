"""Step-by-step diagnostic for each deformation case (2-frame tests).

Each case: ref -> deformed (independent pair), no multi-frame contamination.
Measures accuracy at:
  Step 1 - FFT integer search initial guess
  Step 2 - After local IC-GN (S4)
  Step 3 - After ADMM S5+S6 (3 iterations)
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
from staq_dic.utils.outlier_detection import detect_bad_points, fill_nan_idw
from staq_dic.strain.nodal_strain_fem import global_nodal_strain_fem

OUT_DIR = Path("outputs/diagnose_icgn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W = 1024, 1024
STEP = 4
WS = 16


# ============================================================
# Helpers
# ============================================================

def make_speckle(h, w, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    f = gaussian_filter(noise, sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def apply_displacement(ref, u_field, v_field):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def gt_at_nodes(coords, u_field, v_field):
    h, w = u_field.shape
    spl_u = RectBivariateSpline(np.arange(h), np.arange(w), u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(np.arange(h), np.arange(w), v_field, kx=3, ky=3)
    u_gt = spl_u.ev(coords[:, 1], coords[:, 0])
    v_gt = spl_v.ev(coords[:, 1], coords[:, 0])
    return u_gt, v_gt


def rmse_max(u_est, v_est, u_gt, v_gt):
    eu = u_est - u_gt
    ev = v_est - v_gt
    rmse = np.sqrt(np.nanmean(eu**2 + ev**2))
    maxe = np.nanmax(np.sqrt(eu**2 + ev**2))
    return rmse, maxe


def plot_error_map(coords, u_err, v_err, title, fname):
    mag = np.sqrt(u_err**2 + v_err**2)
    vlim = np.nanpercentile(mag, 98)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=11)
    for ax, vals, lbl in zip(axes, [u_err, v_err, mag],
                              ["u error (px)", "v error (px)", "|error| (px)"]):
        vmin = np.nanpercentile(vals, 2)
        vmax = np.nanpercentile(vals, 98)
        if lbl != "|error| (px)":
            vlim2 = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-vlim2, vcenter=0, vmax=vlim2)
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=0.5,
                            cmap="RdBu_r", norm=norm, rasterized=True)
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=0.5,
                            cmap="hot_r", vmin=0, vmax=vlim, rasterized=True)
        ax.set_title(lbl, fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Deformation cases (each is a single pair: ref -> deformed)
# ============================================================

def make_cases(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    cases = {}

    # 1. Pure translation
    cases["translation"] = dict(
        u=np.full((h, w), 0.8), v=np.full((h, w), 0.3),
        desc="Translation: u=0.8px, v=0.3px",
    )
    # 2. Affine: 2% stretch + 0.5% shear
    cases["affine"] = dict(
        u=0.02 * (xx - cx) + 0.005 * (yy - cy) + 0.5,
        v=0.005 * (xx - cx) + 0.01 * (yy - cy) + 0.2,
        desc="Affine: 2% exx, 1% eyy, 0.5% shear + translation",
    )
    # 3. Large affine: 3% stretch + rotation
    cases["affine_large"] = dict(
        u=0.03 * (xx - cx) - 0.015 * (yy - cy) + 1.0,
        v=0.015 * (xx - cx) + 0.02 * (yy - cy) - 0.5,
        desc="Large affine: 3% exx + 1.5% rotation + translation",
    )
    # 4. Quadratic (parabolic bending, small magnitude)
    cases["quadratic"] = dict(
        u=0.5 + 2e-6 * (xx - cx)**2,
        v=0.2 + 3e-6 * (xx - cx) * (yy - cy),
        desc="Quadratic: u = 0.5 + 2e-6*(x-cx)^2",
    )
    # 5. Complex quadratic (barrel distortion)
    cases["quadratic_complex"] = dict(
        u=1e-6 * ((xx - cx)**2 - (yy - cy)**2) + 0.3,
        v=2e-6 * (xx - cx) * (yy - cy) + 0.1,
        desc="Quadratic complex: barrel distortion",
    )
    return cases


# ============================================================
# Per-case diagnosis
# ============================================================

def diagnose_case(name, case, ref, para, _coords_unused, _mesh_unused):
    """Run full step-by-step diagnosis for one 2-frame case.
    Builds mesh from FFT grid (same as pipeline), no pre-built mesh dependency.
    """
    print(f"\n{'='*70}")
    print(f"  {name}: {case['desc']}")
    u_span = case["u"].max() - case["u"].min()
    v_span = case["v"].max() - case["v"].min()
    print(f"  u: [{case['u'].min():.3f}, {case['u'].max():.3f}] (span {u_span:.3f}px)")
    print(f"  v: [{case['v'].min():.3f}, {case['v'].max():.3f}] (span {v_span:.3f}px)")
    print(f"{'='*70}")

    # Deformed image
    g_raw = apply_displacement(ref, case["u"], case["v"])

    roi = para.gridxy_roi_range
    imgs, clamped = normalize_images([ref, g_raw], roi)
    para_c = replace(para, gridxy_roi_range=clamped)
    f_mask = np.ones((H, W))
    Df = compute_image_gradient(imgs[0], f_mask)
    f_img = imgs[0]
    g_img = imgs[1]

    results = {}

    # --------------------------------------------------------
    # Step 1: FFT integer search — build mesh from FFT grid
    # --------------------------------------------------------
    t0 = time.perf_counter()
    x0, y0, u_grid, v_grid, fft_info = integer_search(f_img, g_img, para_c)
    t_fft = time.perf_counter() - t0

    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)

    # Build mesh from the FFT grid (exactly as pipeline does)
    dic_mesh = mesh_setup(x0, y0, para_c)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]

    # Ground truth at mesh nodes
    u_gt, v_gt = gt_at_nodes(coords, case["u"], case["v"])

    # Resize U_fft if needed (should match n_nodes)
    if len(U_fft) != 2 * n_nodes:
        # Fallback: zero-pad or trim (shouldn't happen)
        U_fft2 = np.zeros(2 * n_nodes)
        m = min(len(U_fft) // 2, n_nodes)
        U_fft2[:2*m:2] = U_fft[:2*m:2]
        U_fft2[1:2*m:2] = U_fft[1:2*m:2]
        U_fft = U_fft2

    u_fft = U_fft[0::2]
    v_fft = U_fft[1::2]

    valid = ~np.isnan(u_fft)
    n_valid = valid.sum()
    fft_rmse, fft_max = rmse_max(u_fft[valid], v_fft[valid], u_gt[valid], v_gt[valid])
    print(f"[S3] FFT integer search  ({t_fft:.2f}s, {n_valid}/{n_nodes} valid)")
    print(f"     RMSE={fft_rmse:.4f}px  max={fft_max:.4f}px")
    results["fft"] = dict(rmse=fft_rmse, maxe=fft_max, t=t_fft, u=u_fft, v=v_fft)

    plot_error_map(
        coords[valid],
        u_fft[valid] - u_gt[valid],
        v_fft[valid] - v_gt[valid],
        f"{name} — Step 1: FFT error",
        OUT_DIR / f"{name}_step1_fft.png",
    )

    # --------------------------------------------------------
    # Step 2: Local IC-GN
    # --------------------------------------------------------
    t0 = time.perf_counter()
    U_icgn, F_icgn, conv_iter, bad_count, _, _ = local_icgn(
        U_fft.copy(), coords, Df, f_img, g_img, para_c, para_c.tol,
    )
    t_icgn = time.perf_counter() - t0

    # Fill NaN
    nan_mask = np.isnan(U_icgn[0::2])
    if nan_mask.any():
        U_icgn = fill_nan_idw(U_icgn, coords, n_components=2)
        F_icgn = fill_nan_idw(F_icgn, coords, n_components=4)
    u_icgn = U_icgn[0::2]
    v_icgn = U_icgn[1::2]
    n_bad = nan_mask.sum()

    icgn_rmse, icgn_max = rmse_max(u_icgn, v_icgn, u_gt, v_gt)
    print(f"[S4] Local IC-GN         ({t_icgn:.2f}s, {n_bad} bad/NaN nodes)")
    print(f"     RMSE={icgn_rmse:.4f}px  max={icgn_max:.4f}px")
    results["icgn"] = dict(rmse=icgn_rmse, maxe=icgn_max, t=t_icgn, u=u_icgn, v=v_icgn, n_bad=n_bad)

    plot_error_map(
        coords,
        u_icgn - u_gt,
        v_icgn - v_gt,
        f"{name} — Step 2: IC-GN error",
        OUT_DIR / f"{name}_step2_icgn.png",
    )

    # --------------------------------------------------------
    # Step 3: ADMM (S5 + S6 x3)
    # --------------------------------------------------------
    mu_val = para_c.mu
    beta_val = 1e-3 * STEP**2 * mu_val
    alpha = 0.0

    # Precompute
    subpb1_pre = precompute_subpb1(coords, Df, f_img, para_c)
    subpb2_cache = precompute_subpb2(dic_mesh, para_c.gauss_pt_order, beta_val, mu_val, alpha)

    U_s1 = U_icgn.copy()
    F_s1 = F_icgn.copy()

    # S5: first subpb2
    U_s2 = subpb2_solver(
        dic_mesh, para_c.gauss_pt_order, beta_val, mu_val,
        U_s1, F_s1, np.zeros(4*n_nodes), np.zeros(2*n_nodes),
        alpha, STEP, precomputed=subpb2_cache,
    )
    F_s2 = global_nodal_strain_fem(dic_mesh, para_c, U_s2)

    grad_dual = F_s2 - F_s1
    disp_dual = U_s2 - U_s1

    admm_norms = []
    winsize_list = np.full((n_nodes, 2), para_c.winsize, dtype=np.float64)
    para_c2 = replace(para_c, winsize_list=winsize_list)

    t0 = time.perf_counter()
    for it in range(1, 4):
        U_s1, _, _, _ = subpb1_solver(
            U_s2, F_s2, disp_dual, grad_dual,
            coords, Df, f_img, g_img,
            mu_val, beta_val, para_c2, para_c.tol,
            precomputed=subpb1_pre,
        )
        F_s1 = F_s2.copy()
        U_s2 = subpb2_solver(
            dic_mesh, para_c.gauss_pt_order, beta_val, mu_val,
            U_s1, F_s1, grad_dual, disp_dual,
            alpha, STEP, precomputed=subpb2_cache,
        )
        F_s2 = global_nodal_strain_fem(dic_mesh, para_c2, U_s2)
        g_norm = np.sqrt(np.mean((F_s2 - F_s1)**2))
        d_norm = np.sqrt(np.mean((U_s2 - U_s1)**2))
        conv_norm = max(g_norm, d_norm)
        admm_norms.append(conv_norm)
        grad_dual = F_s2 - F_s1
        disp_dual = U_s2 - U_s1
        print(f"     ADMM iter {it}: conv_norm={conv_norm:.4e}")

    t_admm = time.perf_counter() - t0

    u_admm = U_s2[0::2]
    v_admm = U_s2[1::2]
    admm_rmse, admm_max = rmse_max(u_admm, v_admm, u_gt, v_gt)
    print(f"[S6] After ADMM x3       ({t_admm:.2f}s)")
    print(f"     RMSE={admm_rmse:.4f}px  max={admm_max:.4f}px")
    results["admm"] = dict(rmse=admm_rmse, maxe=admm_max, t=t_admm, u=u_admm, v=v_admm, norms=admm_norms)

    plot_error_map(
        coords,
        u_admm - u_gt,
        v_admm - v_gt,
        f"{name} — Step 3: After ADMM error",
        OUT_DIR / f"{name}_step3_admm.png",
    )

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"STAQ-DIC Step-by-Step Diagnosis: {H}x{W}, step={STEP}, ws={WS}")
    print(f"Each case is an independent 2-frame test (ref + deformed)")
    print(f"{'='*70}")

    ref = make_speckle(H, W, sigma=3.0, seed=42)
    cases = make_cases(H, W)

    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
        reference_mode="accumulative", show_plots=False,
    )

    # Build mesh once (same for all cases)
    imgs0, clamped0 = normalize_images([ref, ref], roi)
    para0 = replace(para, gridxy_roi_range=clamped0)
    x0 = np.arange(roi.gridx[0], roi.gridx[1] + 1, STEP, dtype=np.float64)
    y0 = np.arange(roi.gridy[0], roi.gridy[1] + 1, STEP, dtype=np.float64)
    dic_mesh = mesh_setup(x0, y0, para0)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]
    print(f"\nMesh: {n_nodes} nodes, {dic_mesh.elements_fem.shape[0]} elements\n")

    # Summary table
    all_results = {}
    for name, case in cases.items():
        all_results[name] = diagnose_case(name, case, ref, para, coords, dic_mesh)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY: RMSE (px) at each step")
    print(f"{'='*70}")
    print(f"{'Case':<22} {'FFT':>8} {'IC-GN':>8} {'ADMM':>8}")
    print("-" * 55)
    for name, r in all_results.items():
        fft_r = r['fft']['rmse']
        icgn_r = r['icgn']['rmse']
        admm_r = r['admm']['rmse']
        print(f"  {name:<20} {fft_r:8.4f} {icgn_r:8.4f} {admm_r:8.4f}")
    print("-" * 55)

    print(f"\nPlots saved to: {OUT_DIR.resolve()}")
