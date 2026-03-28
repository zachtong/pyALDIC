"""Detailed IC-GN diagnosis: bypass detect_bad_points to see raw IC-GN accuracy.

Tests on 20px quadratic field:
1. FFT accuracy
2. Raw Numba IC-GN results (before detect_bad_points/fill_nan_rbf)
3. After detect_bad_points/fill_nan_rbf
"""

import sys
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RectBivariateSpline
from dataclasses import replace

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.io.image_ops import compute_image_gradient, normalize_images
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.solver.integer_search import integer_search
from staq_dic.solver.init_disp import init_disp
from staq_dic.solver.outlier_detection import detect_bad_points, fill_nan_rbf

H, W = 1024, 1024
STEP = 4
WS = 16


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
    return spl_u.ev(coords[:, 1], coords[:, 0]), spl_v.ev(coords[:, 1], coords[:, 0])


if __name__ == "__main__":
    ref = make_speckle(H, W)

    # Quadratic field: u = 10*(xn^2 + yn^2), v = 8*xn*yn
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    xn = (xx - cx) / cx
    yn = (yy - cy) / cy
    u_field = 10.0 * (xn**2 + yn**2)
    v_field = 8.0 * xn * yn

    g_raw = apply_displacement(ref, u_field, v_field)

    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
        size_of_fft_search_region=30, show_plots=False,
    )

    imgs, clamped = normalize_images([ref, g_raw], roi)
    para = replace(para, gridxy_roi_range=clamped)
    f_mask = np.ones((H, W))
    Df = compute_image_gradient(imgs[0], f_mask)

    # FFT
    x0, y0, u_grid, v_grid, fft_info = integer_search(imgs[0], imgs[1], para)
    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]

    u_gt, v_gt = gt_at_nodes(coords, u_field, v_field)
    u_fft, v_fft = U_fft[0::2], U_fft[1::2]
    fft_err = np.sqrt((u_fft - u_gt)**2 + (v_fft - v_gt)**2)
    print(f"FFT: RMSE={np.sqrt(np.mean(fft_err**2)):.4f}, max={fft_err.max():.4f}")

    # ============================================================
    # Raw Numba IC-GN (bypass local_icgn wrapper)
    # ============================================================
    from staq_dic.solver.icgn_batch import precompute_subsets_6dof
    from staq_dic.solver.numba_kernels import icgn_6dof_parallel, HAS_NUMBA

    print(f"\nHAS_NUMBA = {HAS_NUMBA}")

    pre = precompute_subsets_6dof(
        coords, imgs[0], Df.df_dx, Df.df_dy, Df.img_ref_mask, WS,
    )
    print(f"Valid subsets: {pre['valid'].sum()}/{n_nodes}")

    U0_2d = U_fft.reshape(-1, 2)
    rounded_coords = np.round(coords).astype(np.float64)

    t0 = time.perf_counter()
    P_out, conv_iter = icgn_6dof_parallel(
        rounded_coords,
        U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
        pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
        pre["XX_all"], pre["YY_all"], pre["H_all"],
        pre["meanf_all"], pre["bottomf_all"],
        pre["valid"], imgs[1], para.tol, para.icgn_max_iter,
    )
    t_icgn = time.perf_counter() - t0

    u_raw = P_out[:, 4]
    v_raw = P_out[:, 5]
    raw_err = np.sqrt((u_raw - u_gt)**2 + (v_raw - v_gt)**2)

    print(f"\n{'='*60}")
    print(f"Raw Numba IC-GN results ({t_icgn:.3f}s)")
    print(f"{'='*60}")

    # Convergence stats
    valid = pre["valid"]
    ci = conv_iter.ravel()
    print(f"\nConvergence iteration distribution (valid nodes only):")
    for maxv in [1, 2, 3, 5, 10, 20, 50]:
        count = np.sum((ci[valid] >= 1) & (ci[valid] <= maxv))
        print(f"  <= {maxv:3d} iters: {count:6d} / {valid.sum()} ({count/valid.sum()*100:.1f}%)")

    failed = ci[valid] > para.icgn_max_iter
    failed_neg = ci[valid] < 0
    maxiter_exact = ci[valid] == para.icgn_max_iter + 1
    maskfail = ci[valid] == para.icgn_max_iter + 2
    print(f"  Failed (>max_iter):  {maxiter_exact.sum()}")
    print(f"  Mask fail:           {maskfail.sum()}")
    print(f"  Negative (error):    {failed_neg.sum()}")

    # Error on CONVERGED nodes only
    converged = (ci >= 1) & (ci <= para.icgn_max_iter) & valid
    n_conv = converged.sum()
    if n_conv > 0:
        conv_err = raw_err[converged]
        print(f"\nConverged nodes ({n_conv}):")
        print(f"  RMSE = {np.sqrt(np.mean(conv_err**2)):.6f}px")
        print(f"  Max  = {conv_err.max():.6f}px")
        print(f"  P50  = {np.percentile(conv_err, 50):.6f}px")
        print(f"  P95  = {np.percentile(conv_err, 95):.6f}px")
        print(f"  P99  = {np.percentile(conv_err, 99):.6f}px")

    # Error on ALL valid nodes (including failed)
    all_valid_err = raw_err[valid]
    print(f"\nAll valid nodes ({valid.sum()}):")
    print(f"  RMSE = {np.sqrt(np.mean(all_valid_err**2)):.6f}px")
    print(f"  Max  = {all_valid_err.max():.6f}px")

    # After detect_bad_points + fill_nan_rbf (same as local_icgn)
    U_raw = np.empty(2 * n_nodes)
    U_raw[0::2] = u_raw
    U_raw[1::2] = v_raw
    F_raw = np.zeros(4 * n_nodes)
    F_raw[0::4] = P_out[:, 0]
    F_raw[1::4] = P_out[:, 1]
    F_raw[2::4] = P_out[:, 2]
    F_raw[3::4] = P_out[:, 3]

    bad_pts, bad_pt_num = detect_bad_points(
        conv_iter, para.icgn_max_iter, coords,
        sigma_factor=1.0, min_threshold=6,
    )
    print(f"\ndetect_bad_points: {len(bad_pts)} bad nodes (bad_pt_num={bad_pt_num})")
    print(f"  That's {len(bad_pts)/n_nodes*100:.1f}% of all nodes")

    if len(bad_pts) > 0:
        # Show convergence stats for bad points
        bad_ci = ci[bad_pts]
        print(f"  Bad points iter range: [{bad_ci.min()}, {bad_ci.max()}]")
        print(f"  Bad points that converged (iter in [1, max_iter]): "
              f"{((bad_ci >= 1) & (bad_ci <= para.icgn_max_iter)).sum()}")

    U_fill = U_raw.copy()
    U_fill[2 * bad_pts] = np.nan
    U_fill[2 * bad_pts + 1] = np.nan
    U_fill = fill_nan_rbf(U_fill, coords, n_components=2)

    u_fill = U_fill[0::2]
    v_fill = U_fill[1::2]
    fill_err = np.sqrt((u_fill - u_gt)**2 + (v_fill - v_gt)**2)

    print(f"\nAfter detect_bad_points + fill_nan_rbf:")
    print(f"  RMSE = {np.sqrt(np.mean(fill_err**2)):.6f}px")
    print(f"  Max  = {fill_err.max():.6f}px")

    # Show what the detect_bad_points threshold is
    good_pts = np.setdiff1d(np.arange(n_nodes), bad_pts)
    if len(good_pts) > 0:
        good_ci = ci[good_pts]
        good_ci_pos = good_ci[good_ci > 0]
        if len(good_ci_pos) > 0:
            mu_ci = np.mean(good_ci_pos)
            std_ci = np.std(good_ci_pos, ddof=1) if len(good_ci_pos) > 1 else 0
            threshold = max(mu_ci + 1.0 * std_ci, 6)
            print(f"\n  detect_bad_points threshold: max(mean+1*std, 6)")
            print(f"    mean_iter = {mu_ci:.2f}, std = {std_ci:.2f}")
            print(f"    threshold = {threshold:.2f}")
