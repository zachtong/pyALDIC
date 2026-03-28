"""Quick test: does IC-GN work for large UNIFORM translations?

Isolates whether the issue is large displacements or varying fields.
"""

import sys, time, warnings
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
from staq_dic.solver.icgn_batch import precompute_subsets_6dof
from staq_dic.solver.numba_kernels import icgn_6dof_parallel, _icgn_6dof_single

H, W = 1024, 1024
STEP = 4
WS = 16


def make_speckle(h, w, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((h, w)), sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def apply_displacement(ref, u_field, v_field):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def run_test(label, u_field, v_field, ref):
    print(f"\n{'='*65}")
    print(f"  {label}")
    u_span = u_field.max() - u_field.min()
    v_span = v_field.max() - v_field.min()
    print(f"  u: [{u_field.min():.3f}, {u_field.max():.3f}] (span {u_span:.3f})")
    print(f"  v: [{v_field.min():.3f}, {v_field.max():.3f}] (span {v_span:.3f})")
    print(f"{'='*65}")

    g_raw = apply_displacement(ref, u_field, v_field)

    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-6, icgn_max_iter=100, mu=1e-3, alpha=0.0,
        size_of_fft_search_region=30, show_plots=False,
    )
    imgs, clamped = normalize_images([ref, g_raw], roi)
    para = replace(para, gridxy_roi_range=clamped)
    Df = compute_image_gradient(imgs[0], np.ones((H, W)))

    x0, y0, u_grid, v_grid, fft_info = integer_search(imgs[0], imgs[1], para)
    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]

    spl_u = RectBivariateSpline(np.arange(H), np.arange(W), u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(np.arange(H), np.arange(W), v_field, kx=3, ky=3)
    u_gt = spl_u.ev(coords[:, 1], coords[:, 0])
    v_gt = spl_v.ev(coords[:, 1], coords[:, 0])

    fft_err = np.sqrt((U_fft[0::2] - u_gt)**2 + (U_fft[1::2] - v_gt)**2)
    print(f"  FFT: RMSE={np.sqrt(np.mean(fft_err**2)):.6f}  max={fft_err.max():.6f}")

    # Raw Numba IC-GN
    pre = precompute_subsets_6dof(coords, imgs[0], Df.df_dx, Df.df_dy, Df.img_ref_mask, WS)
    rounded = np.round(coords).astype(np.float64)
    U0_2d = U_fft.reshape(-1, 2)

    P_out, conv_iter = icgn_6dof_parallel(
        rounded, U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
        pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
        pre["XX_all"], pre["YY_all"], pre["H_all"],
        pre["meanf_all"], pre["bottomf_all"],
        pre["valid"], imgs[1], 1e-6, 100,
    )

    u_icgn, v_icgn = P_out[:, 4], P_out[:, 5]
    icgn_err = np.sqrt((u_icgn - u_gt)**2 + (v_icgn - v_gt)**2)
    ci = conv_iter[pre["valid"]]
    print(f"  ICG: RMSE={np.sqrt(np.mean(icgn_err**2)):.6f}  max={icgn_err.max():.6f}")
    print(f"       mean_iter={np.mean(ci):.1f}  max_iter={ci.max()}")

    # Single-node detail for center and edge
    center_idx = np.argmin(np.sum((coords - [512, 512])**2, axis=1))
    edge_idx = np.argmin(np.sum((coords - [58, 58])**2, axis=1))
    corner_idx = np.argmin(np.sum((coords - [100, 100])**2, axis=1))

    for tag, idx in [("center", center_idx), ("edge(58,58)", edge_idx), ("(100,100)", corner_idx)]:
        eu = u_icgn[idx] - u_gt[idx]
        ev = v_icgn[idx] - v_gt[idx]
        ef = np.sqrt(eu**2 + ev**2)
        fft_e = fft_err[idx]
        print(f"    {tag:>12}: gt_u={u_gt[idx]:8.3f} icgn_u={u_icgn[idx]:8.3f} "
              f"fft_err={fft_e:.4f} icgn_err={ef:.4f} iter={conv_iter[idx]}")


if __name__ == "__main__":
    ref = make_speckle(H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    xn, yn = (xx - cx) / cx, (yy - cy) / cy

    # Test 1: Large uniform translation (15.7px)
    run_test("15.7px uniform translation",
             np.full((H, W), 15.7), np.full((H, W), 6.3), ref)

    # Test 2: Large uniform translation (sub-pixel: 15.3px)
    run_test("15.3px uniform translation",
             np.full((H, W), 15.3), np.full((H, W), 6.7), ref)

    # Test 3: 20px quadratic (same as before)
    run_test("20px quadratic (original)",
             10.0 * (xn**2 + yn**2), 8.0 * xn * yn, ref)

    # Test 4: 20px quadratic but skip init_disp smoothing — use raw FFT integers
    print(f"\n{'='*65}")
    print("  Test 4: 20px quadratic with PERFECT initial guess")
    print(f"{'='*65}")
    u_q = 10.0 * (xn**2 + yn**2)
    v_q = 8.0 * xn * yn
    g_raw = apply_displacement(ref, u_q, v_q)
    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-6, icgn_max_iter=100, mu=1e-3, alpha=0.0,
        size_of_fft_search_region=30, show_plots=False,
    )
    imgs, clamped = normalize_images([ref, g_raw], roi)
    para = replace(para, gridxy_roi_range=clamped)
    Df = compute_image_gradient(imgs[0], np.ones((H, W)))

    # Build mesh manually
    x0 = np.arange(clamped.gridx[0], clamped.gridx[1] + 1, STEP, dtype=np.float64)
    y0 = np.arange(clamped.gridy[0], clamped.gridy[1] + 1, STEP, dtype=np.float64)
    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]

    spl_u = RectBivariateSpline(np.arange(H), np.arange(W), u_q, kx=3, ky=3)
    spl_v = RectBivariateSpline(np.arange(H), np.arange(W), v_q, kx=3, ky=3)
    u_gt = spl_u.ev(coords[:, 1], coords[:, 0])
    v_gt = spl_v.ev(coords[:, 1], coords[:, 0])

    # Use PERFECT initial guess: round(gt) as integers
    U_perfect = np.zeros(2 * n_nodes)
    U_perfect[0::2] = np.round(u_gt)
    U_perfect[1::2] = np.round(v_gt)

    perfect_err = np.sqrt((U_perfect[0::2] - u_gt)**2 + (U_perfect[1::2] - v_gt)**2)
    print(f"  Perfect init: RMSE={np.sqrt(np.mean(perfect_err**2)):.6f}  max={perfect_err.max():.6f}")

    pre = precompute_subsets_6dof(coords, imgs[0], Df.df_dx, Df.df_dy, Df.img_ref_mask, WS)
    rounded = np.round(coords).astype(np.float64)
    U0_2d = U_perfect.reshape(-1, 2)

    P_out, conv_iter = icgn_6dof_parallel(
        rounded, U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
        pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
        pre["XX_all"], pre["YY_all"], pre["H_all"],
        pre["meanf_all"], pre["bottomf_all"],
        pre["valid"], imgs[1], 1e-6, 100,
    )

    u_icgn, v_icgn = P_out[:, 4], P_out[:, 5]
    icgn_err = np.sqrt((u_icgn - u_gt)**2 + (v_icgn - v_gt)**2)
    ci = conv_iter[pre["valid"]]
    print(f"  ICG: RMSE={np.sqrt(np.mean(icgn_err**2)):.6f}  max={icgn_err.max():.6f}")
    print(f"       mean_iter={np.mean(ci):.1f}  max_iter={ci.max()}")

    center_idx = np.argmin(np.sum((coords - [512, 512])**2, axis=1))
    edge_idx = np.argmin(np.sum((coords - [58, 58])**2, axis=1))
    for tag, idx in [("center", center_idx), ("edge(58,58)", edge_idx)]:
        eu = u_icgn[idx] - u_gt[idx]
        ev = v_icgn[idx] - v_gt[idx]
        ef = np.sqrt(eu**2 + ev**2)
        print(f"    {tag:>12}: gt_u={u_gt[idx]:8.3f} icgn_u={u_icgn[idx]:8.3f} "
              f"icgn_err={ef:.6f} iter={conv_iter[idx]}")
