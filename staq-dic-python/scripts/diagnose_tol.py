"""Test IC-GN accuracy vs convergence tolerance.

Run the same 20px quadratic field with different tol values
to see if IC-GN accuracy improves with tighter tolerance.
Also test single-node behavior to understand convergence.
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


if __name__ == "__main__":
    ref = make_speckle(H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    xn, yn = (xx - cx) / cx, (yy - cy) / cy
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

    x0, y0, u_grid, v_grid, fft_info = integer_search(imgs[0], imgs[1], para)
    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]

    # Ground truth
    spl_u = RectBivariateSpline(np.arange(H), np.arange(W), u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(np.arange(H), np.arange(W), v_field, kx=3, ky=3)
    u_gt = spl_u.ev(coords[:, 1], coords[:, 0])
    v_gt = spl_v.ev(coords[:, 1], coords[:, 0])

    # Precompute subsets
    pre = precompute_subsets_6dof(coords, imgs[0], Df.df_dx, Df.df_dy, Df.img_ref_mask, WS)
    U0_2d = U_fft.reshape(-1, 2)
    rounded_coords = np.round(coords).astype(np.float64)

    print(f"FFT RMSE: {np.sqrt(np.mean((U_fft[0::2]-u_gt)**2 + (U_fft[1::2]-v_gt)**2)):.6f}px")
    print(f"\n{'='*70}")
    print("Testing different tol values")
    print(f"{'='*70}")

    for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        t0 = time.perf_counter()
        P_out, conv_iter = icgn_6dof_parallel(
            rounded_coords,
            U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
            pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
            pre["XX_all"], pre["YY_all"], pre["H_all"],
            pre["meanf_all"], pre["bottomf_all"],
            pre["valid"], imgs[1], tol, 200,
        )
        dt = time.perf_counter() - t0

        u_out, v_out = P_out[:, 4], P_out[:, 5]
        err = np.sqrt((u_out - u_gt)**2 + (v_out - v_gt)**2)
        valid = pre["valid"]
        ci = conv_iter[valid]
        mean_iter = np.mean(ci[ci > 0])
        max_iter_used = np.max(ci[ci > 0]) if (ci > 0).any() else 0

        print(f"  tol={tol:.0e}: RMSE={np.sqrt(np.mean(err**2)):.6f}px, "
              f"max={err.max():.6f}px, "
              f"mean_iter={mean_iter:.1f}, max_iter={max_iter_used}, "
              f"time={dt:.3f}s")

    # ============================================================
    # Single-node deep debug at center and edge
    # ============================================================
    print(f"\n{'='*70}")
    print("Single-node debug (center node vs edge node)")
    print(f"{'='*70}")

    center_idx = np.argmin(np.sum((coords - np.array([cx, cy]))**2, axis=1))
    edge_idx = np.argmin(np.sum((coords - np.array([60.0, 60.0]))**2, axis=1))

    for label, idx in [("center", center_idx), ("edge", edge_idx)]:
        x0n = rounded_coords[idx, 0]
        y0n = rounded_coords[idx, 1]
        u0n = U0_2d[idx, 0]
        v0n = U0_2d[idx, 1]

        Sy = pre["ref_all"].shape[1]
        Sx = pre["ref_all"].shape[2]

        # Run single node with tight tolerance
        P, step = _icgn_6dof_single(
            x0n, y0n, u0n, v0n,
            pre["ref_all"][idx], pre["gx_all"][idx], pre["gy_all"][idx],
            pre["mask_all"][idx],
            pre["XX_all"][idx], pre["YY_all"][idx], pre["H_all"][idx],
            pre["meanf_all"][idx], pre["bottomf_all"][idx],
            imgs[1], 1e-10, 200, H, W, Sy, Sx,
        )

        u_icgn, v_icgn = P[4], P[5]
        u_gt_n, v_gt_n = u_gt[idx], v_gt[idx]
        err_u = u_icgn - u_gt_n
        err_v = v_icgn - v_gt_n
        err_mag = np.sqrt(err_u**2 + err_v**2)

        print(f"\n  {label} node [{idx}] at ({x0n:.0f}, {y0n:.0f}):")
        print(f"    GT:  u={u_gt_n:.6f}, v={v_gt_n:.6f}")
        print(f"    FFT: u={u0n:.6f}, v={v0n:.6f} (err={np.sqrt((u0n-u_gt_n)**2+(v0n-v_gt_n)**2):.6f})")
        print(f"    ICG: u={u_icgn:.6f}, v={v_icgn:.6f} (err={err_mag:.6f})")
        print(f"    F = [{P[0]:.6f}, {P[1]:.6f}, {P[2]:.6f}, {P[3]:.6f}]")
        print(f"    Converged in {step} iterations")

        # Also check: what do GT F values look like?
        # du/dx at this node
        dudx_gt = 20.0 * (x0n - cx) / cx**2
        dudy_gt = 20.0 * (y0n - cy) / cy**2
        dvdx_gt = 8.0 * (y0n - cy) / cx / cy
        dvdy_gt = 8.0 * (x0n - cx) / cx / cy
        print(f"    GT F: [{dudx_gt:.6f}, {dvdx_gt:.6f}, {dudy_gt:.6f}, {dvdy_gt:.6f}]")
        print(f"    F err: [{P[0]-dudx_gt:.6f}, {P[1]-dvdx_gt:.6f}, "
              f"{P[2]-dudy_gt:.6f}, {P[3]-dvdy_gt:.6f}]")
