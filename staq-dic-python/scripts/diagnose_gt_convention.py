"""Diagnose the ground truth convention mismatch in synthetic DIC tests.

The issue: apply_displacement uses Eulerian mapping g(x,y) = f(x-u(x,y), y-v(x,y))
where u is defined at DEFORMED coordinates. But DIC solves for Lagrangian displacement
at REFERENCE coordinates: u_true(X) = u_field(X + u_true).

This script:
1. Shows the naive GT vs corrected GT vs IC-GN results
2. Generates images using corrected Lagrangian inverse mapping
3. Proves IC-GN is accurate when test methodology is correct
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
from staq_dic.solver.numba_kernels import icgn_6dof_parallel

H, W = 1024, 1024
STEP = 4
WS = 16


def make_speckle(h, w, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((h, w)), sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def u_func(x, y, cx, cy):
    """Quadratic u-displacement: u = 10*((x-cx)^2/cx^2 + (y-cy)^2/cy^2)"""
    return 10.0 * ((x - cx) ** 2 / cx ** 2 + (y - cy) ** 2 / cy ** 2)


def v_func(x, y, cx, cy):
    """Quadratic v-displacement: v = 8*(x-cx)*(y-cy)/(cx*cy)"""
    return 8.0 * (x - cx) * (y - cy) / (cx * cy)


def apply_displacement_eulerian(ref, cx, cy):
    """CURRENT (WRONG for large fields): g(x,y) = f(x-u(x,y), y-v(x,y))

    u is evaluated at deformed coordinates (x,y), not reference.
    """
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u_field = u_func(xx, yy, cx, cy)
    v_field = v_func(xx, yy, cx, cy)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def apply_displacement_lagrangian(ref, cx, cy, n_iter=20):
    """CORRECT: For each deformed pixel (x,y), find reference pixel (X,Y)
    such that X + u(X,Y) = x, Y + v(X,Y) = y, then g(x,y) = f(X,Y).

    Uses fixed-point iteration:
      X_{n+1} = x - u(X_n, Y_n)
      Y_{n+1} = y - v(X_n, Y_n)
    """
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Fixed-point iteration to invert the mapping
    X = xx.copy()
    Y = yy.copy()
    for _ in range(n_iter):
        u_at_XY = u_func(X, Y, cx, cy)
        v_at_XY = v_func(X, Y, cx, cy)
        X_new = xx - u_at_XY
        Y_new = yy - v_at_XY
        # Check convergence
        max_change = max(np.max(np.abs(X_new - X)), np.max(np.abs(Y_new - Y)))
        X = X_new
        Y = Y_new
        if max_change < 1e-10:
            break

    coords = np.array([Y.ravel(), X.ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def correct_gt_at_nodes(coords, cx, cy, n_iter=20):
    """Compute corrected GT by solving u_true = u_field(X + u_true, Y + v_true).

    For Eulerian-generated images, the IC-GN finds THIS displacement, not u_field(X,Y).
    """
    n = coords.shape[0]
    x0 = coords[:, 0]
    y0 = coords[:, 1]

    u = u_func(x0, y0, cx, cy)
    v = v_func(x0, y0, cx, cy)
    for _ in range(n_iter):
        u_new = u_func(x0 + u, y0 + v, cx, cy)
        v_new = v_func(x0 + u, y0 + v, cx, cy)
        max_change = max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v)))
        u = u_new
        v = v_new
        if max_change < 1e-10:
            break

    return u, v


def run_icgn(ref_img, def_img, cx, cy, label=""):
    """Run IC-GN pipeline and return results."""
    roi = GridxyROIRange(gridx=(WS, W - WS - 1), gridy=(WS, H - WS - 1))
    para = dicpara_default(
        winsize=WS, winstepsize=STEP, winsize_min=STEP,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-6, icgn_max_iter=100, mu=1e-3, alpha=0.0,
        size_of_fft_search_region=30, show_plots=False,
    )
    imgs, clamped = normalize_images([ref_img, def_img], roi)
    para = replace(para, gridxy_roi_range=clamped)
    Df = compute_image_gradient(imgs[0], np.ones((H, W)))

    x0, y0, u_grid, v_grid, fft_info = integer_search(imgs[0], imgs[1], para)
    U_fft = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    dic_mesh = mesh_setup(x0, y0, para)
    coords = dic_mesh.coordinates_fem

    pre = precompute_subsets_6dof(coords, imgs[0], Df.df_dx, Df.df_dy, Df.img_ref_mask, WS)
    rounded = np.round(coords).astype(np.float64)
    U0_2d = U_fft.reshape(-1, 2)

    t0 = time.perf_counter()
    P_out, conv_iter = icgn_6dof_parallel(
        rounded, U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
        pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
        pre["XX_all"], pre["YY_all"], pre["H_all"],
        pre["meanf_all"], pre["bottomf_all"],
        pre["valid"], imgs[1], 1e-6, 100,
    )
    dt = time.perf_counter() - t0

    return coords, U_fft, P_out, conv_iter, pre["valid"], dt


if __name__ == "__main__":
    ref = make_speckle(H, W)
    cx, cy = W / 2.0, H / 2.0

    # ================================================================
    # Part 1: Show the GT mismatch for Eulerian-generated images
    # ================================================================
    print("=" * 70)
    print("Part 1: Eulerian image + naive GT vs corrected GT vs IC-GN")
    print("=" * 70)

    g_euler = apply_displacement_eulerian(ref, cx, cy)
    coords, U_fft, P_out, conv_iter, valid, dt = run_icgn(ref, g_euler, cx, cy)

    # Naive GT: u_field(X, Y)
    u_naive = u_func(coords[:, 0], coords[:, 1], cx, cy)
    v_naive = v_func(coords[:, 0], coords[:, 1], cx, cy)

    # Corrected GT: solve u_true = u_field(X + u_true, Y + v_true)
    u_correct, v_correct = correct_gt_at_nodes(coords, cx, cy)

    # IC-GN results
    u_icgn, v_icgn = P_out[:, 4], P_out[:, 5]

    err_vs_naive = np.sqrt((u_icgn - u_naive) ** 2 + (v_icgn - v_naive) ** 2)
    err_vs_correct = np.sqrt((u_icgn - u_correct) ** 2 + (v_icgn - v_correct) ** 2)
    gt_diff = np.sqrt((u_naive - u_correct) ** 2 + (v_naive - v_correct) ** 2)

    print(f"\n  Naive GT vs Corrected GT:")
    print(f"    RMSE = {np.sqrt(np.mean(gt_diff ** 2)):.6f}px")
    print(f"    Max  = {gt_diff.max():.6f}px")

    print(f"\n  IC-GN vs Naive GT (what we saw as 'error'):")
    print(f"    RMSE = {np.sqrt(np.mean(err_vs_naive ** 2)):.6f}px")
    print(f"    Max  = {err_vs_naive.max():.6f}px")

    print(f"\n  IC-GN vs Corrected GT (true IC-GN accuracy):")
    print(f"    RMSE = {np.sqrt(np.mean(err_vs_correct ** 2)):.6f}px")
    print(f"    Max  = {err_vs_correct.max():.6f}px")

    # Show specific nodes
    center_idx = np.argmin(np.sum((coords - [512, 512]) ** 2, axis=1))
    edge_idx = np.argmin(np.sum((coords - [58, 58]) ** 2, axis=1))

    print(f"\n  Per-node details:")
    for tag, idx in [("center(512,512)", center_idx), ("edge(58,58)", edge_idx)]:
        print(f"\n    {tag}:")
        print(f"      Naive GT:     u={u_naive[idx]:.6f}  v={v_naive[idx]:.6f}")
        print(f"      Corrected GT: u={u_correct[idx]:.6f}  v={v_correct[idx]:.6f}")
        print(f"      IC-GN:        u={u_icgn[idx]:.6f}  v={v_icgn[idx]:.6f}")
        print(f"      Error vs naive:     {err_vs_naive[idx]:.6f}px")
        print(f"      Error vs corrected: {err_vs_correct[idx]:.6f}px")
        print(f"      GT difference:      {gt_diff[idx]:.6f}px")

    # ================================================================
    # Part 2: Lagrangian image generation + IC-GN
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Part 2: Lagrangian (correct) image generation + IC-GN")
    print("=" * 70)

    t0 = time.perf_counter()
    g_lagr = apply_displacement_lagrangian(ref, cx, cy)
    t_gen = time.perf_counter() - t0
    print(f"  Image generation time: {t_gen:.3f}s")

    coords2, U_fft2, P_out2, conv_iter2, valid2, dt2 = run_icgn(ref, g_lagr, cx, cy)

    u_naive2 = u_func(coords2[:, 0], coords2[:, 1], cx, cy)
    v_naive2 = v_func(coords2[:, 0], coords2[:, 1], cx, cy)
    u_icgn2, v_icgn2 = P_out2[:, 4], P_out2[:, 5]

    # For Lagrangian images, the naive GT IS the correct GT
    err_lagr = np.sqrt((u_icgn2 - u_naive2) ** 2 + (v_icgn2 - v_naive2) ** 2)

    print(f"\n  IC-GN vs u_field(X,Y) (now correct for Lagrangian images):")
    print(f"    RMSE = {np.sqrt(np.mean(err_lagr ** 2)):.6f}px")
    print(f"    Max  = {err_lagr.max():.6f}px")

    for tag, idx in [("center(512,512)", center_idx), ("edge(58,58)", edge_idx)]:
        print(f"\n    {tag}:")
        print(f"      GT:   u={u_naive2[idx]:.6f}  v={v_naive2[idx]:.6f}")
        print(f"      ICGN: u={u_icgn2[idx]:.6f}  v={v_icgn2[idx]:.6f}")
        print(f"      Error: {err_lagr[idx]:.6f}px")

    # ================================================================
    # Part 3: Compare Eulerian vs Lagrangian image differences
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Part 3: Image difference statistics")
    print("=" * 70)
    diff = np.abs(g_euler - g_lagr)
    print(f"  |g_Euler - g_Lagrangian|:")
    print(f"    Mean = {diff.mean():.6f}")
    print(f"    Max  = {diff.max():.6f}")
    print(f"    Std  = {diff.std():.6f}")

    # At what locations is the difference largest?
    max_loc = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"    Max diff at pixel (row={max_loc[0]}, col={max_loc[1]})")
    u_at_max = u_func(max_loc[1], max_loc[0], cx, cy)
    v_at_max = v_func(max_loc[1], max_loc[0], cx, cy)
    print(f"    Displacement there: u={u_at_max:.2f}, v={v_at_max:.2f}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
  The IC-GN solver is CORRECT. The apparent 0.26px RMSE was caused by a
  ground-truth convention mismatch in the test methodology:

  - apply_displacement() uses Eulerian mapping: g(x,y) = f(x-u(x,y), y-v(x,y))
  - The test computed GT as u_field(X,Y) at reference coordinates
  - But the TRUE displacement for this mapping is u_field(X+u_true, Y+v_true)
  - Difference is O(du/dx * u) ~ 0.034 * 15.86 = 0.54px at edges

  FIX: Either:
  A) Use Lagrangian image generation (fixed-point iteration to invert mapping)
     -> Then u_field(X,Y) IS the correct ground truth
     -> IC-GN RMSE drops to ~{np.sqrt(np.mean(err_lagr**2)):.4f}px

  B) Keep Eulerian images but compute corrected GT:
     -> Solve u_true = u_field(X+u_true, Y+v_true) by fixed-point iteration
     -> IC-GN RMSE drops to ~{np.sqrt(np.mean(err_vs_correct**2)):.4f}px
""")
