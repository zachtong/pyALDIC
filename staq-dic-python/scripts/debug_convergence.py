"""Focused convergence diagnostic for translation and affine cases.

Tests IC-GN convergence with perturbed ground-truth initial guesses
to match realistic conditions (MATLAB FFT gives ~0.1-1.0 px error).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from staq_dic.solver.icgn_solver import icgn_solver
from staq_dic.io.image_ops import compute_image_gradient

# ---------------------------------------------------------------------------
# Image generation (same as conftest.py)
# ---------------------------------------------------------------------------

def generate_speckle(h=256, w=256, sigma=3.0, seed=42):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement(ref, u_field, v_field, order=5):
    from scipy.ndimage import map_coordinates
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src_y = yy - v_field
    src_x = xx - u_field
    coords = np.array([src_y.ravel(), src_x.ravel()])
    warped = map_coordinates(ref, coords, order=order, mode="constant", cval=0.0)
    return warped.reshape(h, w)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_single_node_test(
    name: str,
    img_ref: np.ndarray,
    img_def: np.ndarray,
    x0: float,
    y0: float,
    gt_u: float,
    gt_v: float,
    perturbation: float,
    winsize: int = 32,
    tol: float = 1e-2,
    max_iter: int = 100,
    seed: int = 0,
):
    """Run IC-GN on a single node with perturbed initial guess."""
    h, w = img_ref.shape
    mask = np.ones((h, w), dtype=np.float64)
    grads = compute_image_gradient(img_ref)
    df_dx, df_dy = grads.df_dx, grads.df_dy

    # Perturbed initial guess
    rng = np.random.default_rng(seed)
    u0 = gt_u + perturbation * rng.standard_normal()
    v0 = gt_v + perturbation * rng.standard_normal()
    U0 = np.array([u0, v0])

    U, F, step = icgn_solver(
        U0, x0, y0, df_dx, df_dy, mask, img_ref, img_def,
        winsize, tol, max_iter,
    )

    err_u = U[0] - gt_u
    err_v = U[1] - gt_v
    converged = step <= max_iter

    return {
        "x0": x0, "y0": y0,
        "gt_u": gt_u, "gt_v": gt_v,
        "init_u": u0, "init_v": v0,
        "result_u": U[0], "result_v": U[1],
        "F": F,
        "err_u": err_u, "err_v": err_v,
        "err_mag": np.sqrt(err_u**2 + err_v**2),
        "step": step, "converged": converged,
    }


def run_case(
    case_name: str,
    img_ref: np.ndarray,
    img_def: np.ndarray,
    gt_u_field: np.ndarray,
    gt_v_field: np.ndarray,
    perturbations: list[float],
    winsize: int = 32,
):
    """Run IC-GN on a grid of nodes with various perturbation levels."""
    h, w = img_ref.shape
    margin = winsize // 2 + 2
    step = 16
    xs = np.arange(margin, w - margin, step, dtype=np.float64)
    ys = np.arange(margin, h - margin, step, dtype=np.float64)

    print(f"\n{'='*70}")
    print(f"  CASE: {case_name}")
    print(f"  Image: {h}x{w}, winsize={winsize}, grid step={step}")
    print(f"  Nodes: {len(xs)}x{len(ys)} = {len(xs)*len(ys)}")
    print(f"{'='*70}")

    for pert in perturbations:
        results = []
        for iy, y0 in enumerate(ys):
            for ix, x0 in enumerate(xs):
                iy_img = int(round(y0))
                ix_img = int(round(x0))
                gt_u = gt_u_field[iy_img, ix_img]
                gt_v = gt_v_field[iy_img, ix_img]

                r = run_single_node_test(
                    case_name, img_ref, img_def,
                    x0, y0, gt_u, gt_v,
                    perturbation=pert,
                    winsize=winsize,
                    seed=iy * len(xs) + ix,
                )
                results.append(r)

        n_total = len(results)
        converged = [r for r in results if r["converged"]]
        n_conv = len(converged)
        failed = [r for r in results if not r["converged"]]

        print(f"\n  Perturbation: ±{pert:.2f} px")
        print(f"  Converged: {n_conv}/{n_total} ({100*n_conv/n_total:.1f}%)")

        if converged:
            iters = [r["step"] for r in converged]
            errs = [r["err_mag"] for r in converged]
            err_u = [abs(r["err_u"]) for r in converged]
            err_v = [abs(r["err_v"]) for r in converged]
            print(f"  Iterations: min={min(iters)}, max={max(iters)}, "
                  f"mean={np.mean(iters):.1f}, median={np.median(iters):.1f}")
            print(f"  RMSE_u = {np.sqrt(np.mean(np.array(err_u)**2)):.6f} px")
            print(f"  RMSE_v = {np.sqrt(np.mean(np.array(err_v)**2)):.6f} px")
            print(f"  Max |err| = {max(errs):.6f} px")

        if failed:
            print(f"\n  FAILED nodes ({len(failed)}):")
            # Show a few examples
            for r in failed[:5]:
                print(f"    ({r['x0']:.0f},{r['y0']:.0f}): "
                      f"gt=({r['gt_u']:.3f},{r['gt_v']:.3f}), "
                      f"init=({r['init_u']:.3f},{r['init_v']:.3f}), "
                      f"result=({r['result_u']:.3f},{r['result_v']:.3f}), "
                      f"err={r['err_mag']:.4f}, step={r['step']}")
            if len(failed) > 5:
                print(f"    ... and {len(failed)-5} more")

        # Show iteration distribution
        if converged:
            iters = [r["step"] for r in converged]
            for thresh in [1, 2, 3, 5, 10, 20]:
                n = sum(1 for i in iters if i <= thresh)
                print(f"    <= {thresh:2d} iters: {n:4d} ({100*n/n_conv:.1f}%)")


def main():
    h, w = 256, 256
    print("Generating reference speckle (256x256, sigma=3.0)...")
    ref = generate_speckle(h, w, sigma=3.0, seed=42)

    perturbations = [0.0, 0.1, 0.5, 1.0]

    # --- Case 2: Pure translation (2.5, 1.5) ---
    print("\n" + "="*70)
    print("  GENERATING: case2_translation (ux=2.5, vy=1.5)")
    u_field = np.full((h, w), 2.5)
    v_field = np.full((h, w), 1.5)
    img_def = apply_displacement(ref, u_field, v_field)
    run_case("case2_translation", ref, img_def, u_field, v_field, perturbations)

    # --- Case 3: Affine (2% stretch + 1% shear) ---
    print("\n" + "="*70)
    print("  GENERATING: case3_affine (2% stretch + 1% shear)")
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u_field = 0.02 * (xx - cx) + 0.01 * (yy - cy)
    v_field = 0.01 * (xx - cx) + 0.02 * (yy - cy)
    img_def = apply_displacement(ref, u_field, v_field)
    run_case("case3_affine", ref, img_def, u_field, v_field, perturbations)

    # --- Case 3b: Affine with different apply_displacement orders ---
    print("\n" + "="*70)
    print("  TEST: apply_displacement order sensitivity")
    for order in [3, 5, 7]:
        img_def_o = apply_displacement(ref, u_field, v_field, order=order)
        # Single center node test
        x0, y0 = 128.0, 128.0
        gt_u = 0.02 * (x0 - cx) + 0.01 * (y0 - cy)
        gt_v = 0.01 * (x0 - cx) + 0.02 * (y0 - cy)
        r = run_single_node_test(
            f"affine_order{order}", ref, img_def_o,
            x0, y0, gt_u, gt_v,
            perturbation=0.5, winsize=32, seed=999,
        )
        print(f"  order={order}: converged={r['converged']}, step={r['step']}, "
              f"err={r['err_mag']:.6f}")


if __name__ == "__main__":
    main()
