"""Trace IC-GN iteration step-by-step for a single node to find bugs."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.solver.icgn_warp import compose_warp


def generate_speckle(h=256, w=256, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement(ref, u_field, v_field):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src_y = yy - v_field
    src_x = xx - u_field
    coords = np.array([src_y.ravel(), src_x.ravel()])
    warped = map_coordinates(ref, coords, order=5, mode="constant", cval=0.0)
    return warped.reshape(h, w)


def connected_center_mask(binary_mask):
    ny, nx = binary_mask.shape
    cy, cx = ny // 2, nx // 2
    labeled, _ = label(binary_mask)
    center_label = labeled[cy, cx]
    if center_label == 0:
        return np.zeros_like(binary_mask, dtype=np.float64)
    return (labeled == center_label).astype(np.float64)


def trace_icgn(
    U0, x0, y0, df_dx, df_dy, img_ref_mask, img_ref, img_def,
    winsize=32, tol=1e-2, max_iter=20,
):
    """IC-GN with full iteration trace."""
    h, w = img_ref.shape
    half_w = winsize // 2

    P = np.array([0.0, 0.0, 0.0, 0.0, U0[0], U0[1]], dtype=np.float64)
    print(f"  Initial P = [{P[0]:.6f}, {P[1]:.6f}, {P[2]:.6f}, {P[3]:.6f}, {P[4]:.6f}, {P[5]:.6f}]")

    x_lo, x_hi = int(x0 - half_w), int(x0 + half_w)
    y_lo, y_hi = int(y0 - half_w), int(y0 + half_w)
    print(f"  Subset: x=[{x_lo},{x_hi}], y=[{y_lo},{y_hi}]")

    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        print("  OUT OF BOUNDS")
        return P

    tempf_mask = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
    tempf = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1] * tempf_mask
    grad_x = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1]
    grad_y = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1]

    n_masked = np.sum(np.abs(tempf) < 1e-10)
    mask_frac = n_masked / tempf.size
    print(f"  Mask fraction (zeros): {mask_frac:.3f} ({n_masked}/{tempf.size})")

    bw_mask = connected_center_mask(tempf_mask > 0.5)
    tempf = tempf * bw_mask
    grad_x = grad_x * bw_mask
    grad_y = grad_y * bw_mask

    ny, nx = tempf.shape
    xx = np.arange(x_lo, x_hi + 1, dtype=np.float64)
    yy = np.arange(y_lo, y_hi + 1, dtype=np.float64)
    XX_rel = (xx - x0)[np.newaxis, :]
    YY_rel = (yy - y0)[:, np.newaxis]
    XX = np.broadcast_to(XX_rel, (ny, nx)).copy()
    YY = np.broadcast_to(YY_rel, (ny, nx)).copy()

    # Hessian
    gx2 = grad_x ** 2
    gy2 = grad_y ** 2
    gxgy = grad_x * grad_y
    XX2 = XX ** 2
    YY2 = YY ** 2
    XXYY = XX * YY

    H = np.zeros((6, 6), dtype=np.float64)
    H[0, 0] = np.sum(XX2 * gx2)
    H[0, 1] = np.sum(XX2 * gxgy)
    H[0, 2] = np.sum(XXYY * gx2)
    H[0, 3] = np.sum(XXYY * gxgy)
    H[0, 4] = np.sum(XX * gx2)
    H[0, 5] = np.sum(XX * gxgy)
    H[1, 1] = np.sum(XX2 * gy2)
    H[1, 2] = H[0, 3]
    H[1, 3] = np.sum(XXYY * gy2)
    H[1, 4] = H[0, 5]
    H[1, 5] = np.sum(XX * gy2)
    H[2, 2] = np.sum(YY2 * gx2)
    H[2, 3] = np.sum(YY2 * gxgy)
    H[2, 4] = np.sum(YY * gx2)
    H[2, 5] = np.sum(YY * gxgy)
    H[3, 3] = np.sum(YY2 * gy2)
    H[3, 4] = H[2, 5]
    H[3, 5] = np.sum(YY * gy2)
    H[4, 4] = np.sum(gx2)
    H[4, 5] = np.sum(gxgy)
    H[5, 5] = np.sum(gy2)
    H = H + H.T - np.diag(np.diag(H))

    print(f"  H condition number: {np.linalg.cond(H):.2e}")
    print(f"  H diagonal: [{H[0,0]:.2f}, {H[1,1]:.2f}, {H[2,2]:.2f}, {H[3,3]:.2f}, {H[4,4]:.2f}, {H[5,5]:.2f}]")

    valid = np.abs(tempf) > 1e-10
    n_valid = valid.sum()
    meanf = np.mean(tempf[valid])
    varf = np.var(tempf[valid])
    bottomf = np.sqrt(max((n_valid - 1) * varf, 1e-30))
    print(f"  n_valid={n_valid}, meanf={meanf:.4f}, varf={varf:.4f}, bottomf={bottomf:.4f}")

    norm_new = 1.0
    norm_abs = 1.0
    norm_init = None

    print(f"\n  {'Step':>4} | {'norm_abs':>12} | {'norm_new':>12} | {'||dP||':>12} | {'||b_raw||':>12} | {'P[4](Ux)':>12} | {'P[5](Uy)':>12} | {'P[0](F11)':>12} | {'ZNSSD':>12}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for step in range(1, max_iter + 1):
        u22 = (1.0 + P[0]) * XX + P[2] * YY + x0 + P[4]
        v22 = P[1] * XX + (1.0 + P[3]) * YY + y0 + P[5]

        margin = 2.5
        if (np.any(u22 < margin) or np.any(u22 > w - 1 - margin) or
                np.any(v22 < margin) or np.any(v22 > h - 1 - margin)):
            print(f"  {step:4d} | OOB: u22=[{u22.min():.2f},{u22.max():.2f}], v22=[{v22.min():.2f},{v22.max():.2f}]")
            break

        tempg = map_coordinates(img_def, [v22.ravel(), u22.ravel()],
                                order=3, mode='constant', cval=0.0)
        tempg = tempg.reshape(ny, nx)

        g_valid = np.abs(tempg) > 1e-10
        combined_mask = bw_mask.astype(bool) & g_valid

        tempf_iter = tempf * combined_mask
        grad_x_iter = grad_x * combined_mask
        grad_y_iter = grad_y * combined_mask
        tempg = tempg * combined_mask

        # Check if mask changed
        if not np.array_equal(combined_mask, bw_mask.astype(bool)):
            # Recompute...
            H_local = np.zeros((6, 6), dtype=np.float64)
            gx = grad_x_iter
            gy = grad_y_iter
            H_local[0, 0] = np.sum(XX**2 * gx**2)
            # (simplified - just flag it)
            print(f"  {step:4d} | MASK CHANGED ({combined_mask.sum()} vs {bw_mask.sum()} pixels)")

        g_nz = np.abs(tempg) > 1e-10
        meang = np.mean(tempg[g_nz])
        varg = np.var(tempg[g_nz])
        bottomg = np.sqrt(max((g_nz.sum() - 1) * varg, 1e-30))

        residual = (tempf_iter - meanf) / bottomf - (tempg - meang) / bottomg
        znssd = np.sum(residual[combined_mask] ** 2)

        b = np.zeros(6, dtype=np.float64)
        b[0] = np.sum(XX * grad_x_iter * residual)
        b[1] = np.sum(XX * grad_y_iter * residual)
        b[2] = np.sum(YY * grad_x_iter * residual)
        b[3] = np.sum(YY * grad_y_iter * residual)
        b[4] = np.sum(grad_x_iter * residual)
        b[5] = np.sum(grad_y_iter * residual)

        b_raw_norm = np.linalg.norm(b)
        b *= bottomf

        norm_abs = np.linalg.norm(b)
        if norm_init is None:
            norm_init = norm_abs
        norm_new = norm_abs / norm_init if norm_init > tol else 0.0

        try:
            delta_P = -np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            print(f"  {step:4d} | SINGULAR H")
            break

        dp_norm = np.linalg.norm(delta_P)

        print(f"  {step:4d} | {norm_abs:12.6f} | {norm_new:12.6f} | {dp_norm:12.8f} | {b_raw_norm:12.8f} | {P[4]:12.6f} | {P[5]:12.6f} | {P[0]:12.8f} | {znssd:12.6f}")

        if norm_new < tol or norm_abs < tol:
            print(f"  >>> CONVERGED (gradient norm)")
            break

        if dp_norm < tol:
            print(f"  >>> CONVERGED (delta_P)")
            break

        result = compose_warp(P, delta_P)
        if result is None:
            print(f"  >>> compose_warp failed!")
            break
        P = result

    print(f"\n  Final P = [{P[0]:.8f}, {P[1]:.8f}, {P[2]:.8f}, {P[3]:.8f}, {P[4]:.8f}, {P[5]:.8f}]")
    return P


def main():
    h, w = 256, 256
    ref = generate_speckle(h, w, sigma=3.0, seed=42)
    mask = np.ones((h, w), dtype=np.float64)
    grads = compute_image_gradient(ref)
    df_dx, df_dy = grads.df_dx, grads.df_dy

    # --- Test 1: Translation at center node ---
    print("\n" + "="*100)
    print("TEST 1: Translation (2.5, 1.5) at center node (128, 128)")
    print("  Initial guess: GT + 0.5px perturbation")
    print("="*100)

    u_field = np.full((h, w), 2.5)
    v_field = np.full((h, w), 1.5)
    img_def = apply_displacement(ref, u_field, v_field)

    U0 = np.array([2.5 + 0.3, 1.5 - 0.2])  # Perturbed
    trace_icgn(U0, 128.0, 128.0, df_dx, df_dy, mask, ref, img_def,
               winsize=32, max_iter=20)

    # --- Test 2: Translation at a FAILING node from diagnostic ---
    print("\n" + "="*100)
    print("TEST 2: Translation (2.5, 1.5) at edge node (98, 18)")
    print("  Initial guess: GT + 0.1px perturbation")
    print("="*100)

    U0 = np.array([2.5 - 0.08, 1.5 - 0.13])
    trace_icgn(U0, 98.0, 18.0, df_dx, df_dy, mask, ref, img_def,
               winsize=32, max_iter=20)

    # --- Test 3: Affine at center node ---
    print("\n" + "="*100)
    print("TEST 3: Affine (2% stretch + 1% shear) at center node (128, 128)")
    print("  Ground truth F: F11-1=0.02, F21=0.01, F12=0.01, F22-1=0.02")
    print("  Initial guess: GT displacement + 0.5px perturbation")
    print("="*100)

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u_aff = 0.02 * (xx - cx) + 0.01 * (yy - cy)
    v_aff = 0.01 * (xx - cx) + 0.02 * (yy - cy)
    img_def_aff = apply_displacement(ref, u_aff, v_aff)

    gt_u_center = 0.02 * (128 - cx) + 0.01 * (128 - cy)
    gt_v_center = 0.01 * (128 - cx) + 0.02 * (128 - cy)
    print(f"  GT at center: u={gt_u_center:.4f}, v={gt_v_center:.4f}")

    U0 = np.array([gt_u_center + 0.3, gt_v_center - 0.2])
    trace_icgn(U0, 128.0, 128.0, df_dx, df_dy, mask, ref, img_def_aff,
               winsize=32, max_iter=20)

    # --- Test 4: Affine at edge node (failing node) ---
    print("\n" + "="*100)
    print("TEST 4: Affine at node (50, 50)")
    print("  Initial guess: GT displacement + 0.3px perturbation")
    print("="*100)

    gt_u_50 = 0.02 * (50 - cx) + 0.01 * (50 - cy)
    gt_v_50 = 0.01 * (50 - cx) + 0.02 * (50 - cy)
    print(f"  GT at (50,50): u={gt_u_50:.4f}, v={gt_v_50:.4f}")

    U0 = np.array([gt_u_50 + 0.3, gt_v_50 - 0.2])
    trace_icgn(U0, 50.0, 50.0, df_dx, df_dy, mask, ref, img_def_aff,
               winsize=32, max_iter=20)

    # --- Test 5: Affine at far edge node ---
    print("\n" + "="*100)
    print("TEST 5: Affine at node (200, 200)")
    print("  Initial guess: GT displacement + 0.3px perturbation")
    print("="*100)

    gt_u_200 = 0.02 * (200 - cx) + 0.01 * (200 - cy)
    gt_v_200 = 0.01 * (200 - cx) + 0.02 * (200 - cy)
    print(f"  GT at (200,200): u={gt_u_200:.4f}, v={gt_v_200:.4f}")

    U0 = np.array([gt_u_200 + 0.3, gt_v_200 - 0.2])
    trace_icgn(U0, 200.0, 200.0, df_dx, df_dy, mask, ref, img_def_aff,
               winsize=32, max_iter=20)


if __name__ == "__main__":
    main()
