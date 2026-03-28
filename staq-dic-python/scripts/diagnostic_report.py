"""Diagnostic report for synthetic integration tests.

Generates per-node convergence maps, displacement error maps, and
full-field visualizations for all test cases.

Usage:
    python scripts/diagnostic_report.py
"""

from __future__ import annotations

import sys
import os
import time

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import (
    DICPara, DICMesh, GridxyROIRange, ImageGradients,
    merge_uv, split_uv,
)
from staq_dic.core.pipeline import run_aldic
from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.solver.icgn_solver import icgn_solver
from staq_dic.solver.local_icgn import local_icgn

from tests.conftest import (
    apply_displacement, generate_speckle,
    make_annular_mask, make_circular_mask, make_mesh_for_image,
)

# ---------------------------------------------------------------------------
# Constants (match test_synthetic.py)
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0
STEP = 16
MARGIN = 16

CASES = {
    "case1_zero": dict(
        u2=lambda x, y: np.zeros_like(x),
        v2=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid", overrides={},
    ),
    "case2_translation": dict(
        u2=lambda x, y: np.full_like(x, 2.5),
        v2=lambda x, y: np.full_like(x, -1.8),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid", overrides={},
    ),
    "case3_affine": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        mask="solid", overrides={},
    ),
    "case5_shear": dict(
        u2=lambda x, y: 0.015 * (y - CY),
        v2=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.015, F21=0.0,
        mask="solid", overrides={},
    ),
    "case6_large_deform": dict(
        u2=lambda x, y: 0.10 * (x - CX) + 0.05 * (y - CY),
        v2=lambda x, y: 0.05 * (x - CX) + 0.10 * (y - CY),
        F11=0.10, F22=0.10, F12=0.05, F21=0.05,
        mask="solid", overrides=dict(winsize=48),
    ),
    "case8_multiframe_accum": dict(
        u2=lambda x, y: np.full_like(x, 1.0),
        v2=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid", overrides={},
    ),
    "case9_local_only": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        mask="solid", overrides=dict(use_global_step=False),
    ),
    "case10_rotation": dict(
        u2=lambda x, y: (
            (x - CX) * (np.cos(np.pi / 90) - 1)
            - (y - CY) * np.sin(np.pi / 90)
        ),
        v2=lambda x, y: (
            (x - CX) * np.sin(np.pi / 90)
            + (y - CY) * (np.cos(np.pi / 90) - 1)
        ),
        F11=np.cos(np.pi / 90) - 1,
        F22=np.cos(np.pi / 90) - 1,
        F12=-np.sin(np.pi / 90),
        F21=np.sin(np.pi / 90),
        mask="solid", overrides={},
    ),
}


def _case_para(**overrides) -> DICPara:
    defaults = dict(
        winsize=32, winstepsize=16, winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative",
        admm_max_iter=3, admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0, disp_smoothness=0.0, smoothness=0.0,
        show_plots=False, icgn_max_iter=50, tol=1e-2, mu=1e-3,
        gauss_pt_order=2, alpha=0.0,
    )
    defaults.update(overrides)
    return dicpara_default(**defaults)


def build_case_data(case_name: str, ref: np.ndarray):
    """Build test data for a single case."""
    case_def = CASES[case_name]
    para = _case_para(**case_def["overrides"])

    if case_def["mask"] == "annular":
        mask_img = make_annular_mask(IMG_H, IMG_W, cx=CX, cy=CY,
                                     r_outer=90.0, r_inner=40.0)
    else:
        mask_img = make_circular_mask(IMG_H, IMG_W, cx=CX, cy=CY, radius=90.0)

    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    u2_field = case_def["u2"](xx, yy)
    v2_field = case_def["v2"](xx, yy)
    deformed = apply_displacement(ref, u2_field, v2_field)

    mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]
    gt_u = case_def["u2"](node_x, node_y)
    gt_v = case_def["v2"](node_x, node_y)
    U0 = merge_uv(gt_u, gt_v)

    return dict(
        para=para, ref=ref, deformed=deformed, mask_img=mask_img,
        mesh=mesh, U0=U0, gt_u=gt_u, gt_v=gt_v, case_def=case_def,
    )


def run_local_icgn_with_diagnostics(data: dict):
    """Run local ICGN and return raw per-node data (before NaN filling)."""
    para = data["para"]
    ref = data["ref"]
    deformed = data["deformed"]
    mask_img = data["mask_img"]
    mesh = data["mesh"]
    U0 = data["U0"].copy()

    # Compute image gradient
    Df = compute_image_gradient(ref, mask_img)

    n_nodes = mesh.coordinates_fem.shape[0]
    winsize = para.winsize
    max_iter = para.icgn_max_iter
    tol = para.tol

    # Run solver per node, collect raw results
    u_arr = np.zeros(n_nodes, dtype=np.float64)
    v_arr = np.zeros(n_nodes, dtype=np.float64)
    f11 = np.zeros(n_nodes, dtype=np.float64)
    f21 = np.zeros(n_nodes, dtype=np.float64)
    f12 = np.zeros(n_nodes, dtype=np.float64)
    f22 = np.zeros(n_nodes, dtype=np.float64)
    conv_iter = np.zeros(n_nodes, dtype=np.int64)

    for j in range(n_nodes):
        x0 = round(mesh.coordinates_fem[j, 0])
        y0 = round(mesh.coordinates_fem[j, 1])

        try:
            U_j, F_j, step = icgn_solver(
                U0[2 * j: 2 * j + 2],
                float(x0), float(y0),
                Df.df_dx, Df.df_dy, Df.img_ref_mask,
                ref, deformed,
                winsize, tol, max_iter,
            )
            u_arr[j] = U_j[0]
            v_arr[j] = U_j[1]
            f11[j] = F_j[0]
            f21[j] = F_j[1]
            f12[j] = F_j[2]
            f22[j] = F_j[3]
            conv_iter[j] = step
        except Exception as e:
            conv_iter[j] = -1
            u_arr[j] = np.nan
            v_arr[j] = np.nan

    return dict(
        u=u_arr, v=v_arr,
        f11=f11, f21=f21, f12=f12, f22=f22,
        conv_iter=conv_iter,
        max_iter=max_iter,
    )


def classify_convergence(conv_iter, max_iter):
    """Classify each node's convergence status."""
    n = len(conv_iter)
    status = np.empty(n, dtype='U20')
    for i in range(n):
        ci = conv_iter[i]
        if ci < 0:
            status[i] = 'exception'
        elif ci == 0:
            status[i] = 'no_iter'
        elif 1 <= ci <= max_iter:
            status[i] = f'converged({ci})'
        elif ci == max_iter + 1:
            status[i] = 'max_iter'
        elif ci == max_iter + 2:
            status[i] = 'masked_oob'
        elif ci == max_iter + 3:
            status[i] = 'empty_subset'
        else:
            status[i] = f'unknown({ci})'
    return status


def node_mask_filter(coords, mask_img):
    """Return boolean array: True for nodes inside the mask."""
    h, w = mask_img.shape
    cx = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
    return mask_img[cy, cx] > 0.5


def generate_report(case_name, data, diag, output_dir):
    """Generate a full diagnostic report for one case."""
    coords = data["mesh"].coordinates_fem
    gt_u, gt_v = data["gt_u"], data["gt_v"]
    mask = data["mask_img"]
    case_def = data["case_def"]

    in_mask = node_mask_filter(coords, mask)
    converged = (diag["conv_iter"] >= 1) & (diag["conv_iter"] <= diag["max_iter"])
    valid = in_mask & converged & np.isfinite(diag["u"]) & np.isfinite(diag["v"])

    err_u = diag["u"] - gt_u
    err_v = diag["v"] - gt_v

    # --- Summary statistics ---
    n_total = len(coords)
    n_in_mask = in_mask.sum()
    n_converged = converged.sum()
    n_oob = (diag["conv_iter"] == diag["max_iter"] + 2).sum()
    n_maxiter = (diag["conv_iter"] == diag["max_iter"] + 1).sum()
    n_empty = (diag["conv_iter"] == diag["max_iter"] + 3).sum()

    conv_good = diag["conv_iter"][converged]

    rmse_u = np.sqrt(np.mean(err_u[valid] ** 2)) if valid.any() else np.inf
    rmse_v = np.sqrt(np.mean(err_v[valid] ** 2)) if valid.any() else np.inf
    max_err_u = np.max(np.abs(err_u[valid])) if valid.any() else np.inf
    max_err_v = np.max(np.abs(err_v[valid])) if valid.any() else np.inf

    # Strain errors (for cases with nonzero gradients)
    f11_err = diag["f11"][valid] - case_def["F11"]
    f21_err = diag["f21"][valid] - case_def["F21"]
    f12_err = diag["f12"][valid] - case_def["F12"]
    f22_err = diag["f22"][valid] - case_def["F22"]

    print(f"\n{'='*70}")
    print(f"  CASE: {case_name}")
    print(f"{'='*70}")
    print(f"  Nodes total:       {n_total}")
    print(f"  Nodes in mask:     {n_in_mask}")
    print(f"  Converged:         {n_converged}")
    print(f"  Max-iter reached:  {n_maxiter}")
    print(f"  OOB/masked out:    {n_oob}")
    print(f"  Empty subset:      {n_empty}")
    print(f"")
    if len(conv_good) > 0:
        print(f"  Convergence iters (converged nodes):")
        print(f"    min={conv_good.min()}, max={conv_good.max()}, "
              f"mean={conv_good.mean():.1f}, median={np.median(conv_good):.1f}")
        # Distribution
        for threshold in [1, 2, 3, 5, 10, 20, 50]:
            count = (conv_good <= threshold).sum()
            pct = 100.0 * count / len(conv_good)
            print(f"    <= {threshold:2d} iters: {count:4d} ({pct:5.1f}%)")
    print(f"")
    print(f"  Displacement RMSE (in-mask converged):")
    print(f"    RMSE_u = {rmse_u:.6f} px")
    print(f"    RMSE_v = {rmse_v:.6f} px")
    print(f"    Max |err_u| = {max_err_u:.6f} px")
    print(f"    Max |err_v| = {max_err_v:.6f} px")
    print(f"")
    if valid.any():
        print(f"  Gradient RMSE (in-mask converged):")
        print(f"    RMSE_F11 = {np.sqrt(np.mean(f11_err**2)):.6f}")
        print(f"    RMSE_F21 = {np.sqrt(np.mean(f21_err**2)):.6f}")
        print(f"    RMSE_F12 = {np.sqrt(np.mean(f12_err**2)):.6f}")
        print(f"    RMSE_F22 = {np.sqrt(np.mean(f22_err**2)):.6f}")

    # Error by distance from center
    dist = np.sqrt((coords[:, 0] - CX) ** 2 + (coords[:, 1] - CY) ** 2)
    print(f"\n  Error by distance from center (in-mask converged):")
    for r_lo, r_hi in [(0, 30), (30, 50), (50, 70), (70, 90), (90, 200)]:
        band = valid & (dist >= r_lo) & (dist < r_hi)
        n_band = band.sum()
        if n_band > 0:
            rmse_band = np.sqrt(np.mean(err_u[band] ** 2 + err_v[band] ** 2))
            conv_band = diag["conv_iter"][band]
            print(f"    dist [{r_lo:3d}, {r_hi:3d}): n={n_band:3d}, "
                  f"RMSE={rmse_band:.4f}, mean_iter={conv_band.mean():.1f}")
        else:
            print(f"    dist [{r_lo:3d}, {r_hi:3d}): n=0")

    # --- Visualization ---
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Diagnostic: {case_name}", fontsize=16, fontweight='bold')
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Reshape node data for scatter plots
    nx = len(np.unique(coords[:, 0]))
    ny = len(np.unique(coords[:, 1]))

    # 1. Reference image + mesh
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(data["ref"], cmap='gray', origin='upper')
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c='red', alpha=0.5)
    ax.set_title("Reference + mesh nodes")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 2. Mask
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mask, cmap='gray', origin='upper')
    ax.set_title("Mask")

    # 3. Convergence iteration map
    ax = fig.add_subplot(gs[0, 2])
    conv_plot = diag["conv_iter"].astype(float).copy()
    conv_plot[conv_plot > diag["max_iter"]] = np.nan  # don't plot failures
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=conv_plot, s=20,
                    cmap='viridis', edgecolors='none')
    # Mark failed nodes
    failed = diag["conv_iter"] > diag["max_iter"]
    if failed.any():
        ax.scatter(coords[failed, 0], coords[failed, 1], s=30,
                   c='red', marker='x', label=f'failed ({failed.sum()})')
        ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='iterations')
    ax.set_title("Convergence iterations")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 4. Convergence histogram
    ax = fig.add_subplot(gs[0, 3])
    if len(conv_good) > 0:
        ax.hist(conv_good, bins=range(0, min(diag["max_iter"] + 2, conv_good.max() + 3)),
                edgecolor='black', alpha=0.7)
    ax.set_xlabel("Iterations to converge")
    ax.set_ylabel("Node count")
    ax.set_title(f"Convergence distribution (n={len(conv_good)})")

    # 5. Ground truth u field
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=gt_u, s=20,
                    cmap='RdBu_r', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='u (px)')
    ax.set_title("Ground truth u")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 6. Computed u field
    ax = fig.add_subplot(gs[1, 1])
    u_plot = diag["u"].copy()
    u_plot[~valid] = np.nan
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=u_plot, s=20,
                    cmap='RdBu_r', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='u (px)')
    ax.set_title("Computed u (valid nodes)")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 7. Error u field
    ax = fig.add_subplot(gs[1, 2])
    err_u_plot = err_u.copy()
    err_u_plot[~valid] = np.nan
    vmax_err = max(np.nanmax(np.abs(err_u_plot)), 0.01)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=err_u_plot, s=20,
                    cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, label='err_u (px)')
    ax.set_title(f"Error u (RMSE={rmse_u:.4f})")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 8. Error v field
    ax = fig.add_subplot(gs[1, 3])
    err_v_plot = err_v.copy()
    err_v_plot[~valid] = np.nan
    vmax_err_v = max(np.nanmax(np.abs(err_v_plot)), 0.01)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=err_v_plot, s=20,
                    cmap='RdBu_r', vmin=-vmax_err_v, vmax=vmax_err_v,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, label='err_v (px)')
    ax.set_title(f"Error v (RMSE={rmse_v:.4f})")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 9. Error magnitude map
    ax = fig.add_subplot(gs[2, 0])
    err_mag = np.sqrt(err_u ** 2 + err_v ** 2)
    err_mag[~valid] = np.nan
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=err_mag, s=20,
                    cmap='hot_r', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|error| (px)')
    ax.set_title("Error magnitude")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 10. F11 error (deformation gradient)
    ax = fig.add_subplot(gs[2, 1])
    f11_err_plot = (diag["f11"] - case_def["F11"]).copy()
    f11_err_plot[~valid] = np.nan
    vmax_f = max(np.nanmax(np.abs(f11_err_plot)), 0.001)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=f11_err_plot, s=20,
                    cmap='RdBu_r', vmin=-vmax_f, vmax=vmax_f,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, label='F11 error')
    ax.set_title(f"F11 error (gt={case_def['F11']:.4f})")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 11. F12 error
    ax = fig.add_subplot(gs[2, 2])
    f12_err_plot = (diag["f12"] - case_def["F12"]).copy()
    f12_err_plot[~valid] = np.nan
    vmax_f12 = max(np.nanmax(np.abs(f12_err_plot)), 0.001)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=f12_err_plot, s=20,
                    cmap='RdBu_r', vmin=-vmax_f12, vmax=vmax_f12,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, label='F12 error')
    ax.set_title(f"F12 error (gt={case_def['F12']:.4f})")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)

    # 12. Error vs distance scatter
    ax = fig.add_subplot(gs[2, 3])
    if valid.any():
        ax.scatter(dist[valid], err_mag[valid], s=8, alpha=0.5)
        ax.set_xlabel("Distance from center (px)")
        ax.set_ylabel("|displacement error| (px)")
        ax.set_title("Error vs distance")
        ax.axhline(y=0.5, color='r', linestyle='--', label='0.5 px', alpha=0.5)
        ax.legend(fontsize=8)

    plt.savefig(os.path.join(output_dir, f"{case_name}.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {case_name}.png")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating reference speckle...")
    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)

    selected = [
        "case1_zero",
        "case2_translation",
        "case3_affine",
        "case5_shear",
        "case6_large_deform",
        "case8_multiframe_accum",
        "case9_local_only",
        "case10_rotation",
    ]

    for case_name in selected:
        print(f"\nProcessing {case_name}...")
        t0 = time.perf_counter()

        data = build_case_data(case_name, ref)
        diag = run_local_icgn_with_diagnostics(data)
        generate_report(case_name, data, diag, output_dir)

        elapsed = time.perf_counter() - t0
        print(f"  Time: {elapsed:.1f}s")

    print(f"\nAll reports saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
