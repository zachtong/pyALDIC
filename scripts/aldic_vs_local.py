"""Detailed comparison: AL-DIC (with ADMM global step) vs Local DIC (ICGN only).

Compares displacement accuracy, strain accuracy, and convergence behavior
across multiple deformation cases and perturbation levels.

Usage:
    python scripts/aldic_vs_local.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange, merge_uv
from al_dic.core.pipeline import run_aldic
from conftest import (
    apply_displacement,
    compute_disp_rmse,
    compute_strain_rmse,
    generate_speckle,
    make_circular_mask,
    make_mesh_for_image,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 512, 512
CX, CY = 255.0, 255.0
STEP = 16
MARGIN = 24


# ---------------------------------------------------------------------------
# Test cases: designed to show ALDIC advantage
# ---------------------------------------------------------------------------

CASES = {
    "translation_3.7px": dict(
        u2=lambda x, y: np.full_like(x, 3.7),
        v2=lambda x, y: np.full_like(x, -2.3),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        description="Pure translation — baseline, both methods should be equivalent",
    ),
    "affine_2pct": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        description="2% biaxial stretch — moderate strain, ALDIC helps smooth noise",
    ),
    "combined_3pct": dict(
        u2=lambda x, y: 0.03 * (x - CX) + 0.01 * (y - CY),
        v2=lambda x, y: 0.01 * (x - CX) + 0.02 * (y - CY),
        F11=0.03, F22=0.02, F12=0.01, F21=0.01,
        description="3% stretch + 1% shear — complex strain field",
    ),
    "rotation_3deg": dict(
        u2=lambda x, y: (x - CX) * (np.cos(np.pi / 60) - 1) - (y - CY) * np.sin(np.pi / 60),
        v2=lambda x, y: (x - CX) * np.sin(np.pi / 60) + (y - CY) * (np.cos(np.pi / 60) - 1),
        F11=np.cos(np.pi / 60) - 1, F22=np.cos(np.pi / 60) - 1,
        F12=-np.sin(np.pi / 60), F21=np.sin(np.pi / 60),
        description="3-degree rotation — tests off-diagonal strain accuracy",
    ),
    "quadratic_nonuniform": dict(
        u2=lambda x, y: 1e-4 * (x - CX) ** 2 + 0.005 * (y - CY),
        v2=lambda x, y: 1e-4 * (y - CY) ** 2 + 0.005 * (x - CX),
        # Strain varies spatially — average over mask interior is approximate
        F11=0.0, F22=0.0, F12=0.005, F21=0.005,
        description="Quadratic (non-uniform) — ALDIC FEM smoothing should shine",
    ),
}

PERTURBATION_LEVELS = [0.5, 1.5, 3.0]

ADMM_ITERATIONS = [1, 3, 5]


def make_para(use_global_step: bool, admm_max_iter: int = 3):
    return dicpara_default(
        winsize=32, winstepsize=STEP, winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, IMG_W - 1), gridy=(0, IMG_H - 1)),
        reference_mode="accumulative",
        use_global_step=use_global_step,
        admm_max_iter=admm_max_iter, admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0, disp_smoothness=0.0, smoothness=0.0,
        show_plots=False, icgn_max_iter=50, tol=1e-2,
        mu=1e-3, gauss_pt_order=2, alpha=0.0,
    )


def run_single(ref, deformed, mask, mesh, U0, use_global_step, admm_max_iter,
               gt_u2, gt_v2, case_def):
    """Run pipeline and return RMSE results."""
    para = make_para(use_global_step=use_global_step, admm_max_iter=admm_max_iter)
    images = [ref, deformed]
    masks = [mask, mask]

    t0 = time.perf_counter()
    result = run_aldic(
        para, images, masks,
        mesh=mesh, U0=U0,
        compute_strain=True,
    )
    elapsed = time.perf_counter() - t0

    fr = result.result_disp[0]
    U = fr.U_accum if fr.U_accum is not None else fr.U
    rmse_u, rmse_v = compute_disp_rmse(
        U, result.dic_mesh.coordinates_fem, gt_u2, gt_v2, mask,
    )

    max_f_err = np.nan
    strain_rmses = {}
    if result.result_strain:
        sr = result.result_strain[0]
        if sr.dudx is not None:
            strain_rmses = compute_strain_rmse(
                sr,
                case_def["F11"], case_def["F21"],
                case_def["F12"], case_def["F22"],
                result.dic_mesh.coordinates_fem, mask,
            )
            max_f_err = max(strain_rmses.values())

    return dict(
        rmse_u=rmse_u, rmse_v=rmse_v,
        max_f_err=max_f_err,
        strain_rmses=strain_rmses,
        elapsed=elapsed,
    )


def main():
    print(f"=" * 100)
    print(f"  AL-DIC vs Local DIC Comparison")
    print(f"  Image: {IMG_H}x{IMG_W}, Step: {STEP}, Margin: {MARGIN}")
    print(f"=" * 100)

    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)
    mask = make_circular_mask(IMG_H, IMG_W, cx=CX, cy=CY, radius=200.0)
    mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
    n_nodes = len(mesh.coordinates_fem)
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]
    rng = np.random.default_rng(456)

    print(f"  Nodes: {n_nodes}, Elements: {len(mesh.elements_fem)}")
    print()

    # -----------------------------------------------------------------------
    # Part 1: ALDIC (3 ADMM iter) vs Local DIC across cases and perturbations
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  PART 1: ALDIC vs Local DIC — Displacement & Strain Accuracy")
    print(f"{'=' * 100}")

    header = (f"{'Case':>22s} | {'Pert':>4s} | {'Method':>10s} | "
              f"{'RMSE_u':>8s} | {'RMSE_v':>8s} | {'max_F':>8s} | "
              f"{'Improvement':>11s} | {'Time':>6s}")
    print(header)
    print("-" * len(header))

    for case_name, case_def in CASES.items():
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
        u2_field = case_def["u2"](xx, yy)
        v2_field = case_def["v2"](xx, yy)
        deformed = apply_displacement(ref, u2_field, v2_field)
        gt_u2 = case_def["u2"](node_x, node_y)
        gt_v2 = case_def["v2"](node_x, node_y)

        for pert in PERTURBATION_LEVELS:
            pert_u = gt_u2 + pert * rng.standard_normal(n_nodes)
            pert_v = gt_v2 + pert * rng.standard_normal(n_nodes)
            U0 = merge_uv(pert_u, pert_v)

            # Local DIC (use_global_step=False skips ADMM regardless of admm_max_iter)
            r_local = run_single(
                ref, deformed, mask, mesh, U0,
                use_global_step=False, admm_max_iter=1,
                gt_u2=gt_u2, gt_v2=gt_v2, case_def=case_def,
            )

            # ALDIC
            r_aldic = run_single(
                ref, deformed, mask, mesh, U0,
                use_global_step=True, admm_max_iter=3,
                gt_u2=gt_u2, gt_v2=gt_v2, case_def=case_def,
            )

            # Compute improvement
            disp_improvement = (
                (r_local["rmse_u"] - r_aldic["rmse_u"]) / r_local["rmse_u"] * 100
                if r_local["rmse_u"] > 1e-10 else 0.0
            )
            strain_improvement = ""
            if not np.isnan(r_local["max_f_err"]) and r_local["max_f_err"] > 1e-10:
                si = (r_local["max_f_err"] - r_aldic["max_f_err"]) / r_local["max_f_err"] * 100
                strain_improvement = f"{si:+.1f}%"

            print(f"{case_name:>22s} | {pert:4.1f} | {'Local':>10s} | "
                  f"{r_local['rmse_u']:8.4f} | {r_local['rmse_v']:8.4f} | "
                  f"{r_local['max_f_err']:8.5f} | {'(baseline)':>11s} | "
                  f"{r_local['elapsed']:5.1f}s")
            print(f"{'':>22s} | {'':>4s} | {'ALDIC(3)':>10s} | "
                  f"{r_aldic['rmse_u']:8.4f} | {r_aldic['rmse_v']:8.4f} | "
                  f"{r_aldic['max_f_err']:8.5f} | "
                  f"{strain_improvement:>11s} | "
                  f"{r_aldic['elapsed']:5.1f}s")
            print()

    # -----------------------------------------------------------------------
    # Part 2: Effect of ADMM iterations (1, 3, 5) on best case
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  PART 2: Effect of ADMM Iteration Count")
    print(f"  Case: combined_3pct, Perturbation: 1.5 px")
    print(f"{'=' * 100}")

    case_def = CASES["combined_3pct"]
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    u2_field = case_def["u2"](xx, yy)
    v2_field = case_def["v2"](xx, yy)
    deformed = apply_displacement(ref, u2_field, v2_field)
    gt_u2 = case_def["u2"](node_x, node_y)
    gt_v2 = case_def["v2"](node_x, node_y)
    pert_u = gt_u2 + 1.5 * rng.standard_normal(n_nodes)
    pert_v = gt_v2 + 1.5 * rng.standard_normal(n_nodes)
    U0 = merge_uv(pert_u, pert_v)

    print(f"{'Method':>15s} | {'RMSE_u':>8s} | {'RMSE_v':>8s} | "
          f"{'F11':>8s} | {'F21':>8s} | {'F12':>8s} | {'F22':>8s} | {'Time':>6s}")
    print("-" * 90)

    # Local only
    r = run_single(ref, deformed, mask, mesh, U0,
                   use_global_step=False, admm_max_iter=1,
                   gt_u2=gt_u2, gt_v2=gt_v2, case_def=case_def)
    sr = r["strain_rmses"]
    print(f"{'Local':>15s} | {r['rmse_u']:8.4f} | {r['rmse_v']:8.4f} | "
          f"{sr.get('rmse_F11', 0):8.5f} | {sr.get('rmse_F21', 0):8.5f} | "
          f"{sr.get('rmse_F12', 0):8.5f} | {sr.get('rmse_F22', 0):8.5f} | "
          f"{r['elapsed']:5.1f}s")

    # ALDIC with varying iterations
    for n_admm in ADMM_ITERATIONS:
        r = run_single(ref, deformed, mask, mesh, U0,
                       use_global_step=True, admm_max_iter=n_admm,
                       gt_u2=gt_u2, gt_v2=gt_v2, case_def=case_def)
        sr = r["strain_rmses"]
        print(f"{'ALDIC(' + str(n_admm) + ')':>15s} | {r['rmse_u']:8.4f} | {r['rmse_v']:8.4f} | "
              f"{sr.get('rmse_F11', 0):8.5f} | {sr.get('rmse_F21', 0):8.5f} | "
              f"{sr.get('rmse_F12', 0):8.5f} | {sr.get('rmse_F22', 0):8.5f} | "
              f"{r['elapsed']:5.1f}s")

    # -----------------------------------------------------------------------
    # Part 3: Strain detail for rotation case (ALDIC advantage for off-diag)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  PART 3: Detailed Strain Comparison for Rotation (3 deg)")
    print(f"  GT: F11={CASES['rotation_3deg']['F11']:.6f}, "
          f"F22={CASES['rotation_3deg']['F22']:.6f}, "
          f"F12={CASES['rotation_3deg']['F12']:.6f}, "
          f"F21={CASES['rotation_3deg']['F21']:.6f}")
    print(f"{'=' * 100}")

    case_def = CASES["rotation_3deg"]
    u2_field = case_def["u2"](xx, yy)
    v2_field = case_def["v2"](xx, yy)
    deformed = apply_displacement(ref, u2_field, v2_field)
    gt_u2 = case_def["u2"](node_x, node_y)
    gt_v2 = case_def["v2"](node_x, node_y)
    pert_u = gt_u2 + 1.0 * rng.standard_normal(n_nodes)
    pert_v = gt_v2 + 1.0 * rng.standard_normal(n_nodes)
    U0 = merge_uv(pert_u, pert_v)

    print(f"{'Method':>15s} | {'RMSE_u':>8s} | {'RMSE_v':>8s} | "
          f"{'F11':>8s} | {'F21':>8s} | {'F12':>8s} | {'F22':>8s}")
    print("-" * 80)

    for label, gs, n_admm in [("Local", False, 1), ("ALDIC(1)", True, 1),
                                ("ALDIC(3)", True, 3), ("ALDIC(5)", True, 5)]:
        r = run_single(ref, deformed, mask, mesh, U0,
                       use_global_step=gs, admm_max_iter=n_admm,
                       gt_u2=gt_u2, gt_v2=gt_v2, case_def=case_def)
        sr = r["strain_rmses"]
        print(f"{label:>15s} | {r['rmse_u']:8.4f} | {r['rmse_v']:8.4f} | "
              f"{sr.get('rmse_F11', 0):8.5f} | {sr.get('rmse_F21', 0):8.5f} | "
              f"{sr.get('rmse_F12', 0):8.5f} | {sr.get('rmse_F22', 0):8.5f}")

    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
