"""Stress test: IC-GN robustness with larger images and perturbed initial guesses.

Tests the full AL-DIC pipeline on 512x512 images with various perturbation
levels to validate IC-GN convergence under realistic conditions.

Usage:
    python scripts/stress_test_icgn.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange, merge_uv

# Import conftest helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conftest import (
    apply_displacement,
    compute_disp_rmse,
    compute_strain_rmse,
    generate_speckle,
    make_circular_mask,
    make_mesh_for_image,
)

from al_dic.core.pipeline import run_aldic


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 512, 512
CX, CY = 255.0, 255.0
STEP = 16
MARGIN = 24  # Larger margin for 512x512


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES = {
    "translation": dict(
        u2=lambda x, y: np.full_like(x, 3.7),
        v2=lambda x, y: np.full_like(x, -2.3),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        overrides={},
    ),
    "affine_2pct": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        overrides={},
    ),
    "shear_1.5pct": dict(
        u2=lambda x, y: 0.015 * (y - CY),
        v2=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.015, F21=0.0,
        overrides={},
    ),
    "rotation_2deg": dict(
        u2=lambda x, y: (x - CX) * (np.cos(np.pi / 90) - 1) - (y - CY) * np.sin(np.pi / 90),
        v2=lambda x, y: (x - CX) * np.sin(np.pi / 90) + (y - CY) * (np.cos(np.pi / 90) - 1),
        F11=np.cos(np.pi / 90) - 1, F22=np.cos(np.pi / 90) - 1,
        F12=-np.sin(np.pi / 90), F21=np.sin(np.pi / 90),
        overrides={},
    ),
    "combined_stretch_shear": dict(
        u2=lambda x, y: 0.03 * (x - CX) + 0.01 * (y - CY),
        v2=lambda x, y: 0.01 * (x - CX) + 0.02 * (y - CY),
        F11=0.03, F22=0.02, F12=0.01, F21=0.01,
        overrides={},
    ),
}

PERTURBATION_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]


def make_para(**overrides):
    defaults = dict(
        winsize=32, winstepsize=STEP, winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, IMG_W - 1), gridy=(0, IMG_H - 1)),
        reference_mode="accumulative",
        admm_max_iter=3, admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0, disp_smoothness=0.0, smoothness=0.0,
        show_plots=False, icgn_max_iter=50, tol=1e-2,
        mu=1e-3, gauss_pt_order=2, alpha=0.0,
    )
    defaults.update(overrides)
    return dicpara_default(**defaults)


def run_stress_test():
    print(f"Generating reference speckle ({IMG_H}x{IMG_W}, sigma=3.0)...")
    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)
    mask = make_circular_mask(IMG_H, IMG_W, cx=CX, cy=CY, radius=200.0)
    mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
    n_nodes = len(mesh.coordinates_fem)
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]
    print(f"Mesh: {n_nodes} nodes, {len(mesh.elements_fem)} elements\n")

    rng = np.random.default_rng(123)

    print(f"{'Case':>25s} | {'Pert':>5s} | {'RMSE_u':>8s} | {'RMSE_v':>8s} | "
          f"{'max_F_err':>9s} | {'Time':>6s} | {'Status'}")
    print("-" * 90)

    for case_name, case_def in CASES.items():
        # Generate deformed image
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
        u2_field = case_def["u2"](xx, yy)
        v2_field = case_def["v2"](xx, yy)
        deformed = apply_displacement(ref, u2_field, v2_field)

        # Ground truth at nodes
        gt_u2 = case_def["u2"](node_x, node_y)
        gt_v2 = case_def["v2"](node_x, node_y)

        for pert in PERTURBATION_LEVELS:
            # Perturbed initial guess
            pert_u = gt_u2 + pert * rng.standard_normal(n_nodes)
            pert_v = gt_v2 + pert * rng.standard_normal(n_nodes)
            U0 = merge_uv(pert_u, pert_v)

            para = make_para(**case_def["overrides"])
            images = [ref, deformed]
            masks = [mask, mask]

            t0 = time.perf_counter()
            try:
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

                # Strain RMSE
                max_f_err = 0.0
                if result.result_strain:
                    sr = result.result_strain[0]
                    if sr.dudx is not None:
                        rmses = compute_strain_rmse(
                            sr,
                            case_def["F11"], case_def["F21"],
                            case_def["F12"], case_def["F22"],
                            result.dic_mesh.coordinates_fem, mask,
                        )
                        max_f_err = max(rmses.values())

                # Classify result
                if rmse_u < 0.5 and rmse_v < 0.5:
                    status = "PASS"
                elif rmse_u < 1.0 and rmse_v < 1.0:
                    status = "MARGINAL"
                else:
                    status = "FAIL"

                print(f"{case_name:>25s} | {pert:5.1f} | {rmse_u:8.4f} | {rmse_v:8.4f} | "
                      f"{max_f_err:9.5f} | {elapsed:5.1f}s | {status}")

            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"{case_name:>25s} | {pert:5.1f} | {'ERROR':>8s} | {'ERROR':>8s} | "
                      f"{'N/A':>9s} | {elapsed:5.1f}s | {type(e).__name__}: {e}")


if __name__ == "__main__":
    run_stress_test()
