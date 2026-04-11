#!/usr/bin/env python
"""Comprehensive comparison: Local DIC (ICGN) vs AL-DIC (ADMM).

After the g_img masking fix, tests multiple deformation fields × masks.

Deformations:
    1. Translation (5px, 3px)
    2. Affine (F11=0.02, F12=0.005, F21=-0.005, F22=0.015)
    3. Strong affine (F11=0.04, F12=0.01, F21=-0.01, F22=0.03)
    4. Quadratic (parabolic)
    5. Sinusoidal

Masks:
    A. np.ones (no mask)
    B. Square margin=32 (boundary between grid nodes)
    C. Square margin=16 (boundary at grid edge)
    D. Annular (center hole)

Methods:
    - Local DIC (ICGN only)
    - AL-DIC (ADMM, auto beta)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.core.pipeline import run_aldic

IMG_H, IMG_W = 512, 512
CX, CY = IMG_W / 2.0, IMG_H / 2.0
STEP = 16


def speckle(h, w, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((h, w)), sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def warp(ref, uf, vf):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(20):
        X = xx - uf(X, Y); Y = yy - vf(X, Y)
    return map_coordinates(ref, [Y.ravel(), X.ravel()], order=5, mode="nearest").reshape(h, w)


# Deformation fields
DEFORMATIONS = {
    "Translation": {
        "u": lambda x, y: np.full_like(x, 5.0),
        "v": lambda x, y: np.full_like(x, 3.0),
    },
    "Affine": {
        "u": lambda x, y: 0.020 * (x - CX) + 0.005 * (y - CY),
        "v": lambda x, y: -0.005 * (x - CX) + 0.015 * (y - CY),
    },
    "Strong affine": {
        "u": lambda x, y: 0.040 * (x - CX) + 0.010 * (y - CY),
        "v": lambda x, y: -0.010 * (x - CX) + 0.030 * (y - CY),
    },
    "Quadratic": {
        "u": lambda x, y: 8e-5 * (x - CX) ** 2 + 4e-5 * (y - CY) ** 2,
        "v": lambda x, y: 4e-5 * (x - CX) ** 2 + 6e-5 * (y - CY) ** 2,
    },
    "Sinusoidal": {
        "u": lambda x, y: 3.0 * np.sin(2 * np.pi * x / IMG_W) * np.cos(np.pi * y / IMG_H),
        "v": lambda x, y: 2.0 * np.cos(np.pi * x / IMG_W) * np.sin(2 * np.pi * y / IMG_H),
    },
}

# Masks
def make_masks():
    masks = {}
    masks["np.ones"] = np.ones((IMG_H, IMG_W), dtype=np.float64)

    m32 = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    m32[32:IMG_H - 32, 32:IMG_W - 32] = 1.0
    masks["Square m=32"] = m32

    m16 = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    m16[STEP:IMG_H - STEP, STEP:IMG_W - STEP] = 1.0
    masks["Square m=16"] = m16

    annular = np.ones((IMG_H, IMG_W), dtype=np.float64)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    r = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
    annular[r < 80] = 0.0
    masks["Annular r=80"] = annular

    return masks


def rmse_interior(U, coords, uf, vf, margin=48):
    u_gt = uf(coords[:, 0], coords[:, 1])
    v_gt = vf(coords[:, 0], coords[:, 1])
    ok = ((coords[:, 0] >= margin) & (coords[:, 0] <= IMG_W - 1 - margin) &
          (coords[:, 1] >= margin) & (coords[:, 1] <= IMG_H - 1 - margin) &
          np.isfinite(U[0::2]) & np.isfinite(U[1::2]))
    if not np.any(ok):
        return np.inf, np.inf
    return (
        float(np.sqrt(np.mean((U[0::2][ok] - u_gt[ok]) ** 2))),
        float(np.sqrt(np.mean((U[1::2][ok] - v_gt[ok]) ** 2))),
    )


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ref_raw = speckle(IMG_H, IMG_W)
    masks = make_masks()
    roi = GridxyROIRange(gridx=(STEP, IMG_W - 1 - STEP), gridy=(STEP, IMG_H - 1 - STEP))

    # Run all combinations
    all_results = {}
    total = len(DEFORMATIONS) * len(masks) * 2
    count = 0

    for def_name, def_funcs in DEFORMATIONS.items():
        uf, vf = def_funcs["u"], def_funcs["v"]
        def_raw = warp(ref_raw, uf, vf)

        for mask_name, mask in masks.items():
            for method, use_global in [("ICGN", False), ("AL-DIC", True)]:
                count += 1
                label = f"{def_name} | {mask_name} | {method}"
                print(f"[{count}/{total}] {label}...")

                para = dicpara_default(
                    winsize=32, winstepsize=STEP, winsize_min=8,
                    img_size=(IMG_H, IMG_W), gridxy_roi_range=roi,
                    reference_mode="accumulative", show_plots=False,
                    admm_max_iter=10, admm_tol=1e-4,
                    use_global_step=use_global,
                )
                r = run_aldic(para, [ref_raw, def_raw], [mask, mask], compute_strain=False)
                coords = r.dic_mesh.coordinates_fem
                U = r.result_disp[0].U
                ru, rv = rmse_interior(U, coords, uf, vf)
                print(f"  -> u={ru:.5f}, v={rv:.5f}")

                all_results[label] = {
                    "def": def_name, "mask": mask_name, "method": method,
                    "coords": coords, "U": U, "rmse": (ru, rv),
                    "uf": uf, "vf": vf,
                }

    # ===================================================================
    # PDF Report
    # ===================================================================
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "local_vs_aldic_final.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # Page 1: Grand summary table
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

        mask_names = list(masks.keys())
        def_names = list(DEFORMATIONS.keys())

        # Build table: rows = deformation × mask, cols = ICGN, AL-DIC, improvement
        table_data = []
        for def_name in def_names:
            for mask_name in mask_names:
                icgn_key = f"{def_name} | {mask_name} | ICGN"
                aldic_key = f"{def_name} | {mask_name} | AL-DIC"
                icgn_u = all_results[icgn_key]["rmse"][0]
                aldic_u = all_results[aldic_key]["rmse"][0]

                if icgn_u > 0 and icgn_u < 100:
                    imp = (1 - aldic_u / icgn_u) * 100
                    imp_str = f"{imp:+.1f}%"
                else:
                    imp_str = "N/A"
                    imp = 0

                table_data.append([
                    def_name, mask_name,
                    f"{icgn_u:.5f}", f"{aldic_u:.5f}", imp_str,
                ])

        table = ax.table(
            cellText=table_data,
            colLabels=["Deformation", "Mask", "ICGN u RMSE", "AL-DIC u RMSE", "Improvement"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.4)

        for i, row in enumerate(table_data):
            cell = table[i + 1, 4]
            imp_str = row[4]
            if imp_str.startswith("+"):
                cell.set_facecolor("#ddffdd")
            elif imp_str.startswith("-"):
                cell.set_facecolor("#ffdddd")

        fig.suptitle(
            "Local DIC (ICGN) vs AL-DIC (ADMM) — After g_img Fix\n"
            f"(512×512 images, step={STEP}, winsize=32, 10 ADMM iters, auto beta)",
            fontsize=13, fontweight="bold", y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2+: Per-deformation spatial error maps
        for def_name in def_names:
            fig, axes = plt.subplots(len(mask_names), 2, figsize=(14, 4 * len(mask_names)))
            fig.suptitle(f"{def_name}: ICGN vs AL-DIC Spatial Error",
                         fontsize=13, fontweight="bold")

            for row, mask_name in enumerate(mask_names):
                for col, method in enumerate(["ICGN", "AL-DIC"]):
                    key = f"{def_name} | {mask_name} | {method}"
                    r = all_results[key]
                    coords = r["coords"]
                    uf, vf = r["uf"], r["vf"]
                    u_gt = uf(coords[:, 0], coords[:, 1])
                    err = np.abs(r["U"][0::2] - u_gt)
                    err = np.where(np.isfinite(err), err, 0.0)
                    ru = r["rmse"][0]

                    ax = axes[row, col] if len(mask_names) > 1 else axes[col]
                    sc = ax.scatter(coords[:, 0], coords[:, 1], c=err, s=2,
                                    vmin=0, vmax=0.02, cmap="hot")
                    ax.set_title(f"{mask_name} | {method} (u={ru:.5f})", fontsize=9)
                    ax.set_aspect("equal"); ax.invert_yaxis()
                    plt.colorbar(sc, ax=ax, shrink=0.7)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Bar chart: improvement percentage by deformation
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(def_names))
        width = 0.2

        for i, mask_name in enumerate(mask_names):
            imps = []
            for def_name in def_names:
                icgn_key = f"{def_name} | {mask_name} | ICGN"
                aldic_key = f"{def_name} | {mask_name} | AL-DIC"
                icgn_u = all_results[icgn_key]["rmse"][0]
                aldic_u = all_results[aldic_key]["rmse"][0]
                imp = (1 - aldic_u / icgn_u) * 100 if icgn_u > 0 else 0
                imps.append(imp)
            bars = ax.bar(x + i * width, imps, width, label=mask_name, alpha=0.8)

        ax.set_xticks(x + width * (len(mask_names) - 1) / 2)
        ax.set_xticklabels(def_names, fontsize=9)
        ax.set_ylabel("AL-DIC improvement over ICGN (%)")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("AL-DIC vs ICGN: Improvement by Deformation × Mask", fontsize=13, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")


if __name__ == "__main__":
    main()
