#!/usr/bin/env python
"""Comprehensive comparison: Local DIC vs AL-DIC on square & annular ROIs.

Tests 6 deformation fields x 2 mask types x 2 methods = 24 pipeline runs.
All meshes use automatic MaskBoundaryCriterion refinement — both square and
annular ROI boundaries are refined.

Generates a multi-page PDF report with:
    - Summary table of RMSE (u, v) for all 24 configurations
    - Bar charts comparing local DIC vs AL-DIC per deformation field
    - Spatial error maps for selected cases
    - Mesh visualizations showing adaptive refinement

Usage:
    python scripts/report_local_vs_aldic_comparison.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Ensure staq_dic is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.io.image_ops import normalize_images
from staq_dic.mesh.criteria import MaskBoundaryCriterion
from staq_dic.mesh.refinement import RefinementPolicy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 512, 512
CX, CY = IMG_W / 2.0, IMG_H / 2.0
STEP = 16
ROI_MARGIN = 32  # margin from image edge for ROI boundary
HOLE_R = 80  # annular hole radius (pixels)

# Deformation field definitions: (label, u_func, v_func)
DEFORMATION_FIELDS = [
    (
        "Small translation\n(u=0.5, v=0.3)",
        lambda x, y: np.full_like(x, 0.5),
        lambda x, y: np.full_like(x, 0.3),
    ),
    (
        "Large translation\n(u=5.0, v=3.0)",
        lambda x, y: np.full_like(x, 5.0),
        lambda x, y: np.full_like(x, 3.0),
    ),
    (
        "Mild affine\n(F~0.005)",
        lambda x, y: 0.005 * (x - CX) + 0.002 * (y - CY),
        lambda x, y: -0.002 * (x - CX) + 0.005 * (y - CY),
    ),
    (
        "Strong affine\n(F~0.020)",
        lambda x, y: 0.020 * (x - CX) + 0.005 * (y - CY),
        lambda x, y: -0.005 * (x - CX) + 0.015 * (y - CY),
    ),
    (
        "Rotation ~2deg",
        lambda x, y: (
            np.cos(0.035) * (x - CX) - np.sin(0.035) * (y - CY) - (x - CX)
        ),
        lambda x, y: (
            np.sin(0.035) * (x - CX) + np.cos(0.035) * (y - CY) - (y - CY)
        ),
    ),
    (
        "Sinusoidal\n(A=3, lam=128)",
        lambda x, y: 3.0 * np.sin(2 * np.pi * y / 128.0),
        lambda x, y: np.zeros_like(x),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42):
    """Generate a smooth synthetic speckle pattern."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    f = gaussian_filter(noise, sigma=sigma, mode="nearest")
    f -= f.min()
    f /= f.max()
    return 20.0 + 215.0 * f


def warp_image(ref: np.ndarray, u_func, v_func, n_iter: int = 20):
    """Create a deformed image by inverse-mapping the reference."""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    return map_coordinates(
        ref, [Y.ravel(), X.ravel()], order=5, mode="nearest"
    ).reshape(h, w)


def make_square_mask(h: int, w: int, margin: int) -> np.ndarray:
    """Square ROI mask: 1 inside ROI, 0 outside."""
    mask = np.zeros((h, w), dtype=np.float64)
    mask[margin : h - margin, margin : w - margin] = 1.0
    return mask


def make_annular_mask(h: int, w: int, margin: int, hole_r: int) -> np.ndarray:
    """Annular ROI mask: 1 inside ROI but outside the central hole."""
    mask = make_square_mask(h, w, margin)
    yy, xx = np.ogrid[0:h, 0:w]
    dist = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2)
    mask[dist <= hole_r] = 0.0
    return mask


def rmse_interior(U, coords, u_func, v_func, margin: int = 48):
    """Compute RMSE for interior nodes (away from ROI boundary)."""
    u_gt = u_func(coords[:, 0], coords[:, 1])
    v_gt = v_func(coords[:, 0], coords[:, 1])
    ok = (
        (coords[:, 0] >= margin)
        & (coords[:, 0] <= IMG_W - 1 - margin)
        & (coords[:, 1] >= margin)
        & (coords[:, 1] <= IMG_H - 1 - margin)
        & np.isfinite(U[0::2])
        & np.isfinite(U[1::2])
    )
    if not np.any(ok):
        return np.inf, np.inf
    return (
        float(np.sqrt(np.mean((U[0::2][ok] - u_gt[ok]) ** 2))),
        float(np.sqrt(np.mean((U[1::2][ok] - v_gt[ok]) ** 2))),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # --- Build synthetic reference image ---
    ref_raw = generate_speckle(IMG_H, IMG_W)

    # --- Build masks ---
    mask_square = make_square_mask(IMG_H, IMG_W, ROI_MARGIN)
    mask_annular = make_annular_mask(IMG_H, IMG_W, ROI_MARGIN, HOLE_R)
    mask_configs = [
        ("Square ROI", mask_square),
        ("Annular ROI", mask_annular),
    ]

    # --- Refinement policy (MaskBoundaryCriterion for auto-refinement) ---
    policy = RefinementPolicy(
        pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
    )

    # --- Base DICPara ---
    roi = GridxyROIRange(
        gridx=(STEP, IMG_W - 1 - STEP),
        gridy=(STEP, IMG_H - 1 - STEP),
    )
    base_para = dicpara_default(
        winsize=32,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=roi,
        reference_mode="accumulative",
        show_plots=False,
        admm_max_iter=10,
        admm_tol=1e-4,
    )

    # --- Run all configurations ---
    # results[mask_label][deform_label][method_label] = {rmse, n_nodes, mhe, time}
    results: dict[str, dict[str, dict[str, dict]]] = {}
    total_runs = len(mask_configs) * len(DEFORMATION_FIELDS) * 2
    run_count = 0

    for mask_label, mask in mask_configs:
        results[mask_label] = {}

        for deform_label, u_func, v_func in DEFORMATION_FIELDS:
            results[mask_label][deform_label] = {}

            # Generate deformed image
            def_raw = warp_image(ref_raw, u_func, v_func)

            for method_label, use_global in [
                ("Local DIC", False),
                ("AL-DIC", True),
            ]:
                run_count += 1
                short_label = deform_label.replace("\n", " ")
                print(
                    f"  [{run_count:2d}/{total_runs}] "
                    f"{mask_label} | {short_label} | {method_label} ...",
                    end="",
                    flush=True,
                )

                para = replace(base_para, use_global_step=use_global)
                t0 = time.perf_counter()
                r = run_aldic(
                    para,
                    [ref_raw, def_raw],
                    [mask, mask],
                    compute_strain=False,
                    refinement_policy=policy,
                )
                elapsed = time.perf_counter() - t0

                coords = r.dic_mesh.coordinates_fem
                U = r.result_disp[0].U
                ru, rv = rmse_interior(U, coords, u_func, v_func)
                mhe = len(r.dic_mesh.mark_coord_hole_edge)
                n = coords.shape[0]

                results[mask_label][deform_label][method_label] = {
                    "rmse_u": ru,
                    "rmse_v": rv,
                    "n_nodes": n,
                    "mhe": mhe,
                    "time": elapsed,
                    "coords": coords,
                    "U": U,
                    "mesh": r.dic_mesh,
                }

                print(
                    f"  RMSE: u={ru:.5f}, v={rv:.5f}  "
                    f"({n} nodes, {elapsed:.1f}s)"
                )

    # -------------------------------------------------------------------
    # Generate PDF report
    # -------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "local_vs_aldic_comparison.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # ===============================================================
        # Page 1-2: Summary tables (one per mask type)
        # ===============================================================
        for mask_label in results:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.axis("off")

            headers = [
                "Deformation",
                "Method",
                "Nodes",
                "MHE",
                "u RMSE",
                "v RMSE",
                "Time (s)",
                "AL-DIC vs Local",
            ]
            table_rows = []

            for deform_label in results[mask_label]:
                local_r = results[mask_label][deform_label]["Local DIC"]
                aldic_r = results[mask_label][deform_label]["AL-DIC"]

                for method_label, r in [
                    ("Local DIC", local_r),
                    ("AL-DIC", aldic_r),
                ]:
                    if method_label == "Local DIC":
                        vs = "baseline"
                    else:
                        if local_r["rmse_u"] > 1e-10:
                            ratio = aldic_r["rmse_u"] / local_r["rmse_u"]
                            if ratio < 1:
                                vs = f"{(1 - ratio) * 100:.1f}% better"
                            else:
                                vs = f"{(ratio - 1) * 100:.1f}% worse"
                        else:
                            vs = "N/A"

                    short_deform = deform_label.replace("\n", " ")
                    table_rows.append([
                        short_deform,
                        method_label,
                        str(r["n_nodes"]),
                        str(r["mhe"]),
                        f"{r['rmse_u']:.5f}",
                        f"{r['rmse_v']:.5f}",
                        f"{r['time']:.1f}",
                        vs,
                    ])

            table = ax.table(
                cellText=table_rows,
                colLabels=headers,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.4)

            # Color the comparison column
            for i in range(len(table_rows)):
                cell = table[i + 1, 7]
                txt = table_rows[i][7]
                if "worse" in txt:
                    cell.set_facecolor("#ffdddd")
                elif "better" in txt:
                    cell.set_facecolor("#ddffdd")
                else:
                    cell.set_facecolor("#eeeeee")

                # Alternate row shading for readability
                if i % 2 == 0:
                    for j in range(len(headers)):
                        c = table[i + 1, j]
                        if j != 7:  # don't override comparison color
                            c.set_facecolor("#f8f8ff")

            fig.suptitle(
                f"{mask_label}: Local DIC vs AL-DIC Comparison\n"
                f"(512x512, step={STEP}, auto MaskBoundaryCriterion refinement)",
                fontsize=13,
                fontweight="bold",
                y=0.98,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig)
            plt.close(fig)

        # ===============================================================
        # Page 3-4: Bar charts (RMSE comparison per deformation)
        # ===============================================================
        deform_labels_short = [
            d.replace("\n", " ") for d, _, _ in DEFORMATION_FIELDS
        ]

        for mask_label in results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"{mask_label}: RMSE Comparison",
                fontsize=13,
                fontweight="bold",
            )

            for ax, comp in zip(axes, ["u", "v"]):
                local_vals = []
                aldic_vals = []
                for deform_label in results[mask_label]:
                    local_r = results[mask_label][deform_label]["Local DIC"]
                    aldic_r = results[mask_label][deform_label]["AL-DIC"]
                    local_vals.append(local_r[f"rmse_{comp}"])
                    aldic_vals.append(aldic_r[f"rmse_{comp}"])

                x = np.arange(len(deform_labels_short))
                w = 0.35
                bars1 = ax.bar(
                    x - w / 2,
                    local_vals,
                    w,
                    label="Local DIC",
                    color="#4a90d9",
                    alpha=0.85,
                )
                bars2 = ax.bar(
                    x + w / 2,
                    aldic_vals,
                    w,
                    label="AL-DIC",
                    color="#e8713a",
                    alpha=0.85,
                )

                ax.set_ylabel(f"{comp} RMSE (px)")
                ax.set_title(f"{comp}-displacement RMSE")
                ax.set_xticks(x)
                ax.set_xticklabels(deform_labels_short, rotation=30, ha="right", fontsize=7)
                ax.legend()
                ax.set_yscale("log")
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ===============================================================
        # Page 5+: Spatial error maps (selected cases)
        # ===============================================================
        # Show spatial error for strong affine and sinusoidal (most interesting)
        interesting_deforms = [
            d for d, _, _ in DEFORMATION_FIELDS if "Strong" in d or "Sinusoidal" in d
        ]

        for mask_label in results:
            for deform_label in interesting_deforms:
                if deform_label not in results[mask_label]:
                    continue

                u_func = v_func = None
                for dl, uf, vf in DEFORMATION_FIELDS:
                    if dl == deform_label:
                        u_func, v_func = uf, vf
                        break

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                short_deform = deform_label.replace("\n", " ")
                fig.suptitle(
                    f"{mask_label} | {short_deform}: Spatial Error Maps",
                    fontsize=13,
                    fontweight="bold",
                )

                for row, method_label in enumerate(["Local DIC", "AL-DIC"]):
                    r = results[mask_label][deform_label][method_label]
                    coords = r["coords"]
                    U = r["U"]
                    u_gt = u_func(coords[:, 0], coords[:, 1])
                    v_gt = v_func(coords[:, 0], coords[:, 1])
                    err_u = np.abs(U[0::2] - u_gt)
                    err_v = np.abs(U[1::2] - v_gt)

                    # Clip NaN for visualization
                    err_u = np.where(np.isfinite(err_u), err_u, 0.0)
                    err_v = np.where(np.isfinite(err_v), err_v, 0.0)

                    vmax = max(0.01, np.percentile(np.concatenate([err_u, err_v]), 95))

                    for col, (err, comp) in enumerate(
                        zip([err_u, err_v], ["u", "v"])
                    ):
                        ax = axes[row, col]
                        sc = ax.scatter(
                            coords[:, 0],
                            coords[:, 1],
                            c=err,
                            s=1.5,
                            vmin=0,
                            vmax=vmax,
                            cmap="hot",
                        )
                        ax.set_title(
                            f"{method_label}: |{comp} - {comp}_gt|"
                        )
                        ax.set_aspect("equal")
                        ax.invert_yaxis()
                        plt.colorbar(sc, ax=ax, shrink=0.8, label="error (px)")

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # ===============================================================
        # Page: Mesh visualizations (one per mask type)
        # ===============================================================
        # Show the refined mesh for the strong affine case
        ref_deform = [d for d, _, _ in DEFORMATION_FIELDS if "Strong" in d]
        if ref_deform:
            ref_deform = ref_deform[0]
            for mask_label in results:
                if ref_deform not in results[mask_label]:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(
                    f"{mask_label}: Refined Mesh (Strong Affine)",
                    fontsize=13,
                    fontweight="bold",
                )

                r = results[mask_label][ref_deform]["AL-DIC"]
                mesh = r["mesh"]
                coords = mesh.coordinates_fem
                elems = mesh.elements_fem

                # Plot mesh edges (Q4 corners only)
                ax = axes[0]
                corners = elems[:, :4]
                for e in range(corners.shape[0]):
                    idx = list(corners[e]) + [corners[e, 0]]
                    ax.plot(
                        coords[idx, 0],
                        coords[idx, 1],
                        "b-",
                        linewidth=0.3,
                        alpha=0.6,
                    )
                ax.set_title(f"Mesh ({coords.shape[0]} nodes, {elems.shape[0]} elements)")
                ax.set_aspect("equal")
                ax.invert_yaxis()

                # Plot node density as scatter
                ax = axes[1]
                from staq_dic.strain.smooth_field import compute_node_local_spacing

                spacing = compute_node_local_spacing(coords, elems)
                sc = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=spacing,
                    s=2,
                    cmap="viridis",
                )
                ax.set_title(
                    f"Local spacing (range [{spacing.min():.1f}, {spacing.max():.1f}])"
                )
                ax.set_aspect("equal")
                ax.invert_yaxis()
                plt.colorbar(sc, ax=ax, shrink=0.8, label="spacing (px)")

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # ===============================================================
        # Page: Improvement summary chart
        # ===============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "AL-DIC Improvement over Local DIC (u RMSE)",
            fontsize=13,
            fontweight="bold",
        )

        for ax_idx, mask_label in enumerate(results):
            ax = axes[ax_idx]
            improvements = []
            labels = []
            for deform_label in results[mask_label]:
                local_r = results[mask_label][deform_label]["Local DIC"]
                aldic_r = results[mask_label][deform_label]["AL-DIC"]
                if local_r["rmse_u"] > 1e-10:
                    impr = (1 - aldic_r["rmse_u"] / local_r["rmse_u"]) * 100
                else:
                    impr = 0.0
                improvements.append(impr)
                labels.append(deform_label.replace("\n", " "))

            x = np.arange(len(labels))
            colors = ["#4CAF50" if v > 0 else "#F44336" for v in improvements]
            bars = ax.bar(x, improvements, color=colors, alpha=0.85)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_ylabel("Improvement (%)")
            ax.set_title(mask_label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, improvements):
                y = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + (2 if y >= 0 else -4),
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=8,
                )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")


if __name__ == "__main__":
    main()
