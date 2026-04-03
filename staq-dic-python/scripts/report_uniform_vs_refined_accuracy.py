#!/usr/bin/env python
"""Visual report: Uniform vs Refined mesh accuracy on complex 1024x1024 mask.

Applies a known affine displacement field to a synthetic speckle image,
then solves with both uniform and refined (inner+outer boundary) meshes.
Compares displacement accuracy and error distribution.

Output: reports/uniform_vs_refined_accuracy.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.mesh.refinement import build_refinement_policy

# Reuse mask from the complex mask report
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_complex_mask_refinement import build_complex_mask

# ═══════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════

IMG_H, IMG_W = 1024, 1024
STEP = 16
WINSIZE = 32
HALF_WIN = WINSIZE // 2
MIN_ELEM = 8

# Affine displacement: u = a*(x-cx) + b*(y-cy) + tx
#                      v = c*(x-cx) + d*(y-cy) + ty
CX, CY = IMG_W / 2, IMG_H / 2
A, B, TX = 0.003, 0.001, 0.5   # du/dx, du/dy, translation
C, D, TY = 0.001, 0.002, 0.3   # dv/dx, dv/dy, translation


# ═══════════════════════════════════════════════════════════════════
# Synthetic image generation
# ═══════════════════════════════════════════════════════════════════

def generate_speckle(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Synthetic speckle pattern in [20, 235] range."""
    rng = np.random.default_rng(seed)
    raw = gaussian_filter(rng.standard_normal((h, w)), sigma=3.0)
    return 20.0 + 215.0 * (raw - raw.min()) / (raw.max() - raw.min())


def make_gt_fields(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_field, v_field) ground-truth pixel displacement fields."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u = A * (xx - CX) + B * (yy - CY) + TX
    v = C * (xx - CX) + D * (yy - CY) + TY
    return u, v


def apply_displacement(
    ref: np.ndarray, u_field: np.ndarray, v_field: np.ndarray,
) -> np.ndarray:
    """Create deformed image: g(x,y) = f(x - u, y - v)."""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def extract_gt_at_nodes(
    coords: np.ndarray, u_field: np.ndarray, v_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate GT displacement to node positions using bicubic spline."""
    h, w = u_field.shape
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    spl_u = RectBivariateSpline(ys, xs, u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(ys, xs, v_field, kx=3, ky=3)
    u_gt = spl_u.ev(coords[:, 1], coords[:, 0])
    v_gt = spl_v.ev(coords[:, 1], coords[:, 0])
    return u_gt, v_gt


# ═══════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════

def draw_field_on_mesh(ax, mesh, values, mask, title, cmap="RdBu_r",
                       vmin=None, vmax=None):
    """Draw a scalar field (one value per node) on the mesh elements.

    Each element is colored by the mean of its 4 corner node values.
    """
    h, w = IMG_H, IMG_W
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    ax.imshow(mask, cmap="gray", alpha=0.15, origin="upper",
              extent=[0, w, h, 0])

    elem_vals = []
    polys = []
    for e in range(elems.shape[0]):
        c4 = elems[e, :4]
        if np.any(c4 < 0):
            continue
        node_vals = values[c4]
        if np.any(np.isnan(node_vals)):
            continue
        polys.append(np.vstack([coords[c4], coords[c4[0]]]))
        elem_vals.append(np.mean(node_vals))

    if not polys:
        ax.set_title(f"{title}\n(no valid elements)", fontsize=8)
        return

    elem_vals = np.array(elem_vals)
    if vmin is None:
        vmin = np.nanpercentile(elem_vals, 1)
    if vmax is None:
        vmax = np.nanpercentile(elem_vals, 99)
    norm = plt.Normalize(vmin, vmax)
    colors = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(elem_vals)

    for poly, color in zip(polys, colors):
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color,
                edgecolor="gray", linewidth=0.1, alpha=0.85)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)


def mesh_info(mesh) -> str:
    """One-line mesh summary."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem
    valid = np.any(elems >= 0, axis=1)
    return f"{coords.shape[0]} nodes, {int(valid.sum())} elems"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "uniform_vs_refined_accuracy.pdf"

    # ── Generate inputs ───────────────────────────────────────────
    print("Building mask and synthetic images ...")
    mask = build_complex_mask()
    ref_img = generate_speckle(IMG_H, IMG_W, seed=42)
    u_gt_field, v_gt_field = make_gt_fields(IMG_H, IMG_W)
    def_img = apply_displacement(ref_img, u_gt_field, v_gt_field)

    images = [ref_img, def_img]
    masks = [mask, mask]

    # ── DIC parameters ────────────────────────────────────────────
    roi = GridxyROIRange(
        gridx=(STEP, IMG_W - 1 - STEP),
        gridy=(STEP, IMG_H - 1 - STEP),
    )
    para = dicpara_default(
        winsize=WINSIZE,
        winstepsize=STEP,
        winsize_min=STEP,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=roi,
        tol=1e-3,
        icgn_max_iter=50,
        mu=1e-3,
        admm_max_iter=10,
        reference_mode="accumulative",
        show_plots=False,
    )

    # ── Run 1: Uniform mesh ──────────────────────────────────────
    print("Running pipeline with UNIFORM mesh ...")
    result_uni = run_aldic(
        para, images, masks, compute_strain=False,
        refinement_policy=None,
    )

    # ── Run 2: Refined mesh (inner + outer boundary) ─────────────
    print("Running pipeline with REFINED mesh ...")
    policy = build_refinement_policy(
        refine_inner_boundary=True,
        refine_outer_boundary=True,
        min_element_size=MIN_ELEM,
        half_win=HALF_WIN,
    )
    result_ref = run_aldic(
        para, images, masks, compute_strain=False,
        refinement_policy=policy,
    )

    # ── Extract displacement and compute errors ──────────────────
    print("Computing errors ...")
    results = {}
    for label, result in [("Uniform", result_uni), ("Refined", result_ref)]:
        mesh = result.dic_mesh
        coords = mesh.coordinates_fem
        U = result.result_disp[0].U
        u_est = U[0::2]
        v_est = U[1::2]

        u_gt, v_gt = extract_gt_at_nodes(coords, u_gt_field, v_gt_field)

        u_err = u_est - u_gt
        v_err = v_est - v_gt

        # Filter valid nodes (non-NaN)
        valid = ~np.isnan(u_est) & ~np.isnan(v_est)

        rmse_u = np.sqrt(np.nanmean(u_err[valid] ** 2))
        rmse_v = np.sqrt(np.nanmean(v_err[valid] ** 2))
        rmse_total = np.sqrt(np.nanmean(u_err[valid] ** 2 + v_err[valid] ** 2))
        max_err = np.nanmax(np.sqrt(u_err[valid] ** 2 + v_err[valid] ** 2))

        results[label] = dict(
            mesh=mesh, coords=coords,
            u_est=u_est, v_est=v_est,
            u_gt=u_gt, v_gt=v_gt,
            u_err=u_err, v_err=v_err,
            valid=valid, n_valid=int(valid.sum()),
            rmse_u=rmse_u, rmse_v=rmse_v,
            rmse_total=rmse_total, max_err=max_err,
        )
        print(f"  {label}: {mesh_info(mesh)}, "
              f"RMSE={rmse_total:.4f}px, max={max_err:.4f}px")

    # ── Generate PDF report ──────────────────────────────────────
    print("Generating report ...")

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Input overview ────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        fig.suptitle(
            "Input: Complex 1024×1024 Mask + Affine Displacement Field",
            fontsize=13, y=0.98,
        )

        axes[0].imshow(mask, cmap="gray", origin="upper")
        axes[0].set_title("Mask\n(4 domains, 8 holes)", fontsize=9)

        im1 = axes[1].imshow(
            np.where(mask > 0.5, u_gt_field, np.nan),
            cmap="RdBu_r", origin="upper",
        )
        axes[1].set_title("GT u-field (px)", fontsize=9)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(
            np.where(mask > 0.5, v_gt_field, np.nan),
            cmap="RdBu_r", origin="upper",
        )
        axes[2].set_title("GT v-field (px)", fontsize=9)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Quiver plot (subsampled)
        step_q = 32
        yy_q, xx_q = np.mgrid[0:IMG_H:step_q, 0:IMG_W:step_q]
        u_q = u_gt_field[::step_q, ::step_q]
        v_q = v_gt_field[::step_q, ::step_q]
        m_q = mask[::step_q, ::step_q]
        axes[3].imshow(mask, cmap="gray", alpha=0.3, origin="upper")
        axes[3].quiver(
            xx_q[m_q > 0.5], yy_q[m_q > 0.5],
            u_q[m_q > 0.5], v_q[m_q > 0.5],
            scale=50, width=0.002, color="C0",
        )
        axes[3].set_title("Displacement vectors", fontsize=9)

        for ax in axes:
            ax.tick_params(labelsize=6)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Displacement fields ───────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Estimated Displacement Fields", fontsize=13, y=0.98)

        r_uni = results["Uniform"]
        r_ref = results["Refined"]

        # Shared color ranges from GT
        u_vmin, u_vmax = np.nanmin(r_uni["u_gt"]), np.nanmax(r_uni["u_gt"])
        v_vmin, v_vmax = np.nanmin(r_uni["v_gt"]), np.nanmax(r_uni["v_gt"])

        draw_field_on_mesh(
            axes[0, 0], r_uni["mesh"], r_uni["u_est"], mask,
            f"Uniform — u (px)\n{mesh_info(r_uni['mesh'])}",
            cmap="RdBu_r", vmin=u_vmin, vmax=u_vmax,
        )
        draw_field_on_mesh(
            axes[0, 1], r_ref["mesh"], r_ref["u_est"], mask,
            f"Refined — u (px)\n{mesh_info(r_ref['mesh'])}",
            cmap="RdBu_r", vmin=u_vmin, vmax=u_vmax,
        )
        draw_field_on_mesh(
            axes[1, 0], r_uni["mesh"], r_uni["v_est"], mask,
            f"Uniform — v (px)\n{mesh_info(r_uni['mesh'])}",
            cmap="RdBu_r", vmin=v_vmin, vmax=v_vmax,
        )
        draw_field_on_mesh(
            axes[1, 1], r_ref["mesh"], r_ref["v_est"], mask,
            f"Refined — v (px)\n{mesh_info(r_ref['mesh'])}",
            cmap="RdBu_r", vmin=v_vmin, vmax=v_vmax,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Error fields ──────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Displacement Error Fields (estimated − GT)",
                     fontsize=13, y=0.98)

        # Symmetric error range
        all_u_err = np.concatenate([
            r_uni["u_err"][r_uni["valid"]],
            r_ref["u_err"][r_ref["valid"]],
        ])
        all_v_err = np.concatenate([
            r_uni["v_err"][r_uni["valid"]],
            r_ref["v_err"][r_ref["valid"]],
        ])
        ue_lim = max(abs(np.nanpercentile(all_u_err, 1)),
                     abs(np.nanpercentile(all_u_err, 99)))
        ve_lim = max(abs(np.nanpercentile(all_v_err, 1)),
                     abs(np.nanpercentile(all_v_err, 99)))

        draw_field_on_mesh(
            axes[0, 0], r_uni["mesh"], r_uni["u_err"], mask,
            f"Uniform — u error\nRMSE={r_uni['rmse_u']:.4f} px",
            cmap="RdBu_r", vmin=-ue_lim, vmax=ue_lim,
        )
        draw_field_on_mesh(
            axes[0, 1], r_ref["mesh"], r_ref["u_err"], mask,
            f"Refined — u error\nRMSE={r_ref['rmse_u']:.4f} px",
            cmap="RdBu_r", vmin=-ue_lim, vmax=ue_lim,
        )
        draw_field_on_mesh(
            axes[1, 0], r_uni["mesh"], r_uni["v_err"], mask,
            f"Uniform — v error\nRMSE={r_uni['rmse_v']:.4f} px",
            cmap="RdBu_r", vmin=-ve_lim, vmax=ve_lim,
        )
        draw_field_on_mesh(
            axes[1, 1], r_ref["mesh"], r_ref["v_err"], mask,
            f"Refined — v error\nRMSE={r_ref['rmse_v']:.4f} px",
            cmap="RdBu_r", vmin=-ve_lim, vmax=ve_lim,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Error distribution + summary ──────────────────
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Error Distribution Comparison", fontsize=13, y=0.98)

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # Histograms
        bins = np.linspace(-0.5, 0.5, 80)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(r_uni["u_err"][r_uni["valid"]], bins=bins,
                 alpha=0.6, label="Uniform", color="C0", density=True)
        ax1.hist(r_ref["u_err"][r_ref["valid"]], bins=bins,
                 alpha=0.6, label="Refined", color="C1", density=True)
        ax1.set_xlabel("u error (px)", fontsize=8)
        ax1.set_ylabel("Density", fontsize=8)
        ax1.set_title("u-error distribution", fontsize=9)
        ax1.legend(fontsize=7)
        ax1.tick_params(labelsize=6)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(r_uni["v_err"][r_uni["valid"]], bins=bins,
                 alpha=0.6, label="Uniform", color="C0", density=True)
        ax2.hist(r_ref["v_err"][r_ref["valid"]], bins=bins,
                 alpha=0.6, label="Refined", color="C1", density=True)
        ax2.set_xlabel("v error (px)", fontsize=8)
        ax2.set_title("v-error distribution", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=6)

        # Total error magnitude histogram
        mag_uni = np.sqrt(
            r_uni["u_err"][r_uni["valid"]] ** 2
            + r_uni["v_err"][r_uni["valid"]] ** 2
        )
        mag_ref = np.sqrt(
            r_ref["u_err"][r_ref["valid"]] ** 2
            + r_ref["v_err"][r_ref["valid"]] ** 2
        )
        bins_mag = np.linspace(0, 0.5, 60)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(mag_uni, bins=bins_mag, alpha=0.6,
                 label="Uniform", color="C0", density=True)
        ax3.hist(mag_ref, bins=bins_mag, alpha=0.6,
                 label="Refined", color="C1", density=True)
        ax3.set_xlabel("|error| (px)", fontsize=8)
        ax3.set_title("Error magnitude distribution", fontsize=9)
        ax3.legend(fontsize=7)
        ax3.tick_params(labelsize=6)

        # Summary table
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis("off")

        col_labels = [
            "Mesh", "Nodes", "Elements", "Valid Nodes",
            "RMSE u (px)", "RMSE v (px)", "RMSE total (px)",
            "Max |error| (px)", "Median |error| (px)",
        ]
        rows = []
        for label in ["Uniform", "Refined"]:
            r = results[label]
            mag = np.sqrt(r["u_err"][r["valid"]] ** 2
                          + r["v_err"][r["valid"]] ** 2)
            rows.append([
                label,
                str(r["coords"].shape[0]),
                mesh_info(r["mesh"]).split(",")[1].strip(),
                str(r["n_valid"]),
                f"{r['rmse_u']:.4f}",
                f"{r['rmse_v']:.4f}",
                f"{r['rmse_total']:.4f}",
                f"{r['max_err']:.4f}",
                f"{np.median(mag):.4f}",
            ])

        # Improvement row
        r_u, r_r = results["Uniform"], results["Refined"]
        pct_u = (1 - r_r["rmse_u"] / r_u["rmse_u"]) * 100
        pct_v = (1 - r_r["rmse_v"] / r_u["rmse_v"]) * 100
        pct_t = (1 - r_r["rmse_total"] / r_u["rmse_total"]) * 100
        rows.append([
            "Improvement", "", "", "",
            f"{pct_u:+.1f}%", f"{pct_v:+.1f}%", f"{pct_t:+.1f}%",
            "", "",
        ])

        table = ax_table.table(
            cellText=rows, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)

        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(len(rows)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            if i == len(rows) - 1:
                color = "#FFF2CC"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")


if __name__ == "__main__":
    main()
