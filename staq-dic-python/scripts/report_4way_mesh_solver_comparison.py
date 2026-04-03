#!/usr/bin/env python
"""Visual report: 2x2 comparison of mesh type × solver type.

Combinations:
  (1) Uniform  + Local DIC  (IC-GN only, no ADMM)
  (2) Uniform  + AL-DIC     (IC-GN + ADMM global regularization)
  (3) Refined  + Local DIC
  (4) Refined  + AL-DIC

Uses the complex 1024×1024 mask with a quadratic displacement field
and Gaussian noise added to both reference and deformed images.

Output: reports/4way_mesh_solver_comparison.pdf
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
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

# Quadratic displacement field centered at (CX, CY):
#   u(x,y) = a1*(x-cx)^2 + a2*(y-cy)^2 + a3*(x-cx)(y-cy)
#          + a4*(x-cx) + a5*(y-cy) + a6
#   v(x,y) = similar with b-coefficients
CX, CY = IMG_W / 2, IMG_H / 2

# Gaussian noise standard deviation (grayscale units, image range ~[20,235])
NOISE_STD = 5.0


# ═══════════════════════════════════════════════════════════════════
# Synthetic helpers
# ═══════════════════════════════════════════════════════════════════

def generate_speckle(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = gaussian_filter(rng.standard_normal((h, w)), sigma=3.0)
    return 20.0 + 215.0 * (raw - raw.min()) / (raw.max() - raw.min())


def add_noise(img: np.ndarray, std: float, rng) -> np.ndarray:
    """Add Gaussian noise and clip to valid range."""
    noisy = img + rng.normal(0.0, std, img.shape)
    return np.clip(noisy, 0.0, 255.0)


def make_gt_fields(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic displacement field with strain concentration near center.

    Peak displacement ~3 px at corners, strong curvature near holes.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = (xx - CX) / CX   # normalized to [-1, 1]
    dy = (yy - CY) / CY

    u = (1.5 * dx**2 + 0.8 * dy**2 + 0.5 * dx * dy
         + 0.3 * dx + 0.1 * dy + 0.5)
    v = (0.6 * dx**2 + 1.2 * dy**2 - 0.4 * dx * dy
         + 0.1 * dx + 0.2 * dy + 0.3)
    return u, v


def apply_displacement(ref, u_field, v_field):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


def extract_gt_at_nodes(coords, u_field, v_field):
    h, w = u_field.shape
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    spl_u = RectBivariateSpline(ys, xs, u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(ys, xs, v_field, kx=3, ky=3)
    return spl_u.ev(coords[:, 1], coords[:, 0]), spl_v.ev(coords[:, 1], coords[:, 0])


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def draw_field(ax, mesh, values, mask, title, cmap="RdBu_r",
               vmin=None, vmax=None):
    """Draw scalar field on mesh elements (mean of 4 corner values)."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    ax.imshow(mask, cmap="gray", alpha=0.15, origin="upper",
              extent=[0, IMG_W, IMG_H, 0])

    elem_vals, polys = [], []
    for e in range(elems.shape[0]):
        c4 = elems[e, :4]
        if np.any(c4 < 0):
            continue
        nv = values[c4]
        if np.any(np.isnan(nv)):
            continue
        polys.append(np.vstack([coords[c4], coords[c4[0]]]))
        elem_vals.append(np.mean(nv))

    if not polys:
        ax.set_title(f"{title}\n(no valid elements)", fontsize=8)
        return

    arr = np.array(elem_vals)
    if vmin is None:
        vmin = np.nanpercentile(arr, 1)
    if vmax is None:
        vmax = np.nanpercentile(arr, 99)
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(arr)

    for poly, c in zip(polys, colors):
        ax.fill(poly[:, 0], poly[:, 1], facecolor=c,
                edgecolor="gray", linewidth=0.1, alpha=0.85)

    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)


def mesh_info(mesh) -> str:
    valid = np.any(mesh.elements_fem >= 0, axis=1)
    return f"{mesh.coordinates_fem.shape[0]} nodes, {int(valid.sum())} elems"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

# Labels for the 4 combinations (row=mesh, col=solver)
COMBOS = [
    ("Uniform + Local",  False, None),
    ("Uniform + AL-DIC", True,  None),
    ("Refined + Local",  False, "refine"),
    ("Refined + AL-DIC", True,  "refine"),
]


def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "4way_mesh_solver_comparison.pdf"

    # ── Inputs ────────────────────────────────────────────────────
    print("Building mask and synthetic images ...")
    mask = build_complex_mask()
    ref_clean = generate_speckle(IMG_H, IMG_W, seed=42)
    u_gt_field, v_gt_field = make_gt_fields(IMG_H, IMG_W)
    def_clean = apply_displacement(ref_clean, u_gt_field, v_gt_field)

    # Add independent Gaussian noise to both images
    rng_noise = np.random.default_rng(123)
    ref_img = add_noise(ref_clean, NOISE_STD, rng_noise)
    def_img = add_noise(def_clean, NOISE_STD, rng_noise)
    print(f"  Noise std={NOISE_STD}, SNR~{ref_clean.std() / NOISE_STD:.0f}")

    images = [ref_img, def_img]
    masks = [mask, mask]

    roi = GridxyROIRange(
        gridx=(STEP, IMG_W - 1 - STEP),
        gridy=(STEP, IMG_H - 1 - STEP),
    )
    para_base = dicpara_default(
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

    policy = build_refinement_policy(
        refine_inner_boundary=True,
        refine_outer_boundary=True,
        min_element_size=MIN_ELEM,
        half_win=HALF_WIN,
    )

    # ── Run 4 combinations ────────────────────────────────────────
    results = {}
    for label, use_global, mesh_type in COMBOS:
        print(f"  Running: {label} ...")
        t0 = time.perf_counter()

        para = replace(para_base, use_global_step=use_global)
        rp = policy if mesh_type == "refine" else None

        result = run_aldic(
            para, images, masks,
            compute_strain=False,
            refinement_policy=rp,
        )

        elapsed = time.perf_counter() - t0
        m = result.dic_mesh
        coords = m.coordinates_fem
        U = result.result_disp[0].U
        u_est, v_est = U[0::2], U[1::2]
        u_gt, v_gt = extract_gt_at_nodes(coords, u_gt_field, v_gt_field)
        u_err, v_err = u_est - u_gt, v_est - v_gt
        valid = ~np.isnan(u_est) & ~np.isnan(v_est)
        mag_err = np.sqrt(u_err[valid] ** 2 + v_err[valid] ** 2)

        results[label] = dict(
            mesh=m, coords=coords,
            u_est=u_est, v_est=v_est,
            u_gt=u_gt, v_gt=v_gt,
            u_err=u_err, v_err=v_err,
            valid=valid, n_valid=int(valid.sum()),
            rmse_u=float(np.sqrt(np.nanmean(u_err[valid] ** 2))),
            rmse_v=float(np.sqrt(np.nanmean(v_err[valid] ** 2))),
            rmse=float(np.sqrt(np.nanmean(u_err[valid] ** 2 + v_err[valid] ** 2))),
            max_err=float(np.nanmax(mag_err)) if len(mag_err) > 0 else 0.0,
            median_err=float(np.median(mag_err)) if len(mag_err) > 0 else 0.0,
            elapsed=elapsed,
        )
        print(f"    {mesh_info(m)}, RMSE={results[label]['rmse']:.4f}px, "
              f"time={elapsed:.1f}s")

    # ── Generate PDF ──────────────────────────────────────────────
    print("Generating report ...")
    labels = [c[0] for c in COMBOS]

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Input overview ────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Input: Complex 1024×1024 Mask + Quadratic Displacement + Noise (σ={NOISE_STD})",
            fontsize=13, y=0.98,
        )

        axes[0].imshow(mask, cmap="gray", origin="upper")
        axes[0].set_title("Mask (4 domains, 8 holes)", fontsize=10)

        im1 = axes[1].imshow(
            np.where(mask > 0.5, u_gt_field, np.nan),
            cmap="RdBu_r", origin="upper",
        )
        axes[1].set_title("GT u-field (px)", fontsize=10)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(
            np.where(mask > 0.5, v_gt_field, np.nan),
            cmap="RdBu_r", origin="upper",
        )
        axes[2].set_title("GT v-field (px)", fontsize=10)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.tick_params(labelsize=6)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: u-displacement (2×2) ─────────────────────────
        u_vmin = min(r["u_gt"][r["valid"]].min() for r in results.values())
        u_vmax = max(r["u_gt"][r["valid"]].max() for r in results.values())
        v_vmin = min(r["v_gt"][r["valid"]].min() for r in results.values())
        v_vmax = max(r["v_gt"][r["valid"]].max() for r in results.values())

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Estimated u-displacement (px)", fontsize=13, y=0.98)
        for idx, lbl in enumerate(labels):
            r = results[lbl]
            ax = axes[idx // 2, idx % 2]
            draw_field(ax, r["mesh"], r["u_est"], mask,
                       f"{lbl}\n{mesh_info(r['mesh'])}",
                       cmap="RdBu_r", vmin=u_vmin, vmax=u_vmax)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: v-displacement (2×2) ─────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Estimated v-displacement (px)", fontsize=13, y=0.98)
        for idx, lbl in enumerate(labels):
            r = results[lbl]
            ax = axes[idx // 2, idx % 2]
            draw_field(ax, r["mesh"], r["v_est"], mask,
                       f"{lbl}\n{mesh_info(r['mesh'])}",
                       cmap="RdBu_r", vmin=v_vmin, vmax=v_vmax)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: u-error (2×2) ────────────────────────────────
        all_ue = np.concatenate([r["u_err"][r["valid"]] for r in results.values()])
        ue_lim = float(np.nanpercentile(np.abs(all_ue), 99))

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("u-displacement Error (estimated − GT)", fontsize=13, y=0.98)
        for idx, lbl in enumerate(labels):
            r = results[lbl]
            ax = axes[idx // 2, idx % 2]
            draw_field(ax, r["mesh"], r["u_err"], mask,
                       f"{lbl}\nRMSE(u)={r['rmse_u']:.4f} px",
                       cmap="RdBu_r", vmin=-ue_lim, vmax=ue_lim)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 5: v-error (2×2) ────────────────────────────────
        all_ve = np.concatenate([r["v_err"][r["valid"]] for r in results.values()])
        ve_lim = float(np.nanpercentile(np.abs(all_ve), 99))

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("v-displacement Error (estimated − GT)", fontsize=13, y=0.98)
        for idx, lbl in enumerate(labels):
            r = results[lbl]
            ax = axes[idx // 2, idx % 2]
            draw_field(ax, r["mesh"], r["v_err"], mask,
                       f"{lbl}\nRMSE(v)={r['rmse_v']:.4f} px",
                       cmap="RdBu_r", vmin=-ve_lim, vmax=ve_lim)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 6: Error magnitude (2×2) ────────────────────────
        all_mag = np.concatenate([
            np.sqrt(r["u_err"][r["valid"]] ** 2 + r["v_err"][r["valid"]] ** 2)
            for r in results.values()
        ])
        mag_vmax = float(np.nanpercentile(all_mag, 99))

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Displacement Error Magnitude |e| (px)", fontsize=13, y=0.98)
        for idx, lbl in enumerate(labels):
            r = results[lbl]
            mag = np.full_like(r["u_err"], np.nan)
            mag[r["valid"]] = np.sqrt(
                r["u_err"][r["valid"]] ** 2 + r["v_err"][r["valid"]] ** 2
            )
            ax = axes[idx // 2, idx % 2]
            draw_field(ax, r["mesh"], mag, mask,
                       f"{lbl}\nRMSE={r['rmse']:.4f}, max={r['max_err']:.4f} px",
                       cmap="hot_r", vmin=0, vmax=mag_vmax)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 7: Histograms + summary table ───────────────────
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle("Error Distribution & Summary", fontsize=13, y=0.98)
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        colors = {"Uniform + Local": "C0", "Uniform + AL-DIC": "C1",
                  "Refined + Local": "C2", "Refined + AL-DIC": "C3"}

        # u-error histogram
        ax1 = fig.add_subplot(gs[0, 0])
        bins = np.linspace(-0.5, 0.5, 80)
        for lbl in labels:
            r = results[lbl]
            ax1.hist(r["u_err"][r["valid"]], bins=bins, alpha=0.5,
                     label=lbl, color=colors[lbl], density=True)
        ax1.set_xlabel("u error (px)", fontsize=8)
        ax1.set_ylabel("Density", fontsize=8)
        ax1.set_title("u-error distribution", fontsize=9)
        ax1.legend(fontsize=6)
        ax1.tick_params(labelsize=6)

        # v-error histogram
        ax2 = fig.add_subplot(gs[0, 1])
        for lbl in labels:
            r = results[lbl]
            ax2.hist(r["v_err"][r["valid"]], bins=bins, alpha=0.5,
                     label=lbl, color=colors[lbl], density=True)
        ax2.set_xlabel("v error (px)", fontsize=8)
        ax2.set_title("v-error distribution", fontsize=9)
        ax2.legend(fontsize=6)
        ax2.tick_params(labelsize=6)

        # Error magnitude histogram
        ax3 = fig.add_subplot(gs[0, 2])
        bins_mag = np.linspace(0, 0.6, 60)
        for lbl in labels:
            r = results[lbl]
            mag = np.sqrt(r["u_err"][r["valid"]] ** 2
                          + r["v_err"][r["valid"]] ** 2)
            ax3.hist(mag, bins=bins_mag, alpha=0.5,
                     label=lbl, color=colors[lbl], density=True)
        ax3.set_xlabel("|error| (px)", fontsize=8)
        ax3.set_title("Error magnitude distribution", fontsize=9)
        ax3.legend(fontsize=6)
        ax3.tick_params(labelsize=6)

        # Summary table
        ax_t = fig.add_subplot(gs[1, :])
        ax_t.axis("off")

        col_labels = [
            "Configuration", "Nodes", "Elements", "Valid",
            "RMSE u", "RMSE v", "RMSE total",
            "Max |e|", "Median |e|", "Time (s)",
        ]
        rows = []
        for lbl in labels:
            r = results[lbl]
            mi = mesh_info(r["mesh"])
            parts = mi.split(",")
            rows.append([
                lbl,
                parts[0].strip(),
                parts[1].strip(),
                str(r["n_valid"]),
                f"{r['rmse_u']:.4f}",
                f"{r['rmse_v']:.4f}",
                f"{r['rmse']:.4f}",
                f"{r['max_err']:.4f}",
                f"{r['median_err']:.4f}",
                f"{r['elapsed']:.1f}",
            ])

        # Improvement rows
        base = results["Uniform + Local"]
        for lbl in labels[1:]:
            r = results[lbl]
            pct = (1 - r["rmse"] / base["rmse"]) * 100
            rows.append([
                f"vs Local+Uni: {lbl}",
                "", "", "",
                "", "", f"{pct:+.1f}%",
                "", "", "",
            ])

        table = ax_t.table(
            cellText=rows, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.7)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(len(rows)):
            if i < 4:
                color = "#D6E4F0" if i % 2 == 0 else "white"
            else:
                color = "#FFF2CC"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")
    print("  7 pages: input + u-disp + v-disp + u-err + v-err + |err| + summary")


if __name__ == "__main__":
    main()
