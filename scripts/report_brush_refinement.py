#!/usr/bin/env python
"""Visual report: BrushRegionCriterion + build_refinement_policy.

Generates a 4-page PDF comparing four refinement configurations:
  1. Uniform (no refinement)
  2. Boundary-only (MaskBoundaryCriterion)
  3. Brush-only (BrushRegionCriterion)
  4. Boundary + Brush combined

Uses a 256x256 square mask with a central circular hole.
Brush region is a diagonal band simulating user-painted strain concentration.

Output: reports/brush_refinement_report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.mesh.refinement import (
    RefinementContext,
    build_refinement_policy,
    refine_mesh,
)
from al_dic.mesh.criteria import MaskBoundaryCriterion, BrushRegionCriterion
from al_dic.solver.integer_search import integer_search
from al_dic.solver.init_disp import init_disp

IMG_H, IMG_W = 256, 256
STEP = 16
WINSIZE = 32
HOLE_CENTER = (128, 128)
HOLE_RADIUS = 40
BRUSH_WIDTH = 30  # pixels


def make_mask():
    """Square mask with a central circular hole."""
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    margin = 20
    mask[margin : IMG_H - margin, margin : IMG_W - margin] = 1.0
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    dist = np.sqrt((xx - HOLE_CENTER[0]) ** 2 + (yy - HOLE_CENTER[1]) ** 2)
    mask[dist < HOLE_RADIUS] = 0.0
    return mask


def make_brush_mask():
    """Diagonal band brush stroke (simulating strain concentration)."""
    rmask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    # Diagonal band: |y - x| < BRUSH_WIDTH / sqrt(2)
    dist_to_diag = np.abs(yy - xx) / np.sqrt(2)
    rmask[dist_to_diag < BRUSH_WIDTH / 2] = 1.0
    return rmask


def make_base_mesh(mask):
    """Generate base uniform mesh + U0."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(42)
    ref = gaussian_filter(rng.standard_normal((IMG_H, IMG_W)), sigma=3.0)
    ref = 20.0 + 215.0 * (ref - ref.min()) / (ref.max() - ref.min())

    roi = GridxyROIRange(
        gridx=(STEP, IMG_W - 1 - STEP), gridy=(STEP, IMG_H - 1 - STEP)
    )
    para = dicpara_default(
        winsize=WINSIZE,
        winstepsize=STEP,
        winsize_min=STEP,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=roi,
        show_plots=False,
    )

    f_img = ref * mask
    g_img = f_img.copy()  # zero displacement
    x0, y0, u_grid, v_grid, fft_info = integer_search(f_img, g_img, para)
    base_mesh = mesh_setup(x0, y0, para)
    U0 = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    return base_mesh, U0, para


def draw_mesh(ax, mesh, mask, title, brush_mask=None):
    """Draw mesh elements on top of mask."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    # Show mask as background
    ax.imshow(mask, cmap="gray", alpha=0.3, origin="upper", extent=[0, IMG_W, IMG_H, 0])

    # Show brush mask overlay if provided
    if brush_mask is not None:
        brush_overlay = np.ma.masked_where(brush_mask < 0.5, brush_mask)
        ax.imshow(
            brush_overlay,
            cmap="Oranges",
            alpha=0.3,
            origin="upper",
            extent=[0, IMG_W, IMG_H, 0],
        )

    # Draw elements as polygons
    polys = []
    sizes = []
    for e in range(elems.shape[0]):
        corners = elems[e, :4]
        if np.any(corners < 0):
            continue
        verts = coords[corners]
        # Close the polygon
        verts = np.vstack([verts, verts[0]])
        polys.append(verts)
        dx = coords[corners[2], 0] - coords[corners[0], 0]
        dy = coords[corners[2], 1] - coords[corners[0], 1]
        sizes.append(np.sqrt(dx**2 + dy**2))

    if polys:
        sizes_arr = np.array(sizes)
        # Color by element size
        norm = plt.Normalize(sizes_arr.min(), sizes_arr.max())
        colors = plt.cm.viridis(norm(sizes_arr))

        for poly, color in zip(polys, colors):
            ax.fill(
                poly[:, 0],
                poly[:, 1],
                facecolor=color,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.6,
            )

    n_elem = int(np.sum(np.any(elems >= 0, axis=1)))
    n_nodes = coords.shape[0]
    ax.set_title(f"{title}\n{n_nodes} nodes, {n_elem} elements", fontsize=10)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "brush_refinement_report.pdf"

    mask = make_mask()
    brush_mask = make_brush_mask()
    base_mesh, U0, para = make_base_mesh(mask)

    half_win = WINSIZE // 2

    # Build 6 configurations
    configs = {
        "Uniform\n(no refinement)": None,
        "Inner Boundary\n(holes only)": build_refinement_policy(
            refine_inner_boundary=True, min_element_size=8,
        ),
        "Outer Boundary\n(ROI edge)": build_refinement_policy(
            refine_outer_boundary=True, min_element_size=8,
            half_win=half_win,
        ),
        "Inner + Outer\n(all boundaries)": build_refinement_policy(
            refine_inner_boundary=True, refine_outer_boundary=True,
            min_element_size=8, half_win=half_win,
        ),
        "Brush Only\n(diagonal band)": build_refinement_policy(
            refinement_mask=brush_mask, min_element_size=8,
        ),
        "All Combined\n(inner+outer+brush)": build_refinement_policy(
            refine_inner_boundary=True, refine_outer_boundary=True,
            refinement_mask=brush_mask, min_element_size=8,
            half_win=half_win,
        ),
    }

    # Generate meshes
    meshes = {}
    for label, policy in configs.items():
        if policy is None:
            meshes[label] = base_mesh
        else:
            ctx = RefinementContext(mesh=base_mesh, mask=mask)
            refined, _ = refine_mesh(
                base_mesh,
                policy.pre_solve,
                ctx,
                U0,
                mask=mask,
                img_size=(IMG_H, IMG_W),
            )
            meshes[label] = refined

    with PdfPages(str(pdf_path)) as pdf:
        # ── Page 1: Input overview ────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle("Brush Region Refinement — Input Overview", fontsize=14, y=0.98)

        axes[0].imshow(mask, cmap="gray", origin="upper")
        axes[0].set_title("ROI Mask\n(square + circular hole)", fontsize=10)
        axes[0].tick_params(labelsize=7)

        axes[1].imshow(brush_mask, cmap="Oranges", origin="upper")
        axes[1].set_title(
            f"Brush Refinement Mask\n(diagonal band, width={BRUSH_WIDTH}px)",
            fontsize=10,
        )
        axes[1].tick_params(labelsize=7)

        # Combined overlay
        combined = np.stack([mask * 0.3, mask * 0.3, mask * 0.3], axis=-1)
        combined[:, :, 0] += brush_mask * 0.5
        combined[:, :, 1] += brush_mask * 0.2
        combined = np.clip(combined, 0, 1)
        axes[2].imshow(combined, origin="upper")
        axes[2].set_title("Combined\n(mask + brush overlay)", fontsize=10)
        axes[2].tick_params(labelsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Mesh comparison (3x2) ─────────────────────────
        fig, axes = plt.subplots(3, 2, figsize=(12, 16))
        fig.suptitle("Mesh Comparison — Six Configurations", fontsize=14, y=0.99)

        labels = list(configs.keys())
        for idx, label in enumerate(labels):
            ax = axes[idx // 2, idx % 2]
            show_brush = "Brush" in label or "combined" in label.lower() or "All" in label
            draw_mesh(
                ax,
                meshes[label],
                mask,
                label,
                brush_mask=brush_mask if show_brush else None,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Element size distributions ────────────────────
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(
            "Element Size Distributions", fontsize=14, y=0.99
        )

        for idx, label in enumerate(labels):
            ax = axes[idx // 2, idx % 2]
            m = meshes[label]
            elems = m.elements_fem
            coords = m.coordinates_fem

            valid = np.any(elems >= 0, axis=1)
            corners = elems[valid, :4]
            dx = coords[corners[:, 2], 0] - coords[corners[:, 0], 0]
            dy = coords[corners[:, 2], 1] - coords[corners[:, 0], 1]
            diag = np.sqrt(dx**2 + dy**2)

            unique_sizes = np.unique(np.round(diag, 1))
            ax.hist(
                diag,
                bins=max(len(unique_sizes) * 2, 10),
                color="steelblue",
                edgecolor="black",
                alpha=0.7,
            )
            ax.set_xlabel("Element diagonal (px)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title(
                f"{label}\n{len(unique_sizes)} unique sizes, "
                f"{int(valid.sum())} elements",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Summary table ────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle("Summary Table", fontsize=14, y=0.95)
        ax.axis("off")

        rows = []
        for label in labels:
            m = meshes[label]
            n_nodes = m.coordinates_fem.shape[0]
            valid = np.any(m.elements_fem >= 0, axis=1)
            n_elem = int(valid.sum())
            corners = m.elements_fem[valid, :4]
            dx = m.coordinates_fem[corners[:, 2], 0] - m.coordinates_fem[corners[:, 0], 0]
            dy = m.coordinates_fem[corners[:, 2], 1] - m.coordinates_fem[corners[:, 0], 1]
            diag = np.sqrt(dx**2 + dy**2)
            rows.append([
                label.replace("\n", " "),
                str(n_nodes),
                str(n_elem),
                f"{diag.min():.1f}",
                f"{diag.max():.1f}",
                f"{diag.mean():.1f}",
                str(len(np.unique(np.round(diag, 1)))),
            ])

        col_labels = [
            "Configuration",
            "Nodes",
            "Elements",
            "Min Size",
            "Max Size",
            "Mean Size",
            "Unique Sizes",
        ]
        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        # Header style
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")

        # Alternating row colors
        for i in range(len(rows)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"Report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
