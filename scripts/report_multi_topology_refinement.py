#!/usr/bin/env python
"""Visual report: Adaptive refinement across multiple mask topologies.

Tests the full refinement system (inner boundary, outer boundary, brush
region) against six mask topologies and three brush patterns:

Masks:
  1. Simple square (no holes)
  2. Square + one circular hole
  3. Square + three holes (different sizes/shapes)
  4. L-shaped domain
  5. Annular ring
  6. Two separate rectangles (multi-region)

Brush patterns:
  A. Diagonal band (strain concentration)
  B. Cross / plus pattern (intersection region)
  C. Scattered circular spots (multi-point refinement)

For each topology: Uniform → Inner+Outer boundary → All combined (boundary+brush)

Output: reports/multi_topology_refinement_report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.mesh.refinement import (
    RefinementContext,
    build_refinement_policy,
    refine_mesh,
)
from al_dic.solver.integer_search import integer_search
from al_dic.solver.init_disp import init_disp

IMG_H, IMG_W = 256, 256
STEP = 16
WINSIZE = 32
HALF_WIN = WINSIZE // 2
MIN_ELEM = 8


# ═══════════════════════════════════════════════════════════════════════
# Mask generators
# ═══════════════════════════════════════════════════════════════════════


def _base_square(margin: int = 20) -> np.ndarray:
    """Solid square mask with given margin."""
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    mask[margin : IMG_H - margin, margin : IMG_W - margin] = 1.0
    return mask


def _punch_circle(mask: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """Punch a circular hole into a mask (returns new array)."""
    out = mask.copy()
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    out[np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < r] = 0.0
    return out


def _punch_rect(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Punch a rectangular hole into a mask (returns new array)."""
    out = mask.copy()
    out[y0:y1, x0:x1] = 0.0
    return out


def mask_simple_square() -> tuple[np.ndarray, str]:
    """Single connected domain, no holes."""
    return _base_square(margin=20), "Simple Square\n(no holes)"


def mask_one_hole() -> tuple[np.ndarray, str]:
    """Single connected domain, one circular hole."""
    m = _base_square(margin=20)
    m = _punch_circle(m, 128, 128, 40)
    return m, "Square + 1 Hole\n(circular, r=40)"


def mask_three_holes() -> tuple[np.ndarray, str]:
    """Single connected domain, three holes of different shapes/sizes."""
    m = _base_square(margin=15)
    m = _punch_circle(m, 80, 80, 22)       # small circle top-left
    m = _punch_circle(m, 180, 100, 30)      # medium circle right
    m = _punch_rect(m, 60, 160, 120, 200)   # rectangle bottom-left
    return m, "Square + 3 Holes\n(2 circles + 1 rect)"


def mask_l_shaped() -> tuple[np.ndarray, str]:
    """L-shaped domain (square minus top-right quadrant)."""
    m = _base_square(margin=15)
    m[15:125, 135:241] = 0.0  # cut top-right quadrant
    return m, "L-Shaped Domain\n(cut top-right)"


def mask_annular() -> tuple[np.ndarray, str]:
    """Annular ring (large central hole)."""
    m = _base_square(margin=15)
    m = _punch_circle(m, 128, 128, 65)
    return m, "Annular Ring\n(inner r=65)"


def mask_multi_region() -> tuple[np.ndarray, str]:
    """Two separate rectangular regions (multi-connected)."""
    m = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    m[20:110, 20:236] = 1.0    # top rectangle
    m[146:236, 20:236] = 1.0   # bottom rectangle
    return m, "Two Rectangles\n(multi-region)"


ALL_MASKS = [
    mask_simple_square,
    mask_one_hole,
    mask_three_holes,
    mask_l_shaped,
    mask_annular,
    mask_multi_region,
]


# ═══════════════════════════════════════════════════════════════════════
# Brush pattern generators
# ═══════════════════════════════════════════════════════════════════════


def brush_diagonal_band() -> tuple[np.ndarray, str]:
    """Diagonal band (strain concentration along 45-degree line)."""
    rmask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    dist = np.abs(yy - xx) / np.sqrt(2)
    rmask[dist < 15] = 1.0
    return rmask, "Diagonal Band"


def brush_cross() -> tuple[np.ndarray, str]:
    """Cross / plus pattern (vertical + horizontal bands)."""
    rmask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    cx, cy = IMG_W // 2, IMG_H // 2
    half_w = 12
    rmask[cy - half_w : cy + half_w, 30:226] = 1.0   # horizontal
    rmask[30:226, cx - half_w : cx + half_w] = 1.0    # vertical
    return rmask, "Cross Pattern"


def brush_scattered_spots() -> tuple[np.ndarray, str]:
    """Multiple circular spots at different locations."""
    rmask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    spots = [(70, 70, 20), (190, 70, 15), (128, 180, 25), (60, 200, 12), (200, 190, 18)]
    for sx, sy, sr in spots:
        rmask[np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) < sr] = 1.0
    return rmask, "Scattered Spots (5)"


ALL_BRUSHES = [
    brush_diagonal_band,
    brush_cross,
    brush_scattered_spots,
]


# ═══════════════════════════════════════════════════════════════════════
# Mesh helpers
# ═══════════════════════════════════════════════════════════════════════


def make_base_mesh(mask: np.ndarray):
    """Generate base uniform mesh + U0 for a given mask."""
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
    g_img = f_img.copy()
    x0, y0, u_grid, v_grid, fft_info = integer_search(f_img, g_img, para)
    base_mesh = mesh_setup(x0, y0, para)
    U0 = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
    return base_mesh, U0, para


def refine_with_policy(base_mesh, policy, mask, U0):
    """Apply a refinement policy and return the refined mesh."""
    ctx = RefinementContext(mesh=base_mesh, mask=mask)
    refined, _ = refine_mesh(
        base_mesh,
        policy.pre_solve,
        ctx,
        U0,
        mask=mask,
        img_size=(IMG_H, IMG_W),
    )
    return refined


def mesh_stats(mesh) -> dict:
    """Compute element count, node count, size stats."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem
    valid = np.any(elems >= 0, axis=1)
    n_elem = int(valid.sum())
    n_nodes = coords.shape[0]
    if n_elem == 0:
        return dict(n_nodes=n_nodes, n_elem=0, min_d=0, max_d=0, mean_d=0, n_sizes=0)
    corners = elems[valid, :4]
    dx = coords[corners[:, 2], 0] - coords[corners[:, 0], 0]
    dy = coords[corners[:, 2], 1] - coords[corners[:, 0], 1]
    diag = np.sqrt(dx ** 2 + dy ** 2)
    return dict(
        n_nodes=n_nodes, n_elem=n_elem,
        min_d=float(diag.min()), max_d=float(diag.max()),
        mean_d=float(diag.mean()),
        n_sizes=len(np.unique(np.round(diag, 1))),
    )


# ═══════════════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════════════


def draw_mesh(ax, mesh, mask, title, brush_mask=None, img_size=(IMG_H, IMG_W)):
    """Draw mesh elements coloured by size on top of mask."""
    h, w = img_size
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    ax.imshow(mask, cmap="gray", alpha=0.3, origin="upper", extent=[0, w, h, 0])

    if brush_mask is not None:
        overlay = np.ma.masked_where(brush_mask < 0.5, brush_mask)
        ax.imshow(overlay, cmap="Oranges", alpha=0.3, origin="upper",
                  extent=[0, w, h, 0])

    polys, sizes = [], []
    for e in range(elems.shape[0]):
        c4 = elems[e, :4]
        if np.any(c4 < 0):
            continue
        verts = np.vstack([coords[c4], coords[c4[0]]])
        polys.append(verts)
        dx = coords[c4[2], 0] - coords[c4[0], 0]
        dy = coords[c4[2], 1] - coords[c4[0], 1]
        sizes.append(np.sqrt(dx ** 2 + dy ** 2))

    if polys:
        sizes_arr = np.array(sizes)
        # Use a fixed global range so colours are comparable across panels
        norm = plt.Normalize(5, 25)
        colors = plt.cm.viridis(norm(sizes_arr))
        for poly, color in zip(polys, colors):
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color,
                    edgecolor="black", linewidth=0.25, alpha=0.6)

    st = mesh_stats(mesh)
    ax.set_title(f"{title}\n{st['n_nodes']} nodes, {st['n_elem']} elems", fontsize=8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "multi_topology_refinement_report.pdf"

    # Pre-generate all masks and brushes
    masks = [fn() for fn in ALL_MASKS]
    brushes = [fn() for fn in ALL_BRUSHES]

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Mask + brush overview ───────────────────────────
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle("Input Overview — 6 Mask Topologies + 3 Brush Patterns",
                      fontsize=13, y=0.98)

        for idx, (m, label) in enumerate(masks):
            ax = axes[idx // 3, idx % 3]
            ax.imshow(m, cmap="gray", origin="upper")
            ax.set_title(label, fontsize=9)
            ax.tick_params(labelsize=6)

        for idx, (b, label) in enumerate(brushes):
            ax = axes[2, idx]
            ax.imshow(b, cmap="Oranges", origin="upper")
            ax.set_title(f"Brush: {label}", fontsize=9)
            ax.tick_params(labelsize=6)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Pages 2–7: Per-topology refinement comparison ───────────
        # Use cross brush for the per-topology pages (most informative)
        cross_brush, cross_label = brushes[1]

        all_summary_rows = []

        for mask_arr, mask_label in masks:
            print(f"  Processing: {mask_label.replace(chr(10), ' ')} ...")

            base_mesh, U0, para = make_base_mesh(mask_arr)

            # 4 configs per topology
            configs = [
                ("Uniform", None),
                ("Inner Boundary", build_refinement_policy(
                    refine_inner_boundary=True, min_element_size=MIN_ELEM)),
                ("Outer Boundary", build_refinement_policy(
                    refine_outer_boundary=True, min_element_size=MIN_ELEM,
                    half_win=HALF_WIN)),
                ("Inner + Outer\n+ Brush", build_refinement_policy(
                    refine_inner_boundary=True, refine_outer_boundary=True,
                    refinement_mask=cross_brush, min_element_size=MIN_ELEM,
                    half_win=HALF_WIN)),
            ]

            meshes = []
            for cfg_label, policy in configs:
                if policy is None:
                    meshes.append(base_mesh)
                else:
                    meshes.append(refine_with_policy(base_mesh, policy, mask_arr, U0))

            fig, axes = plt.subplots(2, 2, figsize=(11, 11))
            clean_name = mask_label.replace("\n", " ")
            fig.suptitle(f"Topology: {clean_name}", fontsize=13, y=0.98)

            for idx, ((cfg_label, _), m) in enumerate(zip(configs, meshes)):
                ax = axes[idx // 2, idx % 2]
                show_brush = "Brush" in cfg_label
                draw_mesh(ax, m, mask_arr, cfg_label,
                          brush_mask=cross_brush if show_brush else None)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # Collect summary
            for (cfg_label, _), m in zip(configs, meshes):
                st = mesh_stats(m)
                all_summary_rows.append([
                    clean_name,
                    cfg_label.replace("\n", " "),
                    str(st["n_nodes"]),
                    str(st["n_elem"]),
                    f"{st['min_d']:.1f}",
                    f"{st['max_d']:.1f}",
                    f"{st['mean_d']:.1f}",
                    str(st["n_sizes"]),
                ])

        # ── Page 8: Brush pattern comparison on one-hole mask ───────
        one_hole_mask, _ = masks[1]
        base_mesh_oh, U0_oh, _ = make_base_mesh(one_hole_mask)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle("Brush Pattern Comparison — Square + 1 Hole",
                      fontsize=13, y=0.98)

        for idx, (brush_arr, brush_label) in enumerate(brushes):
            # Top row: brush pattern overlay on mask
            ax_top = axes[0, idx]
            combined = np.stack([one_hole_mask * 0.3] * 3, axis=-1)
            combined[:, :, 0] += brush_arr * 0.5
            combined[:, :, 1] += brush_arr * 0.2
            combined = np.clip(combined, 0, 1)
            ax_top.imshow(combined, origin="upper")
            ax_top.set_title(f"Brush: {brush_label}", fontsize=9)
            ax_top.tick_params(labelsize=6)

            # Bottom row: refined mesh
            ax_bot = axes[1, idx]
            policy = build_refinement_policy(
                refine_inner_boundary=True, refine_outer_boundary=True,
                refinement_mask=brush_arr, min_element_size=MIN_ELEM,
                half_win=HALF_WIN,
            )
            refined = refine_with_policy(base_mesh_oh, policy, one_hole_mask, U0_oh)
            draw_mesh(ax_bot, refined, one_hole_mask,
                      f"All Combined\n({brush_label})", brush_mask=brush_arr)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 9: Element size distributions for all topologies ───
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(
            "Element Size Distributions — All Combined (inner+outer+brush)",
            fontsize=13, y=0.98,
        )

        for idx, (mask_arr, mask_label) in enumerate(masks):
            ax = axes[idx // 3, idx % 3]
            base_mesh_t, U0_t, _ = make_base_mesh(mask_arr)
            policy = build_refinement_policy(
                refine_inner_boundary=True, refine_outer_boundary=True,
                refinement_mask=cross_brush, min_element_size=MIN_ELEM,
                half_win=HALF_WIN,
            )
            refined = refine_with_policy(base_mesh_t, policy, mask_arr, U0_t)

            st = mesh_stats(refined)
            coords = refined.coordinates_fem
            elems = refined.elements_fem
            valid = np.any(elems >= 0, axis=1)
            if valid.any():
                corners = elems[valid, :4]
                dx = coords[corners[:, 2], 0] - coords[corners[:, 0], 0]
                dy = coords[corners[:, 2], 1] - coords[corners[:, 0], 1]
                diag = np.sqrt(dx ** 2 + dy ** 2)
                n_unique = len(np.unique(np.round(diag, 1)))
                ax.hist(diag, bins=max(n_unique * 2, 10),
                        color="steelblue", edgecolor="black", alpha=0.7)

            clean = mask_label.replace("\n", " ")
            ax.set_xlabel("Element diagonal (px)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(f"{clean}\n{st['n_elem']} elems, {st.get('n_sizes', 0)} sizes",
                         fontsize=8)
            ax.tick_params(labelsize=6)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 10: Summary table ──────────────────────────────────
        # Split into two half-page tables for readability
        col_labels = [
            "Topology", "Configuration", "Nodes", "Elements",
            "Min", "Max", "Mean", "Sizes",
        ]

        n_rows = len(all_summary_rows)
        half = n_rows // 2

        for part, (start, end) in enumerate([(0, half), (half, n_rows)]):
            fig, ax = plt.subplots(figsize=(12, 0.3 * (end - start + 2) + 1))
            fig.suptitle(
                f"Summary Table (part {part + 1}/2)", fontsize=13, y=0.98,
            )
            ax.axis("off")

            rows = all_summary_rows[start:end]
            table = ax.table(
                cellText=rows, colLabels=col_labels,
                cellLoc="center", loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.5)

            for j in range(len(col_labels)):
                cell = table[0, j]
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")
            for i in range(len(rows)):
                color = "#D6E4F0" if i % 2 == 0 else "white"
                for j in range(len(col_labels)):
                    table[i + 1, j].set_facecolor(color)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"Report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
