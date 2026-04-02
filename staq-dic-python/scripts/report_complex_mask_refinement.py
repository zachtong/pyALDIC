#!/usr/bin/env python
"""Visual report: Complex 1024x1024 mask with multiple geometric shapes.

Tests the refinement system against a single highly complex mask featuring:
  - Multiple connected domains (main body + detached islands)
  - Multiple holes: circular, rectangular, triangular, star-shaped
  - Non-convex outer boundary (L-notch, semicircular bite)
  - A detached annular ring island
  - A detached triangle island

Refinement configs tested:
  1. Uniform (baseline)
  2. Inner boundary only
  3. Outer boundary only
  4. Inner + outer boundary
  5. Inner + outer + brush (diagonal strain band)

Output: reports/complex_mask_refinement_report.pdf
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

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.mesh.refinement import (
    RefinementContext,
    build_refinement_policy,
    refine_mesh,
)
from staq_dic.solver.integer_search import integer_search
from staq_dic.solver.init_disp import init_disp

IMG_H, IMG_W = 1024, 1024
STEP = 16
WINSIZE = 32
HALF_WIN = WINSIZE // 2
MIN_ELEM = 8


# ===================================================================
# Mask construction helpers
# ===================================================================

def _circle_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    """Boolean mask for a filled circle."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2


def _triangle_mask(
    h: int, w: int, v0: tuple, v1: tuple, v2: tuple,
) -> np.ndarray:
    """Boolean mask for a filled triangle (3 vertices, CCW or CW)."""
    yy, xx = np.mgrid[0:h, 0:w]
    # Barycentric coordinate method
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    denom = float((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
    if abs(denom) < 1e-10:
        return np.zeros((h, w), dtype=bool)
    a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
    b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
    c = 1.0 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)


def _star_mask(h: int, w: int, cx: int, cy: int, r_outer: int, r_inner: int,
               n_points: int = 5) -> np.ndarray:
    """Boolean mask for a star polygon (n_points tips)."""
    angles_outer = np.linspace(-np.pi / 2, 3 * np.pi / 2, n_points, endpoint=False)
    angles_inner = angles_outer + np.pi / n_points

    # Build star polygon vertices
    verts_x, verts_y = [], []
    for ao, ai in zip(angles_outer, angles_inner):
        verts_x.append(cx + r_outer * np.cos(ao))
        verts_y.append(cy + r_outer * np.sin(ao))
        verts_x.append(cx + r_inner * np.cos(ai))
        verts_y.append(cy + r_inner * np.sin(ai))

    # Point-in-polygon via matplotlib path
    from matplotlib.path import Path as MplPath
    poly = np.column_stack([verts_x, verts_y])
    poly = np.vstack([poly, poly[0]])  # close
    path = MplPath(poly)

    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = path.contains_points(pts).reshape(h, w)
    return inside


def _ellipse_mask(h: int, w: int, cx: int, cy: int,
                  rx: int, ry: int) -> np.ndarray:
    """Boolean mask for a filled axis-aligned ellipse."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0


def build_complex_mask() -> np.ndarray:
    """Build a 1024x1024 mask with multiple domains and holes.

    Layout (approximate):
      +-------------------------------------------------+
      |                                                 |
      |   [Main body: large non-convex region]          |
      |     - L-notch cut from top-right corner         |
      |     - Semicircular bite from bottom edge         |
      |     - Holes: 2 circles, 1 rectangle, 1 star,   |
      |              1 triangle, 1 ellipse               |
      |                                                 |
      |          [Detached island 1: annular ring]       |
      |                                                 |
      |          [Detached island 2: triangle]           |
      |                                                 |
      |          [Detached island 3: small rectangle]    |
      +-------------------------------------------------+
    """
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    h, w = IMG_H, IMG_W

    # ── Region 1: Main body (large, non-convex) ──────────────
    # Base: large rectangle
    mask[40:700, 40:750] = 1.0

    # Cut: L-shaped notch from top-right
    mask[40:250, 550:750] = 0.0

    # Cut: semicircular bite from bottom edge
    bite = _circle_mask(h, w, cx=400, cy=700, r=120)
    mask[bite] = 0.0

    # ── Holes in main body ────────────────────────────────────
    # Hole 1: large circle
    h1 = _circle_mask(h, w, cx=200, cy=300, r=70)
    mask[h1] = 0.0

    # Hole 2: small circle
    h2 = _circle_mask(h, w, cx=500, cy=450, r=40)
    mask[h2] = 0.0

    # Hole 3: rectangle
    mask[150:250, 350:470] = 0.0

    # Hole 4: star-shaped (5-point)
    h4 = _star_mask(h, w, cx=350, cy=550, r_outer=55, r_inner=22)
    mask[h4] = 0.0

    # Hole 5: triangle
    h5 = _triangle_mask(h, w, (130, 480), (220, 600), (80, 600))
    mask[h5] = 0.0

    # Hole 6: elliptical hole
    h6 = _ellipse_mask(h, w, cx=600, cy=500, rx=50, ry=25)
    mask[h6] = 0.0

    # ── Region 2: Detached annular ring (top-right) ──────────
    ring_outer = _circle_mask(h, w, cx=870, cy=180, r=120)
    ring_inner = _circle_mask(h, w, cx=870, cy=180, r=55)
    mask[ring_outer & ~ring_inner] = 1.0

    # ── Region 3: Detached triangle island (bottom-right) ────
    tri = _triangle_mask(h, w, (800, 580), (980, 750), (750, 750))
    mask[tri] = 1.0

    # ── Region 4: Small detached rectangle (bottom-left) ─────
    mask[800:930, 80:250] = 1.0

    # Punch a small circular hole in this rectangle
    h_small = _circle_mask(h, w, cx=165, cy=865, r=30)
    mask[h_small] = 0.0

    return mask


def build_brush_mask() -> np.ndarray:
    """Diagonal strain concentration band + two circular hotspots."""
    rmask = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]

    # Diagonal band across the main body
    dist = np.abs(yy - xx + 200) / np.sqrt(2)
    rmask[dist < 30] = 1.0

    # Circular hotspot near the annular ring
    rmask[_circle_mask(IMG_H, IMG_W, cx=870, cy=180, r=30)] = 1.0

    # Circular hotspot on the triangle island
    rmask[_circle_mask(IMG_H, IMG_W, cx=850, cy=680, r=35)] = 1.0

    return rmask


# ===================================================================
# Mesh helpers
# ===================================================================

def make_base_mesh(mask: np.ndarray):
    """Generate base uniform mesh + U0 for a 1024x1024 mask."""
    rng = np.random.default_rng(42)
    ref = gaussian_filter(rng.standard_normal((IMG_H, IMG_W)), sigma=5.0)
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


def get_elem_diags(mesh) -> np.ndarray:
    """Return per-element diagonal length array."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem
    valid = np.any(elems >= 0, axis=1)
    if not valid.any():
        return np.array([])
    corners = elems[valid, :4]
    dx = coords[corners[:, 2], 0] - coords[corners[:, 0], 0]
    dy = coords[corners[:, 2], 1] - coords[corners[:, 0], 1]
    return np.sqrt(dx ** 2 + dy ** 2)


# ===================================================================
# Drawing
# ===================================================================

def draw_mesh(ax, mesh, mask, title, brush_mask=None):
    """Draw mesh elements coloured by diagonal size on top of mask."""
    h, w = IMG_H, IMG_W
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    ax.imshow(mask, cmap="gray", alpha=0.25, origin="upper",
              extent=[0, w, h, 0])

    if brush_mask is not None:
        overlay = np.ma.masked_where(brush_mask < 0.5, brush_mask)
        ax.imshow(overlay, cmap="Oranges", alpha=0.25, origin="upper",
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
        norm = plt.Normalize(5, 25)
        colors = plt.cm.viridis(norm(sizes_arr))
        for poly, color in zip(polys, colors):
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color,
                    edgecolor="black", linewidth=0.15, alpha=0.6)

    st = mesh_stats(mesh)
    ax.set_title(f"{title}\n{st['n_nodes']} nodes, {st['n_elem']} elems",
                 fontsize=8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)


def draw_mesh_zoomed(ax, mesh, mask, region, title, brush_mask=None):
    """Draw a zoomed-in region of the mesh. region = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = region
    h, w = IMG_H, IMG_W
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    ax.imshow(mask, cmap="gray", alpha=0.25, origin="upper",
              extent=[0, w, h, 0])

    if brush_mask is not None:
        overlay = np.ma.masked_where(brush_mask < 0.5, brush_mask)
        ax.imshow(overlay, cmap="Oranges", alpha=0.25, origin="upper",
                  extent=[0, w, h, 0])

    polys, sizes = [], []
    for e in range(elems.shape[0]):
        c4 = elems[e, :4]
        if np.any(c4 < 0):
            continue
        verts = coords[c4]
        # Check if element is within or overlaps the region
        ex_min, ex_max = verts[:, 0].min(), verts[:, 0].max()
        ey_min, ey_max = verts[:, 1].min(), verts[:, 1].max()
        if ex_max < x0 or ex_min > x1 or ey_max < y0 or ey_min > y1:
            continue
        closed = np.vstack([verts, verts[0]])
        polys.append(closed)
        dx = coords[c4[2], 0] - coords[c4[0], 0]
        dy = coords[c4[2], 1] - coords[c4[0], 1]
        sizes.append(np.sqrt(dx ** 2 + dy ** 2))

    if polys:
        sizes_arr = np.array(sizes)
        norm = plt.Normalize(5, 25)
        colors = plt.cm.viridis(norm(sizes_arr))
        for poly, color in zip(polys, colors):
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color,
                    edgecolor="black", linewidth=0.3, alpha=0.7)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)


# ===================================================================
# Main report
# ===================================================================

def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "complex_mask_refinement_report.pdf"

    print("Building complex 1024x1024 mask ...")
    mask = build_complex_mask()
    brush = build_brush_mask()

    print("Generating base mesh ...")
    base_mesh, U0, para = make_base_mesh(mask)

    # 5 refinement configs
    configs = [
        ("Uniform (baseline)", None),
        ("Inner Boundary Only", build_refinement_policy(
            refine_inner_boundary=True, min_element_size=MIN_ELEM)),
        ("Outer Boundary Only", build_refinement_policy(
            refine_outer_boundary=True, min_element_size=MIN_ELEM,
            half_win=HALF_WIN)),
        ("Inner + Outer Boundary", build_refinement_policy(
            refine_inner_boundary=True, refine_outer_boundary=True,
            min_element_size=MIN_ELEM, half_win=HALF_WIN)),
        ("All Combined\n(Inner+Outer+Brush)", build_refinement_policy(
            refine_inner_boundary=True, refine_outer_boundary=True,
            refinement_mask=brush, min_element_size=MIN_ELEM,
            half_win=HALF_WIN)),
    ]

    print("Running refinement configs ...")
    meshes = []
    for label, policy in configs:
        clean = label.replace("\n", " ")
        print(f"  {clean} ...")
        if policy is None:
            meshes.append(base_mesh)
        else:
            meshes.append(refine_with_policy(base_mesh, policy, mask, U0))

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Mask + brush overview ─────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle(
            "Complex 1024x1024 Mask — Input Overview",
            fontsize=14, y=0.98,
        )

        axes[0].imshow(mask, cmap="gray", origin="upper")
        axes[0].set_title("Mask (4 domains, 7 holes)", fontsize=10)

        axes[1].imshow(brush, cmap="Oranges", origin="upper")
        axes[1].set_title("Brush Pattern\n(diagonal + 2 hotspots)", fontsize=10)

        # Composite
        composite = np.zeros((IMG_H, IMG_W, 3))
        composite[:, :, 1] = mask * 0.4        # green = valid region
        composite[:, :, 0] = brush * 0.6        # red = brush
        composite[:, :, 2] = (1 - mask) * 0.15  # faint blue = background
        composite = np.clip(composite, 0, 1)
        axes[2].imshow(composite, origin="upper")
        axes[2].set_title("Composite\n(green=mask, red=brush)", fontsize=10)

        for ax in axes:
            ax.tick_params(labelsize=6)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Full mesh comparison (3 configs) ──────────────
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle(
            "Refinement Comparison — Full View (1 of 2)",
            fontsize=14, y=0.98,
        )
        for idx in range(3):
            label, _ = configs[idx]
            draw_mesh(axes[idx], meshes[idx], mask, label)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Full mesh comparison (2 configs) ──────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            "Refinement Comparison — Full View (2 of 2)",
            fontsize=14, y=0.98,
        )
        for idx_ax, idx_cfg in enumerate([3, 4]):
            label, _ = configs[idx_cfg]
            show_brush = "Brush" in label
            draw_mesh(axes[idx_ax], meshes[idx_cfg], mask, label,
                      brush_mask=brush if show_brush else None)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Pages 4-5: Zoomed-in regions on "All Combined" mesh ──
        refined_all = meshes[4]

        zoom_regions = [
            # (x0, y0, x1, y1), title
            ((100, 200, 380, 480), "Zoom: Large circular hole\n(inner boundary)"),
            ((250, 100, 530, 300), "Zoom: Rectangular hole\n+ star hole"),
            ((400, 350, 700, 650), "Zoom: Small circle + ellipse\n+ semicircular bite"),
            ((720, 50, 1010, 350), "Zoom: Annular ring island\n(inner+outer boundary)"),
            ((650, 500, 1010, 800), "Zoom: Triangle island\n(outer boundary)"),
            ((20, 750, 300, 960), "Zoom: Small rectangle island\n+ circular hole"),
        ]

        # Page 4: first 3 zoom regions
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            "Zoomed Regions — All Combined (1 of 2)", fontsize=14, y=0.98,
        )
        for idx in range(3):
            region, title = zoom_regions[idx]
            draw_mesh_zoomed(axes[idx], refined_all, mask, region, title,
                             brush_mask=brush)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 5: last 3 zoom regions
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            "Zoomed Regions — All Combined (2 of 2)", fontsize=14, y=0.98,
        )
        for idx in range(3, 6):
            region, title = zoom_regions[idx]
            draw_mesh_zoomed(axes[idx - 3], refined_all, mask, region, title,
                             brush_mask=brush)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 6: Element size distribution ─────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Element Size Distributions", fontsize=14, y=0.98)

        for idx, (label, _) in enumerate(configs):
            ax = axes[idx // 3, idx % 3]
            diags = get_elem_diags(meshes[idx])
            clean = label.replace("\n", " ")
            st = mesh_stats(meshes[idx])
            if len(diags) > 0:
                n_unique = len(np.unique(np.round(diags, 1)))
                ax.hist(diags, bins=max(n_unique * 3, 15),
                        color="steelblue", edgecolor="black", alpha=0.7)
            ax.set_xlabel("Element diagonal (px)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(
                f"{clean}\n{st['n_elem']} elems, "
                f"diag: [{st['min_d']:.1f}, {st['max_d']:.1f}]",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)

        # Hide unused subplot
        axes[1, 2].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 7: Summary table ─────────────────────────────────
        col_labels = [
            "Configuration", "Nodes", "Elements",
            "Min Diag", "Max Diag", "Mean Diag", "Unique Sizes",
        ]
        rows = []
        for (label, _), m in zip(configs, meshes):
            st = mesh_stats(m)
            rows.append([
                label.replace("\n", " "),
                str(st["n_nodes"]),
                str(st["n_elem"]),
                f"{st['min_d']:.1f}",
                f"{st['max_d']:.1f}",
                f"{st['mean_d']:.1f}",
                str(st["n_sizes"]),
            ])

        fig, ax = plt.subplots(figsize=(14, 3))
        fig.suptitle("Summary Table", fontsize=14, y=0.95)
        ax.axis("off")

        table = ax.table(
            cellText=rows, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(len(rows)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        fig.tight_layout(rect=[0, 0, 1, 0.88])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")
    print(f"  Pages: 7 (overview + 2 full + 2 zoom + distribution + summary)")


if __name__ == "__main__":
    main()
