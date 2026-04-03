#!/usr/bin/env python
"""Visual report: mask warping from reference to deformed configuration.

Tests warp_mask with progressively complex scenarios:
  1. Uniform translation (simplest case)
  2. Uniform stretch (area change)
  3. Rotation (rigid body)
  4. Quadratic deformation (non-uniform strain)
  5. Complex mask with holes + large quadratic deformation
  6. Multi-domain mask + shear + stretch combo

For each case: original mask, displacement quiver, warped mask, overlay.

Output: reports/warp_mask_validation.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as ndimage_label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from staq_dic.utils.warp_mask import warp_mask


def topology(mask):
    """Return (n_domains, n_holes) for a binary mask."""
    h, w = mask.shape
    _, n_domains = ndimage_label(mask > 0.5)
    labeled_z, n_zero = ndimage_label(mask < 0.5)
    border = set()
    border.update(labeled_z[0, :].tolist())
    border.update(labeled_z[h - 1, :].tolist())
    border.update(labeled_z[:, 0].tolist())
    border.update(labeled_z[:, w - 1].tolist())
    border.discard(0)
    return n_domains, n_zero - len(border)


def det_F(u, v):
    """Compute det(I + grad(u)) — correct area scaling factor."""
    du_dx = np.gradient(u, axis=1)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    dv_dy = np.gradient(v, axis=0)
    return (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx

# ═══════════════════════════════════════════════════════════════════
# Mask builders
# ═══════════════════════════════════════════════════════════════════

def make_square_mask(h: int, w: int) -> np.ndarray:
    """Simple centered square."""
    mask = np.zeros((h, w), dtype=np.float64)
    margin = h // 8
    mask[margin:h - margin, margin:w - margin] = 1.0
    return mask


def make_annular_mask(h: int, w: int) -> np.ndarray:
    """Annular ring (circle with center hole)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = ((r < min(h, w) * 0.4) & (r > min(h, w) * 0.12)).astype(np.float64)
    return mask


def make_multi_hole_mask(h: int, w: int) -> np.ndarray:
    """Rectangle with 3 circular holes of different sizes."""
    mask = np.zeros((h, w), dtype=np.float64)
    mask[h // 8 : 7 * h // 8, w // 8 : 7 * w // 8] = 1.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    holes = [
        (w * 0.3, h * 0.35, min(h, w) * 0.08),
        (w * 0.6, h * 0.5, min(h, w) * 0.10),
        (w * 0.4, h * 0.7, min(h, w) * 0.06),
    ]
    for hx, hy, hr in holes:
        dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        mask[dist < hr] = 0.0

    return mask


def make_multi_domain_mask(h: int, w: int) -> np.ndarray:
    """Two separate connected domains + one with a hole."""
    mask = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Domain 1: large rectangle (left half) with elliptical hole
    mask[h // 6 : 5 * h // 6, w // 10 : 4 * w // 10] = 1.0
    ex, ey = w * 0.25, h * 0.5
    ellipse = ((xx - ex) / (w * 0.06)) ** 2 + ((yy - ey) / (h * 0.10)) ** 2
    mask[ellipse < 1.0] = 0.0

    # Domain 2: circle (right half)
    cx, cy, cr = w * 0.7, h * 0.4, min(h, w) * 0.18
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask[dist < cr] = 1.0

    # Domain 3: small triangle (bottom right)
    tri_cx, tri_cy = w * 0.75, h * 0.8
    tri_r = min(h, w) * 0.08
    in_tri = (
        (yy > tri_cy - tri_r)
        & (yy < tri_cy + tri_r)
        & (xx > tri_cx - tri_r)
        & (xx < tri_cx + (tri_cy + tri_r - yy))
        & (xx > tri_cx - (tri_cy + tri_r - yy))
    )
    mask[in_tri] = 1.0

    return mask


# ═══════════════════════════════════════════════════════════════════
# Displacement field builders
# ═══════════════════════════════════════════════════════════════════

def make_translation(h: int, w: int, tx: float, ty: float):
    u = np.full((h, w), tx, dtype=np.float64)
    v = np.full((h, w), ty, dtype=np.float64)
    return u, v


def make_stretch(h: int, w: int, strain_x: float, strain_y: float):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    u = strain_x * (xx - cx)
    v = strain_y * (yy - cy)
    return u, v


def make_rotation(h: int, w: int, angle_deg: float):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    dx, dy = xx - cx, yy - cy
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u = (cos_t - 1) * dx - sin_t * dy
    v = sin_t * dx + (cos_t - 1) * dy
    return u, v


def make_quadratic(h: int, w: int, amplitude: float):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    dx = (xx - cx) / cx  # normalized [-1, 1]
    dy = (yy - cy) / cy
    u = amplitude * (1.5 * dx**2 + 0.8 * dy**2 + 0.5 * dx * dy
                     + 0.3 * dx + 0.1 * dy)
    v = amplitude * (0.6 * dx**2 + 1.2 * dy**2 - 0.4 * dx * dy
                     + 0.1 * dx + 0.2 * dy)
    return u, v


def make_shear_stretch(h: int, w: int, shear: float, stretch: float):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    dx = (xx - cx) / cx
    dy = (yy - cy) / cy
    u = stretch * (xx - cx) + shear * (yy - cy)
    v = 0.5 * stretch * (yy - cy) - 0.3 * shear * (xx - cx)
    return u, v


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

# Custom colormaps for mask overlay
MASK_CMAP_REF = ListedColormap(["none", "#2196F3"])    # blue = reference
MASK_CMAP_DEF = ListedColormap(["none", "#FF5722"])    # red = warped
MASK_CMAP_ORIG = ListedColormap(["none", "#4CAF50"])   # green = original


def draw_case(fig, gs_row, case_name, mask, u, v, n_iter=5):
    """Draw one test case as a row of 4 panels."""
    h, w = mask.shape
    warped = warp_mask(mask, u, v, n_iter=n_iter)

    d0, h0 = topology(mask)
    d1, h1 = topology(warped)
    topo_ok = (d0 == d1 and h0 == h1)

    # Panel 1: Original mask
    ax1 = fig.add_subplot(gs_row[0])
    ax1.imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax1.set_title(f"Reference Mask\n{d0}D {h0}H", fontsize=8)
    ax1.set_ylabel(case_name, fontsize=9, fontweight="bold")
    ax1.tick_params(labelsize=5)

    # Panel 2: Displacement field (quiver on mask)
    ax2 = fig.add_subplot(gs_row[1])
    ax2.imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1, alpha=0.3)
    step = max(h // 16, 1)
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    X, Y = np.meshgrid(xs, ys)
    U_q = u[::step, ::step]
    V_q = v[::step, ::step]
    mag = np.sqrt(U_q**2 + V_q**2)
    ax2.quiver(X, Y, U_q, V_q, mag, cmap="coolwarm", scale_units="xy",
               scale=1, width=0.003, headwidth=3, headlength=4)
    ax2.set_title(f"Displacement (max={np.sqrt(u**2+v**2).max():.1f} px)", fontsize=8)
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.tick_params(labelsize=5)

    # Panel 3: Warped mask
    ax3 = fig.add_subplot(gs_row[2])
    ax3.imshow(warped, cmap="gray", origin="upper", vmin=0, vmax=1)
    area_change = (warped.sum() / max(mask.sum(), 1) - 1) * 100
    topo_str = "OK" if topo_ok else f"CHANGED {d1}D {h1}H"
    topo_color = "green" if topo_ok else "red"
    ax3.set_title(f"Warped Mask (area {area_change:+.1f}%)\n"
                  f"Topology: {d1}D {h1}H [{topo_str}]",
                  fontsize=8, color=topo_color if not topo_ok else "black")
    ax3.tick_params(labelsize=5)

    # Panel 4: Overlay (blue=ref only, red=warped only, green=both)
    ax4 = fig.add_subplot(gs_row[3])
    overlay = np.zeros((h, w, 3), dtype=np.float64)
    ref_only = (mask > 0.5) & (warped < 0.5)
    warp_only = (mask < 0.5) & (warped > 0.5)
    both = (mask > 0.5) & (warped > 0.5)
    overlay[ref_only] = [0.13, 0.59, 0.95]    # blue
    overlay[warp_only] = [1.0, 0.34, 0.13]    # red/orange
    overlay[both] = [0.30, 0.69, 0.31]        # green
    ax4.imshow(overlay, origin="upper")
    ax4.set_title("Overlay (blue=ref, red=warped, green=both)", fontsize=7)
    ax4.tick_params(labelsize=5)

    return warped


# ═══════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════

IMG_SIZE = 256

CASES = [
    {
        "name": "1. Translation\n(tx=15, ty=10)",
        "mask_fn": lambda: make_square_mask(IMG_SIZE, IMG_SIZE),
        "disp_fn": lambda: make_translation(IMG_SIZE, IMG_SIZE, 15.0, 10.0),
    },
    {
        "name": "2. Stretch\n(ex=15%, ey=10%)",
        "mask_fn": lambda: make_annular_mask(IMG_SIZE, IMG_SIZE),
        "disp_fn": lambda: make_stretch(IMG_SIZE, IMG_SIZE, 0.15, 0.10),
    },
    {
        "name": "3. Rotation\n(15 degrees)",
        "mask_fn": lambda: make_square_mask(IMG_SIZE, IMG_SIZE),
        "disp_fn": lambda: make_rotation(IMG_SIZE, IMG_SIZE, 15.0),
    },
    {
        "name": "4. Quadratic\n(amp=10 px)",
        "mask_fn": lambda: make_multi_hole_mask(IMG_SIZE, IMG_SIZE),
        "disp_fn": lambda: make_quadratic(IMG_SIZE, IMG_SIZE, 10.0),
    },
    {
        "name": "5. Large Quadratic\n(amp=25 px)",
        "mask_fn": lambda: make_multi_hole_mask(IMG_SIZE, IMG_SIZE),
        "disp_fn": lambda: make_quadratic(IMG_SIZE, IMG_SIZE, 25.0),
    },
    {
        "name": "6. Shear+Stretch\nMulti-domain",
        "mask_fn": lambda: make_multi_domain_mask(IMG_SIZE, IMG_SIZE),
        "disp_fn": lambda: make_shear_stretch(IMG_SIZE, IMG_SIZE, 0.08, 0.12),
    },
]


# ═══════════════════════════════════════════════════════════════════
# Convergence analysis
# ═══════════════════════════════════════════════════════════════════

def convergence_page(pdf):
    """Page showing how fixed-point iterations affect accuracy."""
    h, w = IMG_SIZE, IMG_SIZE
    mask = make_multi_hole_mask(h, w)

    strains = [0.05, 0.15, 0.30, 0.50]
    max_iters = [1, 2, 3, 5, 8, 12]

    fig, axes = plt.subplots(len(strains), len(max_iters) + 1,
                             figsize=(22, 14))
    fig.suptitle("Convergence: Fixed-Point Iterations vs Strain Magnitude",
                 fontsize=14, y=0.98)

    # Reference result with many iterations
    for row, strain in enumerate(strains):
        u, v = make_stretch(h, w, strain, strain * 0.8)
        ref_result = warp_mask(mask, u, v, n_iter=20)

        # Original mask
        axes[row, 0].imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=8)
        axes[row, 0].set_ylabel(f"strain={strain:.0%}", fontsize=9,
                                 fontweight="bold")
        axes[row, 0].tick_params(labelsize=4)

        for col, ni in enumerate(max_iters):
            result = warp_mask(mask, u, v, n_iter=ni)
            diff = np.abs(result - ref_result).sum()

            ax = axes[row, col + 1]
            # Show overlay: green=match ref, red=differs
            overlay = np.zeros((h, w, 3))
            match = (result > 0.5) == (ref_result > 0.5)
            overlay[match & (ref_result > 0.5)] = [0.3, 0.7, 0.3]
            overlay[~match & (result > 0.5)] = [1.0, 0.3, 0.1]
            overlay[~match & (ref_result > 0.5)] = [0.1, 0.3, 1.0]
            ax.imshow(overlay, origin="upper")

            if row == 0:
                ax.set_title(f"n_iter={ni}", fontsize=8)
            ax.text(0.5, 0.02, f"diff={int(diff)} px",
                    transform=ax.transAxes, fontsize=6, ha="center",
                    color="white", bbox=dict(boxstyle="round,pad=0.2",
                                              facecolor="black", alpha=0.7))
            ax.tick_params(labelsize=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "warp_mask_validation.pdf"

    print("Generating warp_mask validation report ...")

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Cases 1-3 ─────────────────────────────────────
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle("Mask Warping Validation (Cases 1-3: Simple Deformations)",
                     fontsize=14, y=0.99)
        gs = fig.add_gridspec(3, 4, hspace=0.30, wspace=0.20)

        for i, case in enumerate(CASES[:3]):
            mask = case["mask_fn"]()
            u, v = case["disp_fn"]()
            draw_case(fig, [gs[i, j] for j in range(4)],
                      case["name"], mask, u, v)
            print(f"  {case['name'].split(chr(10))[0]} done")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Cases 4-6 ─────────────────────────────────────
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle("Mask Warping Validation (Cases 4-6: Complex Deformations)",
                     fontsize=14, y=0.99)
        gs = fig.add_gridspec(3, 4, hspace=0.30, wspace=0.20)

        for i, case in enumerate(CASES[3:6]):
            mask = case["mask_fn"]()
            u, v = case["disp_fn"]()
            draw_case(fig, [gs[i, j] for j in range(4)],
                      case["name"], mask, u, v)
            print(f"  {case['name'].split(chr(10))[0]} done")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Convergence analysis ──────────────────────────
        print("  Convergence analysis ...")
        convergence_page(pdf)

        # ── Page 4: Area preservation summary ─────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Area Preservation Analysis", fontsize=14, y=0.98)

        summaries = []
        for idx, case in enumerate(CASES):
            ax = axes[idx // 3, idx % 3]
            mask = case["mask_fn"]()
            u, v = case["disp_fn"]()

            # Area scaling = det(I + grad(u)), including cross terms
            det_f = det_F(u, v)
            expected_ratio = float(np.mean(det_f[mask > 0.5]))

            warped = warp_mask(mask, u, v)
            actual_ratio = warped.sum() / mask.sum()

            summaries.append({
                "name": case["name"].replace("\n", " "),
                "expected": expected_ratio,
                "actual": actual_ratio,
            })

            # Bar chart
            bars = ax.bar(
                ["Expected", "Actual"],
                [expected_ratio, actual_ratio],
                color=["#2196F3", "#FF5722"],
                width=0.5,
            )
            ax.set_ylim(min(expected_ratio, actual_ratio) * 0.9,
                        max(expected_ratio, actual_ratio) * 1.1)
            ax.set_title(case["name"].replace("\n", " "), fontsize=9)
            ax.set_ylabel("Area Ratio (warped / original)", fontsize=7)
            ax.tick_params(labelsize=6)

            for bar, val in zip(bars, [expected_ratio, actual_ratio]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 5: Summary table ─────────────────────────────────
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle("Warp Mask Validation Summary", fontsize=14, y=0.95)
        ax = fig.add_subplot(111)
        ax.axis("off")

        col_labels = [
            "Test Case", "Orig Area", "Warp Area",
            "Expected Ratio", "Actual Ratio", "Area Err",
            "Orig Topo", "Warp Topo", "Topo OK",
        ]
        rows = []
        for idx, case in enumerate(CASES):
            mask = case["mask_fn"]()
            u, v = case["disp_fn"]()
            warped = warp_mask(mask, u, v)
            s = summaries[idx]
            err_pct = abs(s["actual"] - s["expected"]) / max(s["expected"], 1e-12) * 100
            d0, h0 = topology(mask)
            d1, h1 = topology(warped)
            topo_ok = "YES" if (d0 == d1 and h0 == h1) else "NO"
            rows.append([
                s["name"],
                f"{int(mask.sum())}",
                f"{int(warped.sum())}",
                f"{s['expected']:.4f}",
                f"{s['actual']:.4f}",
                f"{err_pct:.2f}%",
                f"{d0}D {h0}H",
                f"{d1}D {h1}H",
                topo_ok,
            ])

        table = ax.table(
            cellText=rows, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 2.0)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(len(rows)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")
    print("  5 pages: simple cases + complex cases + convergence + area + summary")


if __name__ == "__main__":
    main()
