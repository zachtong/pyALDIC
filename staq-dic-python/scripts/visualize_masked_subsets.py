#!/usr/bin/env python3
"""Visual diagnostic for masked-subset IC-GN window splitting.

Creates a report showing, for each node near a mask boundary:
  - The raw reference image patch
  - The mask patch (original mask over the subset window)
  - The connected-component mask (bw) — pixels actually used by IC-GN
  - Gradient magnitude from raw image vs. from masked image

Uses small synthetic images (64x64) with ~6 carefully placed nodes
to make each case clearly readable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, label

# ── project imports ──
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.solver.icgn_batch import (
    _connected_center_mask,
    _precompute_subsets_6dof_python,
)


OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "masked_subset_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Helpers
# =====================================================================

def make_speckle(h=64, w=64, sigma=2.5, seed=42):
    """Small smooth speckle image for diagnostics."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def extract_subset_info(img_raw, mask, coord, winsize):
    """Extract detailed per-node subset info for visualization.

    Returns dict with:
        x0, y0, x_lo, x_hi, y_lo, y_hi,
        raw_patch, mask_patch, bw, ref_sub_new, ref_sub_old,
        gx_new, gx_old, status
    """
    x0, y0 = coord
    half_w = winsize // 2
    h, w = img_raw.shape

    x_lo = int(round(x0) - half_w)
    x_hi = int(round(x0) + half_w)
    y_lo = int(round(y0) - half_w)
    y_hi = int(round(y0) + half_w)

    info = dict(x0=x0, y0=y0, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
                winsize=winsize)

    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        info["status"] = "OOB"
        return info

    raw_patch = img_raw[y_lo:y_hi + 1, x_lo:x_hi + 1]
    mask_patch = mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
    bw = _connected_center_mask(mask_patch > 0.5)

    n_connected = int(np.sum(bw > 0.5))
    total = mask_patch.size

    # NEW: ref_sub from raw image, masked by bw
    ref_sub_new = raw_patch * bw

    # OLD: ref_sub from (img*mask), masked by bw
    masked_img = img_raw * mask
    ref_sub_old = masked_img[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw

    # Gradients
    Df_new = compute_image_gradient(masked_img, mask, img_raw=img_raw)
    Df_old = compute_image_gradient(masked_img, mask)

    gx_new = Df_new.df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw
    gx_old = Df_old.df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw

    # Status
    if n_connected < int(0.5 * total):
        status = f"REJECTED (ratio={n_connected/total:.2f}<0.5)"
    else:
        status = f"VALID (ratio={n_connected/total:.2f})"

    info.update(
        raw_patch=raw_patch, mask_patch=mask_patch, bw=bw,
        ref_sub_new=ref_sub_new, ref_sub_old=ref_sub_old,
        gx_new=gx_new, gx_old=gx_old,
        n_connected=n_connected, total=total,
        status=status,
    )
    return info


# =====================================================================
# Figure 1: Overview — image + mask + node locations
# =====================================================================

def plot_overview(img, mask, coords, winsize, title, filename):
    """Plot image with mask overlay and subset windows for each node."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: raw image with mask contour
    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
    ax.set_title("Raw image + mask boundary")

    # Draw subset windows
    half_w = winsize // 2
    colors = plt.cm.Set1(np.linspace(0, 1, len(coords)))
    for i, (x0, y0) in enumerate(coords):
        x_lo = round(x0) - half_w
        y_lo = round(y0) - half_w
        rect = Rectangle(
            (x_lo - 0.5, y_lo - 0.5), winsize + 1, winsize + 1,
            linewidth=2, edgecolor=colors[i], facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.plot(x0, y0, "o", color=colors[i], markersize=6, markeredgecolor="white")
        ax.annotate(
            f"N{i}", (x0 + 1, y0 - 2), color=colors[i], fontsize=10,
            fontweight="bold", backgroundcolor=(1, 1, 1, 0.7),
        )

    # Right: mask with alpha overlay
    ax = axes[1]
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask > 0.5] = [0, 0.6, 0, 0.3]   # green = valid
    rgba[mask < 0.5] = [1, 0, 0, 0.3]      # red = masked out
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.imshow(rgba)
    ax.set_title("Mask overlay (green=valid, red=masked)")

    for i, (x0, y0) in enumerate(coords):
        ax.plot(x0, y0, "o", color=colors[i], markersize=6, markeredgecolor="white")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# =====================================================================
# Figure 2: Per-node subset detail
# =====================================================================

def plot_node_details(infos, title, filename):
    """For each node, show 5 columns: raw patch, mask, bw, gx_old, gx_new."""
    n_nodes = len(infos)
    fig, axes = plt.subplots(n_nodes, 5, figsize=(18, 3.2 * n_nodes + 0.8))
    if n_nodes == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Raw patch",
        "Mask patch",
        "Connected mask (bw)\n(pixels used by IC-GN)",
        "Gradient dx (OLD)\n(from img*mask)",
        "Gradient dx (NEW)\n(from raw img)",
    ]

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold")

    # Custom colormaps
    bw_cmap = ListedColormap(["#ff4444", "#44cc44"])  # red=dropped, green=kept

    for i, info in enumerate(infos):
        row_label = f"N{i}  ({info['x0']:.0f},{info['y0']:.0f})\n{info['status']}"
        axes[i, 0].set_ylabel(row_label, fontsize=9, fontweight="bold", rotation=0,
                              labelpad=90, va="center")

        if info.get("status", "").startswith("OOB"):
            for c in range(5):
                axes[i, c].text(0.5, 0.5, "OUT OF BOUNDS", transform=axes[i, c].transAxes,
                                ha="center", va="center", fontsize=12, color="red")
                axes[i, c].set_xticks([])
                axes[i, c].set_yticks([])
            continue

        S = info["raw_patch"].shape[0]
        center_y = S // 2
        center_x = S // 2

        # Col 0: raw patch
        ax = axes[i, 0]
        ax.imshow(info["raw_patch"], cmap="gray", vmin=0, vmax=255)
        ax.plot(center_x, center_y, "r+", markersize=12, markeredgewidth=2)
        ax.set_xticks(range(0, S, max(1, S // 4)))
        ax.set_yticks(range(0, S, max(1, S // 4)))

        # Col 1: mask patch
        ax = axes[i, 1]
        ax.imshow(info["mask_patch"], cmap=bw_cmap, vmin=0, vmax=1)
        ax.plot(center_x, center_y, "k+", markersize=12, markeredgewidth=2)
        n_masked = int(np.sum(info["mask_patch"] < 0.5))
        ax.set_xlabel(f"{n_masked}/{info['total']} masked", fontsize=8)

        # Col 2: connected component mask (bw)
        ax = axes[i, 2]
        # Show 3 categories: kept (green), dropped-in-mask (yellow), outside-mask (red)
        display = np.zeros_like(info["mask_patch"])
        display[info["mask_patch"] < 0.5] = 0       # outside mask → red
        display[(info["mask_patch"] > 0.5) & (info["bw"] < 0.5)] = 0.5  # in mask but dropped
        display[info["bw"] > 0.5] = 1.0             # kept
        tri_cmap = ListedColormap(["#ff4444", "#ffcc00", "#44cc44"])
        ax.imshow(display, cmap=tri_cmap, vmin=0, vmax=1)
        ax.plot(center_x, center_y, "k+", markersize=12, markeredgewidth=2)
        n_kept = int(np.sum(info["bw"] > 0.5))
        ax.set_xlabel(f"{n_kept}/{info['total']} kept", fontsize=8)

        # Col 3: gradient dx OLD
        ax = axes[i, 3]
        gx_old = info["gx_old"]
        vmax_old = max(np.max(np.abs(gx_old)), 1)
        im = ax.imshow(gx_old, cmap="RdBu_r", vmin=-vmax_old, vmax=vmax_old)
        ax.plot(center_x, center_y, "k+", markersize=12, markeredgewidth=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        valid_gx = gx_old[info["bw"] > 0.5]
        if len(valid_gx) > 0:
            ax.set_xlabel(f"max|gx|={np.max(np.abs(valid_gx)):.1f}", fontsize=8)

        # Col 4: gradient dx NEW
        ax = axes[i, 4]
        gx_new = info["gx_new"]
        vmax_new = max(np.max(np.abs(gx_new)), 1)
        # Use same scale as OLD for fair comparison
        vmax_both = max(vmax_old, vmax_new)
        im = ax.imshow(gx_new, cmap="RdBu_r", vmin=-vmax_both, vmax=vmax_both)
        ax.plot(center_x, center_y, "k+", markersize=12, markeredgewidth=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        valid_gx = gx_new[info["bw"] > 0.5]
        if len(valid_gx) > 0:
            ax.set_xlabel(f"max|gx|={np.max(np.abs(valid_gx)):.1f}", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# =====================================================================
# Figure 3: precompute_subsets_6dof end-to-end comparison
# =====================================================================

def plot_precompute_comparison(img, mask, coords, winsize, title, filename):
    """Run actual precompute on same nodes, compare old vs new pipeline."""
    Df_new = compute_image_gradient(img * mask, mask, img_raw=img)
    Df_old = compute_image_gradient(img * mask, mask)

    # NEW pipeline: raw image + mask-based validity
    pre_new = _precompute_subsets_6dof_python(
        coords, img, Df_new.df_dx, Df_new.df_dy, mask, winsize,
    )

    # OLD pipeline: masked image + pixel-value-based validity
    # (simulate old behavior by passing img*mask as img_ref)
    pre_old = _precompute_subsets_6dof_python_old(
        coords, img * mask, Df_old.df_dx, Df_old.df_dy, mask, winsize,
    )

    N = coords.shape[0]
    fig, axes = plt.subplots(N, 4, figsize=(14, 3.2 * N + 0.8))
    if N == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "OLD: ref_sub\n(from img*mask)",
        "NEW: ref_sub\n(from raw img)",
        "OLD: gx_sub",
        "NEW: gx_sub",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold")

    for i in range(N):
        label = f"N{i} ({coords[i,0]:.0f},{coords[i,1]:.0f})"
        valid_old = pre_old["valid"][i]
        valid_new = pre_new["valid"][i]
        status = f"old={'V' if valid_old else 'X'} new={'V' if valid_new else 'X'}"
        axes[i, 0].set_ylabel(f"{label}\n{status}", fontsize=9, fontweight="bold",
                              rotation=0, labelpad=80, va="center")

        S = pre_new["Sy"]

        # OLD ref_sub
        ax = axes[i, 0]
        ref_old = pre_old["ref_all"][i]
        ax.imshow(ref_old, cmap="gray", vmin=0, vmax=255)
        ax.plot(S // 2, S // 2, "r+", markersize=10, markeredgewidth=2)

        # NEW ref_sub
        ax = axes[i, 1]
        ref_new = pre_new["ref_all"][i]
        ax.imshow(ref_new, cmap="gray", vmin=0, vmax=255)
        ax.plot(S // 2, S // 2, "r+", markersize=10, markeredgewidth=2)

        # OLD gx
        ax = axes[i, 2]
        gx_old = pre_old["gx_all"][i]
        vmax = max(np.max(np.abs(gx_old)), np.max(np.abs(pre_new["gx_all"][i])), 1)
        im = ax.imshow(gx_old, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot(S // 2, S // 2, "k+", markersize=10, markeredgewidth=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # NEW gx
        ax = axes[i, 3]
        gx_new = pre_new["gx_all"][i]
        im = ax.imshow(gx_new, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot(S // 2, S // 2, "k+", markersize=10, markeredgewidth=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def _precompute_subsets_6dof_python_old(
    coords, img_ref_masked, df_dx, df_dy, img_ref_mask, winsize,
):
    """OLD precompute logic (before masked-subset fix) for comparison.

    Uses pixel-value proxy (abs(val) < 1e-10) for validity detection.
    """
    N = coords.shape[0]
    half_w = winsize // 2
    Sy = Sx = winsize + 1
    h, w = img_ref_masked.shape

    ref_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    gx_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    gy_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    mask_all = np.zeros((N, Sy, Sx), dtype=np.float64)
    H_all = np.zeros((N, 6, 6), dtype=np.float64)
    meanf_all = np.zeros(N, dtype=np.float64)
    bottomf_all = np.ones(N, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)
    mark_hole = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        x0, y0 = coords[i]
        x_lo = int(round(x0) - half_w)
        x_hi = int(round(x0) + half_w)
        y_lo = int(round(y0) - half_w)
        y_hi = int(round(y0) + half_w)

        if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
            mark_hole[i] = True
            continue

        mask_patch = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ref_patch = img_ref_masked[y_lo:y_hi + 1, x_lo:x_hi + 1] * mask_patch

        n_masked = np.sum(np.abs(ref_patch) < 1e-10)
        if n_masked > 0.4 * ref_patch.size:
            mark_hole[i] = True
            continue

        bw = _connected_center_mask(mask_patch > 0.5)
        ref_sub = ref_patch * bw
        gx_sub = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw
        gy_sub = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw

        ref_all[i] = ref_sub
        gx_all[i] = gx_sub
        gy_all[i] = gy_sub
        mask_all[i] = bw

        nz = np.abs(ref_sub) > 1e-10
        n_valid = nz.sum()
        if n_valid < 4:
            continue

        meanf_all[i] = np.mean(ref_sub[nz])
        varf = np.var(ref_sub[nz])
        bottomf_all[i] = np.sqrt(max((n_valid - 1) * varf, 1e-30))
        valid[i] = True

    return {
        "ref_all": ref_all, "gx_all": gx_all, "gy_all": gy_all,
        "mask_all": mask_all, "H_all": H_all,
        "meanf_all": meanf_all, "bottomf_all": bottomf_all,
        "valid": valid, "mark_hole": mark_hole,
        "Sy": Sy, "Sx": Sx, "img_h": h, "img_w": w,
    }


# =====================================================================
# Scenario definitions
# =====================================================================

def run_scenario_straight_edge():
    """Scenario A: Straight vertical mask edge at x=40."""
    print("\n=== Scenario A: Straight vertical edge ===")
    h, w, winsize = 64, 64, 8
    img = make_speckle(h, w, sigma=2.0, seed=10)
    mask = np.ones((h, w), dtype=np.float64)
    mask[:, 40:] = 0.0

    # N0: fully interior, N1: partially clipped, N2: mostly outside, N3: at boundary
    coords = np.array([
        [24.0, 32.0],   # N0: fully inside mask (far from edge)
        [37.0, 32.0],   # N1: 3px from edge, subset clips into masked region
        [44.0, 32.0],   # N2: 4px outside edge, center in masked region
        [40.0, 32.0],   # N3: exactly on edge
    ])

    plot_overview(img, mask, coords, winsize,
                  "Scenario A: Straight edge (x=40)", "A_overview.png")

    infos = [extract_subset_info(img, mask, c, winsize) for c in coords]
    plot_node_details(infos,
                      "Scenario A: Per-node subset detail (winsize=8)",
                      "A_node_details.png")

    plot_precompute_comparison(img, mask, coords, winsize,
                               "Scenario A: OLD vs NEW precompute",
                               "A_precompute_cmp.png")


def run_scenario_circular_hole():
    """Scenario B: Circular hole in center."""
    print("\n=== Scenario B: Circular hole ===")
    h, w, winsize = 64, 64, 8
    img = make_speckle(h, w, sigma=2.0, seed=20)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.ones((h, w), dtype=np.float64)
    mask[((xx - 32) ** 2 + (yy - 32) ** 2) < 12 ** 2] = 0.0

    coords = np.array([
        [16.0, 32.0],   # N0: far from hole
        [24.0, 32.0],   # N1: subset touches hole edge
        [28.0, 32.0],   # N2: subset straddles hole boundary
        [32.0, 18.0],   # N3: above hole, near inner edge
    ])

    plot_overview(img, mask, coords, winsize,
                  "Scenario B: Circular hole (r=12)", "B_overview.png")

    infos = [extract_subset_info(img, mask, c, winsize) for c in coords]
    plot_node_details(infos,
                      "Scenario B: Per-node subset detail (winsize=8)",
                      "B_node_details.png")

    plot_precompute_comparison(img, mask, coords, winsize,
                               "Scenario B: OLD vs NEW precompute",
                               "B_precompute_cmp.png")


def run_scenario_two_islands():
    """Scenario C: Narrow gap creating two disconnected regions in subset."""
    print("\n=== Scenario C: Narrow gap (two islands) ===")
    h, w, winsize = 128, 128, 20
    img = make_speckle(h, w, sigma=3.0, seed=30)
    mask = np.ones((h, w), dtype=np.float64)
    mask[:, 60:64] = 0.0  # 4px-wide vertical gap at x=60..63

    coords = np.array([
        [54.0, 64.0],   # N0: center left of gap, window [44..74] spans gap at 60
        [70.0, 64.0],   # N1: center right of gap, window [60..80] spans gap at 60
        [64.0, 64.0],   # N2: center ON the gap (center in masked region)
        [32.0, 64.0],   # N3: fully left, no gap in subset (control)
    ])

    plot_overview(img, mask, coords, winsize,
                  "Scenario C: Narrow gap (x=60..63, winsize=20)", "C_overview.png")

    infos = [extract_subset_info(img, mask, c, winsize) for c in coords]
    plot_node_details(infos,
                      "Scenario C: Per-node subset detail (winsize=20)",
                      "C_node_details.png")

    plot_precompute_comparison(img, mask, coords, winsize,
                               "Scenario C: OLD vs NEW precompute (winsize=20)",
                               "C_precompute_cmp.png")


def run_scenario_annular():
    """Scenario D: Annular mask (inner + outer boundary)."""
    print("\n=== Scenario D: Annular mask ===")
    h, w, winsize = 96, 96, 8
    img = make_speckle(h, w, sigma=2.0, seed=40)
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = (xx - 48) ** 2 + (yy - 48) ** 2
    mask = ((dist2 <= 40 ** 2) & (dist2 > 20 ** 2)).astype(np.float64)

    coords = np.array([
        [30.0, 48.0],   # N0: mid-annulus (fully valid)
        [24.0, 48.0],   # N1: near inner boundary, subset clips inner edge
        [12.0, 48.0],   # N2: near outer boundary, subset clips outer edge
        [48.0, 28.0],   # N3: top of annulus, near inner edge
        [48.0, 10.0],   # N4: outside outer boundary
    ])

    plot_overview(img, mask, coords, winsize,
                  "Scenario D: Annular mask (r_in=14, r_out=28)", "D_overview.png")

    infos = [extract_subset_info(img, mask, c, winsize) for c in coords]
    plot_node_details(infos,
                      "Scenario D: Per-node subset detail (winsize=8)",
                      "D_node_details.png")

    plot_precompute_comparison(img, mask, coords, winsize,
                               "Scenario D: OLD vs NEW precompute",
                               "D_precompute_cmp.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    run_scenario_straight_edge()
    run_scenario_circular_hole()
    run_scenario_two_islands()
    run_scenario_annular()
    print(f"\nAll figures saved to {OUT_DIR}")
