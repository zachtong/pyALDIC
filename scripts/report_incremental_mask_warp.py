#!/usr/bin/env python
"""Visual report: incremental mask warping over many frames.

Simulates incremental DIC tracking where the reference frame is updated
every N frames. At each update, the mask is warped forward using the
inter-frame displacement field. Tests cumulative error, topology stability,
and performance over 20+ consecutive warps.

Scenarios:
  A. Uniform stretch (constant strain per frame)
  B. Rotation (constant angular velocity)
  C. Quadratic with drift (spatially varying, accumulating)
  D. Complex mask with holes + mixed deformation

For each: compares incremental chain vs single direct warp (ground truth).

Output: reports/incremental_mask_warp.pdf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import label as ndimage_label, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from al_dic.utils.warp_mask import warp_mask

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

IMG_SIZE = 512


def topology(mask):
    h, w = mask.shape
    _, n_d = ndimage_label(mask > 0.5)
    lz, nz = ndimage_label(mask < 0.5)
    bl = set()
    bl.update(lz[0, :].tolist())
    bl.update(lz[h - 1, :].tolist())
    bl.update(lz[:, 0].tolist())
    bl.update(lz[:, w - 1].tolist())
    bl.discard(0)
    return n_d, nz - len(bl)


def iou(a, b):
    """Intersection-over-union for binary masks."""
    inter = ((a > 0.5) & (b > 0.5)).sum()
    union = ((a > 0.5) | (b > 0.5)).sum()
    return inter / max(union, 1)


def compose_displacement(u1, v1, u2, v2):
    """Compose two displacement fields: first (u1,v1), then (u2,v2).

    Given: point x -> x + d1(x) -> x + d1(x) + d2(x + d1(x))
    Result: total displacement d_total(x) = d1(x) + d2(x + d1(x))
    """
    h, w = u1.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Evaluate d2 at deformed positions (x + d1)
    deformed_x = xx + u1
    deformed_y = yy + v1
    coords = np.array([deformed_y.ravel(), deformed_x.ravel()])
    u2_at_def = map_coordinates(u2, coords, order=1, mode="constant", cval=0.0).reshape(h, w)
    v2_at_def = map_coordinates(v2, coords, order=1, mode="constant", cval=0.0).reshape(h, w)
    return u1 + u2_at_def, v1 + v2_at_def


# ═══════════════════════════════════════════════════════════════════
# Masks
# ═══════════════════════════════════════════════════════════════════

def make_square_mask():
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    m = IMG_SIZE // 6
    mask[m : IMG_SIZE - m, m : IMG_SIZE - m] = 1.0
    return mask


def make_annular_mask():
    h, w = IMG_SIZE, IMG_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ((r < w * 0.38) & (r > w * 0.10)).astype(np.float64)


def make_complex_mask():
    h, w = IMG_SIZE, IMG_SIZE
    mask = np.zeros((h, w), dtype=np.float64)
    mask[h // 8 : 7 * h // 8, w // 8 : 7 * w // 8] = 1.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    holes = [
        (w * 0.30, h * 0.35, w * 0.06),
        (w * 0.60, h * 0.50, w * 0.08),
        (w * 0.40, h * 0.70, w * 0.05),
    ]
    for hx, hy, hr in holes:
        mask[np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2) < hr] = 0.0
    return mask


# ═══════════════════════════════════════════════════════════════════
# Per-frame displacement generators
# ═══════════════════════════════════════════════════════════════════

def per_frame_stretch(strain_per_frame=0.005):
    """Uniform radial stretch from center, per frame."""
    h, w = IMG_SIZE, IMG_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    u = strain_per_frame * (xx - cx)
    v = strain_per_frame * (yy - cy)
    return u, v


def per_frame_rotation(deg_per_frame=1.0):
    """Rigid rotation about center, per frame."""
    h, w = IMG_SIZE, IMG_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    theta = np.radians(deg_per_frame)
    dx, dy = xx - cx, yy - cy
    u = (np.cos(theta) - 1) * dx - np.sin(theta) * dy
    v = np.sin(theta) * dx + (np.cos(theta) - 1) * dy
    return u, v


def per_frame_quadratic(amp=0.8):
    """Spatially varying quadratic deformation, per frame."""
    h, w = IMG_SIZE, IMG_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    dx = (xx - cx) / cx
    dy = (yy - cy) / cy
    u = amp * (0.5 * dx ** 2 + 0.3 * dy ** 2 + 0.2 * dx * dy + 0.1 * dx)
    v = amp * (0.3 * dx ** 2 + 0.4 * dy ** 2 - 0.1 * dx * dy + 0.05 * dy)
    return u, v


def per_frame_mixed(frame_idx):
    """Mixed: stretch + small rotation + quadratic, varying per frame."""
    h, w = IMG_SIZE, IMG_SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h / 2
    dx, dy = (xx - cx) / cx, (yy - cy) / cy

    # Small stretch
    strain = 0.003
    u = strain * (xx - cx)
    v = strain * (yy - cy)

    # Small rotation (alternating direction)
    theta = np.radians(0.3 * (-1) ** frame_idx)
    ddx, ddy = xx - cx, yy - cy
    u += (np.cos(theta) - 1) * ddx - np.sin(theta) * ddy
    v += np.sin(theta) * ddx + (np.cos(theta) - 1) * ddy

    # Small quadratic
    u += 0.3 * (dx ** 2 + 0.5 * dy ** 2)
    v += 0.2 * (0.5 * dx ** 2 + dy ** 2)

    return u, v


# ═══════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════

N_FRAMES = 30

SCENARIOS = [
    {
        "name": "A. Uniform Stretch\n(0.5%/frame, 30 frames -> 15% total)",
        "short": "Stretch",
        "mask_fn": make_square_mask,
        "disp_fn": lambda _: per_frame_stretch(0.005),
        "n_frames": N_FRAMES,
    },
    {
        "name": "B. Rotation\n(1 deg/frame, 30 frames -> 30 deg total)",
        "short": "Rotation",
        "mask_fn": make_annular_mask,
        "disp_fn": lambda _: per_frame_rotation(1.0),
        "n_frames": N_FRAMES,
    },
    {
        "name": "C. Quadratic\n(amp=0.8/frame, 30 frames)",
        "short": "Quadratic",
        "mask_fn": make_complex_mask,
        "disp_fn": lambda _: per_frame_quadratic(0.8),
        "n_frames": N_FRAMES,
    },
    {
        "name": "D. Mixed Deformation\n(stretch+rot+quad, complex mask)",
        "short": "Mixed",
        "mask_fn": make_complex_mask,
        "disp_fn": per_frame_mixed,
        "n_frames": N_FRAMES,
    },
]


def run_scenario(scenario):
    """Run one scenario, return per-frame metrics."""
    mask_0 = scenario["mask_fn"]()
    n = scenario["n_frames"]
    d0, h0 = topology(mask_0)

    # ── Incremental chain ──────────────────────────────────────
    mask_inc = mask_0.copy()
    # Accumulate total displacement for ground truth comparison
    u_total = np.zeros_like(mask_0)
    v_total = np.zeros_like(mask_0)

    metrics = {
        "frame": [],
        "iou": [],
        "area_ratio": [],
        "topo_domains": [],
        "topo_holes": [],
        "topo_ok": [],
        "time_ms": [],
        "pixel_diff": [],
    }

    area_0 = mask_0.sum()

    for i in range(n):
        u_frame, v_frame = scenario["disp_fn"](i)

        # Accumulate total displacement (compose)
        u_total, v_total = compose_displacement(u_total, v_total, u_frame, v_frame)

        # Incremental warp
        t0 = time.perf_counter()
        mask_inc = warp_mask(mask_inc, u_frame, v_frame)
        elapsed = (time.perf_counter() - t0) * 1000

        # Ground truth: direct warp from frame 0
        mask_gt = warp_mask(mask_0, u_total, v_total)

        d_inc, h_inc = topology(mask_inc)
        iou_val = iou(mask_inc, mask_gt)
        diff_px = int(np.abs(mask_inc - mask_gt).sum())

        metrics["frame"].append(i + 1)
        metrics["iou"].append(iou_val)
        metrics["area_ratio"].append(mask_inc.sum() / max(area_0, 1))
        metrics["topo_domains"].append(d_inc)
        metrics["topo_holes"].append(h_inc)
        metrics["topo_ok"].append(d_inc == d0 and h_inc == h0)
        metrics["time_ms"].append(elapsed)
        metrics["pixel_diff"].append(diff_px)

    # Store final masks for visualization
    metrics["mask_0"] = mask_0
    metrics["mask_inc_final"] = mask_inc
    metrics["mask_gt_final"] = mask_gt
    metrics["orig_topo"] = (d0, h0)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_metrics_page(pdf, all_results):
    """Page 1: IoU, area ratio, pixel diff, timing across frames."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Incremental Mask Warp: {N_FRAMES}-Frame Chain ({IMG_SIZE}x{IMG_SIZE})",
        fontsize=14, y=0.98,
    )

    colors = ["C0", "C1", "C2", "C3"]

    # IoU vs frame
    ax = axes[0, 0]
    for idx, (sc, m) in enumerate(all_results):
        ax.plot(m["frame"], m["iou"], "-o", color=colors[idx],
                label=sc["short"], markersize=3, linewidth=1.5)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("IoU (incremental vs direct)", fontsize=10)
    ax.set_title("Accuracy: IoU with Ground Truth", fontsize=11)
    ax.set_ylim(0.9, 1.005)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Area ratio vs frame
    ax = axes[0, 1]
    for idx, (sc, m) in enumerate(all_results):
        ax.plot(m["frame"], m["area_ratio"], "-o", color=colors[idx],
                label=sc["short"], markersize=3, linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Area Ratio (warped / original)", fontsize=10)
    ax.set_title("Area Drift Over Frames", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pixel diff vs frame
    ax = axes[1, 0]
    for idx, (sc, m) in enumerate(all_results):
        total_px = m["mask_0"].sum()
        pct = [d / total_px * 100 for d in m["pixel_diff"]]
        ax.plot(m["frame"], pct, "-o", color=colors[idx],
                label=sc["short"], markersize=3, linewidth=1.5)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Pixel Difference (%)", fontsize=10)
    ax.set_title("Cumulative Error (incremental vs direct)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Timing
    ax = axes[1, 1]
    for idx, (sc, m) in enumerate(all_results):
        ax.plot(m["frame"], m["time_ms"], "-o", color=colors[idx],
                label=sc["short"], markersize=3, linewidth=1.5)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Time per warp (ms)", fontsize=10)
    ax.set_title("Performance", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_topology_page(pdf, all_results):
    """Page 2: Topology stability per scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Topology Preservation Over Frames", fontsize=14, y=0.98)

    for idx, (sc, m) in enumerate(all_results):
        ax = axes[idx // 2, idx % 2]
        frames = m["frame"]
        d0, h0 = m["orig_topo"]

        ax.plot(frames, m["topo_domains"], "s-", color="C0",
                label="Domains", markersize=4, linewidth=1.5)
        ax.plot(frames, m["topo_holes"], "^-", color="C3",
                label="Holes", markersize=4, linewidth=1.5)
        ax.axhline(d0, color="C0", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(h0, color="C3", linestyle="--", linewidth=0.8, alpha=0.5)

        n_ok = sum(m["topo_ok"])
        ax.set_title(f"{sc['short']} (orig: {d0}D {h0}H, "
                     f"preserved: {n_ok}/{len(frames)})", fontsize=10)
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, max(max(m["topo_domains"]), max(m["topo_holes"])) + 1)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_visual_page(pdf, all_results):
    """Page 3: Visual comparison — original, incremental final, GT final, overlay."""
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(f"Final Mask Comparison (Frame {N_FRAMES})", fontsize=14, y=0.99)

    col_titles = ["Original (Frame 0)", "Incremental Chain",
                  "Direct Warp (GT)", "Overlay (blue=GT only, red=Inc only)"]

    for idx, (sc, m) in enumerate(all_results):
        mask_0 = m["mask_0"]
        mask_inc = m["mask_inc_final"]
        mask_gt = m["mask_gt_final"]

        axes[idx, 0].imshow(mask_0, cmap="gray", origin="upper", vmin=0, vmax=1)
        axes[idx, 0].set_ylabel(sc["short"], fontsize=11, fontweight="bold")

        axes[idx, 1].imshow(mask_inc, cmap="gray", origin="upper", vmin=0, vmax=1)
        d1, h1 = topology(mask_inc)
        axes[idx, 1].set_title(f"{d1}D {h1}H" if idx == 0 else f"{d1}D {h1}H",
                                fontsize=8)

        axes[idx, 2].imshow(mask_gt, cmap="gray", origin="upper", vmin=0, vmax=1)
        d2, h2 = topology(mask_gt)
        axes[idx, 2].set_title(f"{d2}D {h2}H", fontsize=8)

        # Overlay
        h, w = mask_0.shape
        overlay = np.zeros((h, w, 3))
        gt_only = (mask_gt > 0.5) & (mask_inc < 0.5)
        inc_only = (mask_inc > 0.5) & (mask_gt < 0.5)
        both = (mask_gt > 0.5) & (mask_inc > 0.5)
        overlay[gt_only] = [0.13, 0.59, 0.95]
        overlay[inc_only] = [1.0, 0.34, 0.13]
        overlay[both] = [0.30, 0.69, 0.31]
        axes[idx, 3].imshow(overlay, origin="upper")
        iou_val = iou(mask_inc, mask_gt)
        axes[idx, 3].set_title(f"IoU={iou_val:.4f}", fontsize=9)

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t + "\n" + axes[0, j].get_title(), fontsize=9)

    for ax in axes.flat:
        ax.tick_params(labelsize=5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_summary_table(pdf, all_results):
    """Page 4: Summary table."""
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Incremental Mask Warp Summary", fontsize=14, y=0.95)
    ax = fig.add_subplot(111)
    ax.axis("off")

    col_labels = [
        "Scenario", "Frames", "Orig Topo",
        "Final IoU", "Final Area Ratio",
        "Max Pixel Diff (%)", "Topo Preserved",
        "Avg Time (ms)", "Total Time (ms)",
    ]
    rows = []
    for sc, m in all_results:
        total_px = m["mask_0"].sum()
        max_diff_pct = max(m["pixel_diff"]) / total_px * 100
        n_ok = sum(m["topo_ok"])
        d0, h0 = m["orig_topo"]
        rows.append([
            sc["short"],
            str(len(m["frame"])),
            f"{d0}D {h0}H",
            f"{m['iou'][-1]:.4f}",
            f"{m['area_ratio'][-1]:.4f}",
            f"{max_diff_pct:.2f}%",
            f"{n_ok}/{len(m['frame'])}",
            f"{np.mean(m['time_ms']):.1f}",
            f"{sum(m['time_ms']):.0f}",
        ])

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.2)

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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / "incremental_mask_warp.pdf"

    print(f"Incremental mask warp test: {N_FRAMES} frames, {IMG_SIZE}x{IMG_SIZE}")

    all_results = []
    for sc in SCENARIOS:
        print(f"  Running: {sc['short']} ...", end=" ", flush=True)
        m = run_scenario(sc)
        print(f"final IoU={m['iou'][-1]:.4f}, "
              f"avg={np.mean(m['time_ms']):.0f}ms/frame")
        all_results.append((sc, m))

    print("Generating report ...")
    with PdfPages(str(pdf_path)) as pdf:
        plot_metrics_page(pdf, all_results)
        plot_topology_page(pdf, all_results)
        plot_visual_page(pdf, all_results)
        plot_summary_table(pdf, all_results)

    print(f"\nReport saved to: {pdf_path}")
    print("  4 pages: metrics + topology + visual comparison + summary")


if __name__ == "__main__":
    main()
