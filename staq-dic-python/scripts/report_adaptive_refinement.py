"""Comprehensive visualization report for the adaptive refinement framework.

Generates a multi-page PDF report covering:
  Part 1: Mesh comparison (uniform, geometry, manual, post-solve, combinations)
  Part 2: Per-frame independent mesh (incremental tracking mode)
  Part 3: Per-criterion independent min_element_size
  Part 4: Feasibility, accuracy, efficiency benchmarks

Usage:
    cd staq-dic-python
    python scripts/report_adaptive_refinement.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np
from numpy.typing import NDArray

# --- Project imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICMesh, DICPara, GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.mesh.refinement import (
    RefinementContext,
    RefinementPolicy,
    refine_mesh,
)
from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from staq_dic.mesh.criteria.manual_selection import ManualSelectionCriterion

# Also import conftest helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conftest import generate_speckle, apply_displacement_lagrangian


# ============================================================================
# Helper functions
# ============================================================================

def plot_mesh(
    ax: plt.Axes,
    mesh: DICMesh,
    title: str = "",
    color: str = "steelblue",
    show_nodes: bool = True,
    mask: NDArray | None = None,
    highlight_elements: NDArray | None = None,
    highlight_color: str = "orangered",
) -> None:
    """Draw a quadrilateral mesh on the given axes."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem[:, :4]
    n_elem = elems.shape[0]

    # Draw mask background
    if mask is not None:
        ax.imshow(
            mask, cmap="gray", origin="upper", alpha=0.3,
            extent=[0, mask.shape[1], mask.shape[0], 0],
        )

    # Draw elements as quadrilaterals
    segments = []
    for i in range(n_elem):
        corners = elems[i]
        pts = coords[corners]
        for j in range(4):
            p0 = pts[j]
            p1 = pts[(j + 1) % 4]
            segments.append([p0, p1])

    lc = LineCollection(segments, colors=color, linewidths=0.5, alpha=0.7)
    ax.add_collection(lc)

    # Highlight specific elements
    if highlight_elements is not None and len(highlight_elements) > 0:
        patches = []
        for idx in highlight_elements:
            if idx < n_elem:
                corners = elems[idx]
                pts = coords[corners]
                poly = mpatches.Polygon(pts, closed=True)
                patches.append(poly)
        if patches:
            pc = PatchCollection(
                patches, alpha=0.35, facecolor=highlight_color,
                edgecolor=highlight_color, linewidth=1,
            )
            ax.add_collection(pc)

    # Draw nodes
    if show_nodes:
        ax.scatter(
            coords[:, 0], coords[:, 1], s=2, c=color, zorder=5, alpha=0.8,
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(coords[:, 0].min() - 5, coords[:, 0].max() + 5)
    ax.set_ylim(coords[:, 1].max() + 5, coords[:, 1].min() - 5)  # flip y
    ax.tick_params(labelsize=7)


def mesh_stats_text(mesh: DICMesh) -> str:
    """Return a summary string for a mesh."""
    n_nodes = mesh.coordinates_fem.shape[0]
    n_elem = mesh.elements_fem.shape[0]
    corners = mesh.elements_fem[:, :4]
    dx = mesh.coordinates_fem[corners[:, 2], 0] - mesh.coordinates_fem[corners[:, 0], 0]
    sizes = np.abs(dx)
    unique_sizes = np.unique(np.round(sizes, 1))
    return (
        f"Nodes: {n_nodes}  |  Elements: {n_elem}\n"
        f"Element sizes: {', '.join(f'{s:.0f}' for s in sorted(unique_sizes))} px"
    )


def make_annular_mask(h, w, cx, cy, r_outer, r_inner):
    """Create an annular (ring) binary mask."""
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.ones((h, w), dtype=np.float64)
    mask[dist < r_inner] = 0.0
    mask[dist > r_outer] = 0.0
    return mask


def make_rect_hole_mask(h, w, hole_x, hole_y, hole_w, hole_h):
    """Create a mask with a rectangular hole."""
    mask = np.ones((h, w), dtype=np.float64)
    mask[hole_y:hole_y + hole_h, hole_x:hole_x + hole_w] = 0.0
    return mask


# ============================================================================
# Part 1: Mesh Comparison
# ============================================================================

def part1_mesh_comparison(pdf: PdfPages) -> None:
    """Compare uniform, geometry, manual, post-solve, and combined meshes."""
    print("  Part 1: Mesh comparison...")

    h, w = 256, 256
    step = 16
    margin = step

    # Create annular mask
    mask = make_annular_mask(h, w, cx=128, cy=128, r_outer=105, r_inner=35)

    # Build parameters and uniform mesh
    para = dicpara_default(
        img_size=(h, w), winsize=32, winstepsize=step, winsize_min=4,
        img_ref_mask=mask,
    )
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    uniform_mesh = mesh_setup(xs, ys, para)
    n_nodes_uniform = uniform_mesh.coordinates_fem.shape[0]
    U0 = np.zeros(2 * n_nodes_uniform, dtype=np.float64)

    # --- (a) Uniform mesh ---
    # --- (b) Geometry-based (MaskBoundary) ---
    mask_crit = MaskBoundaryCriterion(min_element_size=4)
    ctx_geo = RefinementContext(mesh=uniform_mesh, mask=mask)
    mesh_geo, _ = refine_mesh(
        uniform_mesh, [mask_crit], ctx_geo, U0,
        mask=mask, img_size=(h, w),
    )

    # --- (c) Manual selection (user-defined region) ---
    # Select elements in the top-right quadrant
    centroids_x = []
    centroids_y = []
    elems = uniform_mesh.elements_fem[:, :4]
    coords = uniform_mesh.coordinates_fem
    for i in range(elems.shape[0]):
        pts = coords[elems[i]]
        centroids_x.append(pts[:, 0].mean())
        centroids_y.append(pts[:, 1].mean())
    centroids_x = np.array(centroids_x)
    centroids_y = np.array(centroids_y)

    # Select top-right corner region
    manual_idx = np.where((centroids_x > 160) & (centroids_y < 96))[0]
    manual_crit = ManualSelectionCriterion(
        element_indices=manual_idx, min_element_size=4,
    )
    ctx_manual = RefinementContext(mesh=uniform_mesh)
    mesh_manual, _ = refine_mesh(
        uniform_mesh, [manual_crit], ctx_manual, U0,
    )

    # --- (d) Combined: geometry + manual ---
    ctx_combo1 = RefinementContext(mesh=uniform_mesh, mask=mask)
    mesh_combo1, _ = refine_mesh(
        uniform_mesh, [mask_crit, manual_crit], ctx_combo1, U0,
        mask=mask, img_size=(h, w),
    )

    # --- Plot Page 1: 2x2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(
        "Part 1: Adaptive Mesh Refinement — Criterion Comparison",
        fontsize=14, fontweight="bold", y=0.98,
    )

    configs = [
        (axes[0, 0], uniform_mesh, "(a) Uniform Mesh", "steelblue", None),
        (axes[0, 1], mesh_geo, "(b) Geometry-Based\n(MaskBoundaryCriterion)", "forestgreen", mask),
        (axes[1, 0], mesh_manual, "(c) Manual Selection\n(top-right quadrant)", "darkorange", None),
        (axes[1, 1], mesh_combo1, "(d) Combined:\nGeometry + Manual", "purple", mask),
    ]

    for ax, mesh_obj, title, color, m in configs:
        plot_mesh(ax, mesh_obj, title=title, color=color, mask=m)
        stats = mesh_stats_text(mesh_obj)
        ax.text(
            0.02, 0.02, stats, transform=ax.transAxes,
            fontsize=7, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Mark visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(
        "Part 1b: Element Marking Visualization — Which Elements Get Refined?",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Show which elements are marked by each criterion
    marks_geo = mask_crit.mark(RefinementContext(mesh=uniform_mesh, mask=mask))
    marks_manual = manual_crit.mark(RefinementContext(mesh=uniform_mesh))
    marks_combo1 = marks_geo | marks_manual

    mark_configs = [
        (axes[0, 0], uniform_mesh, "No Marks\n(Uniform)", None, "steelblue"),
        (axes[0, 1], uniform_mesh, "Geometry Marks\n(boundary elements)", np.where(marks_geo)[0], "forestgreen"),
        (axes[1, 0], uniform_mesh, "Manual Marks\n(user selection)", np.where(marks_manual)[0], "darkorange"),
        (axes[1, 1], uniform_mesh, "Union: Geo + Manual", np.where(marks_combo1)[0], "purple"),
    ]

    for ax, mesh_obj, title, highlights, color in mark_configs:
        plot_mesh(
            ax, mesh_obj, title=title, color="gray", mask=mask,
            highlight_elements=highlights, highlight_color=color,
        )
        n_marked = 0 if highlights is None else len(highlights)
        ax.text(
            0.02, 0.02, f"Marked: {n_marked}/{uniform_mesh.elements_fem.shape[0]} elements",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================
# Part 2: Per-frame Independent Mesh (Incremental Mode)
# ============================================================================

def part2_per_frame_mesh(pdf: PdfPages) -> None:
    """Visualize per-frame independent mesh in incremental tracking mode."""
    print("  Part 2: Per-frame independent mesh (incremental mode)...")

    h, w = 256, 256
    step = 16

    # Create a mask with a hole that conceptually could shift between frames
    mask = make_annular_mask(h, w, cx=128, cy=128, r_outer=100, r_inner=30)

    # Generate 3-frame synthetic images (zero displacement for simplicity)
    ref = generate_speckle(h, w, seed=1) / 255.0
    f2 = generate_speckle(h, w, seed=2) / 255.0
    f3 = generate_speckle(h, w, seed=3) / 255.0

    margin = step
    para = dicpara_default(
        winsize=32, winstepsize=step, winsize_min=4,
        reference_mode="incremental",
        gridxy_roi_range=GridxyROIRange(
            gridx=(margin, w - margin), gridy=(margin, h - margin)
        ),
    )

    policy = RefinementPolicy(
        pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
    )

    t_start = time.perf_counter()
    result_with = run_aldic(
        para, [ref, f2, f3], [mask, mask, mask],
        compute_strain=False, refinement_policy=policy,
    )
    t_with = time.perf_counter() - t_start

    t_start = time.perf_counter()
    result_without = run_aldic(
        para, [ref, f2, f3], [mask, mask, mask],
        compute_strain=False, refinement_policy=None,
    )
    t_without = time.perf_counter() - t_start

    # Plot comparison: 2 rows x 2 cols
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Part 2: Per-Frame Independent Mesh (Incremental Tracking Mode)\n"
        f"3 frames, mask=annular ring, step={step}, winsize_min=4",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Without refinement — both frames share same uniform mesh
    for i, (ax, label) in enumerate(zip(
        [axes[0, 0], axes[0, 1]], ["Frame 2 (no policy)", "Frame 3 (no policy)"]
    )):
        m = result_without.result_fe_mesh_each_frame[i]
        plot_mesh(ax, m, title=label, color="steelblue", mask=mask)
        ax.text(
            0.02, 0.02, mesh_stats_text(m), transform=ax.transAxes,
            fontsize=7, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # With refinement — each frame gets its own mesh
    for i, (ax, label) in enumerate(zip(
        [axes[1, 0], axes[1, 1]],
        ["Frame 2 (pre-solve refinement)", "Frame 3 (pre-solve refinement)"],
    )):
        m = result_with.result_fe_mesh_each_frame[i]
        plot_mesh(ax, m, title=label, color="forestgreen", mask=mask)
        ax.text(
            0.02, 0.02, mesh_stats_text(m), transform=ax.transAxes,
            fontsize=7, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add timing comparison
    fig.text(
        0.5, 0.01,
        f"Pipeline time:  Without refinement: {t_without:.2f}s  |  "
        f"With pre-solve refinement: {t_with:.2f}s  |  "
        f"Overhead: {t_with - t_without:.2f}s ({(t_with/t_without - 1)*100:.0f}%)",
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================
# Part 3: Per-Criterion Independent min_element_size
# ============================================================================

def part3_min_element_size(pdf: PdfPages) -> None:
    """Visualize independent min_element_size per criterion."""
    print("  Part 3: Per-criterion min_element_size...")

    h, w = 256, 256
    step = 16
    margin = step

    mask = make_annular_mask(h, w, cx=128, cy=128, r_outer=100, r_inner=30)

    para = dicpara_default(
        img_size=(h, w), winsize=32, winstepsize=step, winsize_min=2,
        img_ref_mask=mask,
    )
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    uniform_mesh = mesh_setup(xs, ys, para)
    n_nodes = uniform_mesh.coordinates_fem.shape[0]
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)

    # Manual selection: bottom-left elements
    coords = uniform_mesh.coordinates_fem
    elems = uniform_mesh.elements_fem[:, :4]
    cx_arr = np.array([coords[elems[i], 0].mean() for i in range(elems.shape[0])])
    cy_arr = np.array([coords[elems[i], 1].mean() for i in range(elems.shape[0])])
    bl_idx = np.where((cx_arr < 100) & (cy_arr > 160))[0]

    # Test different min_element_size configurations
    configs = [
        ("(a) Both min_size=8", 8, 8),
        ("(b) Both min_size=4", 4, 4),
        ("(c) Both min_size=2", 2, 2),
        ("(d) Geometry=8, Manual=4", 8, 4),
        ("(e) Geometry=4, Manual=8", 4, 8),
        ("(f) Geometry=2, Manual=8", 2, 8),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Part 3: Per-Criterion Independent min_element_size\n"
        "Geometry (annular mask) + Manual (bottom-left region) — different stopping depths",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ax, (title, geo_min, manual_min) in zip(axes.flat, configs):
        geo_crit = MaskBoundaryCriterion(min_element_size=geo_min)
        manual_crit = ManualSelectionCriterion(
            element_indices=bl_idx, min_element_size=manual_min,
        )
        ctx = RefinementContext(mesh=uniform_mesh, mask=mask)
        refined, _ = refine_mesh(
            uniform_mesh, [geo_crit, manual_crit], ctx, U0,
            mask=mask, img_size=(h, w),
        )
        plot_mesh(ax, refined, title=title, color="teal", mask=mask)
        stats = mesh_stats_text(refined)
        ax.text(
            0.02, 0.02,
            f"geo_min={geo_min}, manual_min={manual_min}\n{stats}",
            transform=ax.transAxes, fontsize=7, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================
# Part 4: Feasibility, Accuracy, and Efficiency
# ============================================================================

def part4_benchmarks(pdf: PdfPages) -> None:
    """Benchmark feasibility, accuracy, and efficiency of each configuration."""
    print("  Part 4: Feasibility, accuracy, efficiency benchmarks...")

    h, w = 256, 256
    step = 16
    margin = step

    # Known displacement: u(x,y) = 3.5, v(x,y) = -2.0 (uniform translation)
    u_gt = 3.5
    v_gt = -2.0

    ref = generate_speckle(h, w, seed=42) / 255.0
    defm = apply_displacement_lagrangian(
        ref * 255.0, lambda x, y: u_gt * np.ones_like(x), lambda x, y: v_gt * np.ones_like(x),
    ) / 255.0

    # Annular mask
    mask = make_annular_mask(h, w, cx=128, cy=128, r_outer=100, r_inner=30)

    configs_to_test = {
        "Uniform\n(no refinement)": None,
        "Geometry\n(MaskBoundary\nmin=8)": RefinementPolicy(
            pre_solve=[MaskBoundaryCriterion(min_element_size=8)],
        ),
        "Geometry\n(MaskBoundary\nmin=4)": RefinementPolicy(
            pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
        ),
        "Manual\n(center region)": "manual",  # special handling
    }

    results = {}
    for label, policy in configs_to_test.items():
        print(f"    Running: {label.replace(chr(10), ' ')}...")

        # Handle manual selection specially (need mesh to pick elements)
        if policy == "manual":
            # Create a ManualSelectionCriterion for center elements
            para_tmp = dicpara_default(
                winsize=32, winstepsize=step, winsize_min=4,
                gridxy_roi_range=GridxyROIRange(
                    gridx=(margin, w - margin), gridy=(margin, h - margin)
                ),
            )
            xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
            ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
            tmp_mesh = mesh_setup(xs, ys, para_tmp)
            elems = tmp_mesh.elements_fem[:, :4]
            coords = tmp_mesh.coordinates_fem
            cx_arr = np.array([coords[elems[i], 0].mean() for i in range(elems.shape[0])])
            cy_arr = np.array([coords[elems[i], 1].mean() for i in range(elems.shape[0])])
            center_idx = np.where(
                (cx_arr > 80) & (cx_arr < 176) & (cy_arr > 80) & (cy_arr < 176)
            )[0]
            policy = RefinementPolicy(
                pre_solve=[ManualSelectionCriterion(
                    element_indices=center_idx, min_element_size=4,
                )],
            )

        para = dicpara_default(
            winsize=32, winstepsize=step, winsize_min=4,
            gridxy_roi_range=GridxyROIRange(
                gridx=(margin, w - margin), gridy=(margin, h - margin)
            ),
        )

        t_start = time.perf_counter()
        result = run_aldic(
            para, [ref, defm], [mask, mask],
            compute_strain=False, refinement_policy=policy,
        )
        elapsed = time.perf_counter() - t_start

        # Extract metrics
        mesh_final = result.result_fe_mesh_each_frame[0]
        disp = result.result_disp[0]
        n_nodes = mesh_final.coordinates_fem.shape[0]
        n_elem = mesh_final.elements_fem.shape[0]

        # Compute RMSE on interior nodes (skip boundary)
        U = disp.U
        u_vals = U[0::2]
        v_vals = U[1::2]
        valid = ~np.isnan(u_vals)

        # Also check mask membership
        node_x = np.clip(
            np.round(mesh_final.coordinates_fem[:, 0]).astype(int), 0, w - 1
        )
        node_y = np.clip(
            np.round(mesh_final.coordinates_fem[:, 1]).astype(int), 0, h - 1
        )
        in_mask = mask[node_y, node_x] > 0.5
        interior = valid & in_mask

        if interior.sum() > 0:
            rmse_u = np.sqrt(np.mean((u_vals[interior] - u_gt) ** 2))
            rmse_v = np.sqrt(np.mean((v_vals[interior] - v_gt) ** 2))
        else:
            rmse_u = rmse_v = np.nan

        results[label] = {
            "n_nodes": n_nodes,
            "n_elem": n_elem,
            "time": elapsed,
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
            "rmse_total": np.sqrt(rmse_u**2 + rmse_v**2),
            "mesh": mesh_final,
        }

    # --- Plot Page: Bar charts ---
    labels = list(results.keys())
    n_configs = len(labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Part 4: Feasibility, Accuracy & Efficiency Benchmarks\n"
        f"256x256 annular mask, translation u={u_gt}px v={v_gt}px, step={step}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    colors = ["steelblue", "forestgreen", "limegreen", "darkorange", "crimson", "teal"]

    # 4a: Node/element count
    ax = axes[0, 0]
    x = np.arange(n_configs)
    bar_w = 0.35
    nodes = [results[l]["n_nodes"] for l in labels]
    elems = [results[l]["n_elem"] for l in labels]
    ax.bar(x - bar_w / 2, nodes, bar_w, label="Nodes", color=colors, alpha=0.7)
    ax.bar(x + bar_w / 2, elems, bar_w, label="Elements", color=colors, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("Count")
    ax.set_title("(a) Mesh Size", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 4b: Pipeline time
    ax = axes[0, 1]
    times = [results[l]["time"] for l in labels]
    bars = ax.bar(x, times, color=colors, alpha=0.7)
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{t:.2f}s", ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("(b) Pipeline Execution Time", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 4c: Displacement RMSE
    ax = axes[1, 0]
    rmse_u = [results[l]["rmse_u"] for l in labels]
    rmse_v = [results[l]["rmse_v"] for l in labels]
    ax.bar(x - bar_w / 2, rmse_u, bar_w, label="RMSE u", color=colors, alpha=0.7)
    ax.bar(x + bar_w / 2, rmse_v, bar_w, label="RMSE v", color=colors, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("RMSE (pixels)")
    ax.set_title("(c) Displacement Accuracy", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Add value labels
    for i, (ru, rv) in enumerate(zip(rmse_u, rmse_v)):
        if not np.isnan(ru):
            ax.text(i - bar_w/2, ru + 0.001, f"{ru:.4f}", ha="center", fontsize=6, rotation=45)
        if not np.isnan(rv):
            ax.text(i + bar_w/2, rv + 0.001, f"{rv:.4f}", ha="center", fontsize=6, rotation=45)

    # 4d: Efficiency (accuracy per second)
    ax = axes[1, 1]
    # Compute nodes per second and RMSE (lower is better)
    nodes_per_sec = [results[l]["n_nodes"] / results[l]["time"] for l in labels]
    total_rmse = [results[l]["rmse_total"] for l in labels]

    ax2 = ax.twinx()
    bars1 = ax.bar(x - bar_w / 2, nodes_per_sec, bar_w, label="Nodes/sec", color=colors, alpha=0.7)
    bars2 = ax2.bar(x + bar_w / 2, total_rmse, bar_w, label="Total RMSE", color=colors, alpha=0.3, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("Nodes / second", color="steelblue")
    ax2.set_ylabel("Total RMSE (px)", color="crimson")
    ax.set_title("(d) Efficiency: Throughput vs Accuracy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Combined legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=6, alpha=0.7, label="Nodes/sec (higher=faster)"),
        Line2D([0], [0], color="crimson", lw=6, alpha=0.3, label="RMSE (lower=better)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Summary table ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    fig.suptitle(
        "Part 4b: Detailed Results Table",
        fontsize=14, fontweight="bold", y=0.95,
    )

    col_labels = [
        "Configuration", "Nodes", "Elements",
        "Time (s)", "RMSE u (px)", "RMSE v (px)",
        "Total RMSE", "Status",
    ]
    table_data = []
    for label in labels:
        r = results[label]
        status = "PASS" if r["rmse_total"] < 0.1 else "CHECK"
        table_data.append([
            label.replace("\n", " "),
            str(r["n_nodes"]),
            str(r["n_elem"]),
            f"{r['time']:.3f}",
            f"{r['rmse_u']:.5f}" if not np.isnan(r["rmse_u"]) else "N/A",
            f"{r['rmse_v']:.5f}" if not np.isnan(r["rmse_v"]) else "N/A",
            f"{r['rmse_total']:.5f}" if not np.isnan(r["rmse_total"]) else "N/A",
            status,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color status cells
    last_col = len(col_labels) - 1
    for i, row in enumerate(table_data):
        if row[-1] == "PASS":
            table[i + 1, last_col].set_facecolor("#C6EFCE")
        else:
            table[i + 1, last_col].set_facecolor("#FFC7CE")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(exist_ok=True)
    pdf_path = output_dir / "adaptive_refinement_report.pdf"

    print(f"Generating adaptive refinement report -> {pdf_path}")
    print("=" * 60)

    with PdfPages(str(pdf_path)) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")
        ax.text(
            0.5, 0.65,
            "Adaptive Mesh Refinement Framework\nVisualization Report",
            ha="center", va="center", fontsize=24, fontweight="bold",
        )
        ax.text(
            0.5, 0.45,
            "STAQ-DIC Python Port\n"
            "Protocol-Based Strategy Pattern for Extensible Mesh Refinement",
            ha="center", va="center", fontsize=14, color="gray",
        )
        ax.text(
            0.5, 0.25,
            "Part 1: Criterion Comparison (Uniform / Geometry / Manual / Post-Solve / Combined)\n"
            "Part 2: Per-Frame Independent Mesh (Incremental Tracking Mode)\n"
            "Part 3: Per-Criterion Independent min_element_size\n"
            "Part 4: Feasibility, Accuracy & Efficiency Benchmarks",
            ha="center", va="center", fontsize=11,
        )
        ax.text(
            0.5, 0.08,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
            ha="center", va="center", fontsize=10, color="gray",
        )
        pdf.savefig(fig)
        plt.close(fig)

        part1_mesh_comparison(pdf)
        part2_per_frame_mesh(pdf)
        part3_min_element_size(pdf)
        part4_benchmarks(pdf)

    print("=" * 60)
    print(f"Report saved to: {pdf_path}")
    print("Done!")


if __name__ == "__main__":
    main()
