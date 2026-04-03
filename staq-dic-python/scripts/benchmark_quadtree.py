#!/usr/bin/env python
"""Benchmark quadtree DIC: accuracy + speed across masks, refinement levels, deformations.

Evaluation matrix:
    3 Masks × 4 Refinement Levels × 4 Deformations = 48 DIC runs

Masks:
    - Annular (r_in=60, r_out=90)
    - Double holes
    - Rectangular notch

Refinement (winstepsize=16 fixed):
    - Uniform:  winsize_min=16  (no refinement)
    - 1-level:  winsize_min=8   (16->8)
    - 2-level:  winsize_min=4   (16->8->4)
    - 3-level:  winsize_min=2   (16->8->4->2)

Deformations:
    - Translation (2.5, -1.5)
    - Expansion (2%)
    - Pure shear (1.5%)
    - Rotation (2 deg)

Output:
    fig1_meshes.png        — 3x4 mesh overview (mask x refinement)
    fig2_rmse_bars.png     — RMSE bar chart per mask
    fig3_timing_bars.png   — Timing bar chart per mask
    fig4_error_fields.png  — Error fields for one representative case per mask
    Console: full 48-row table

Usage:
    python scripts/benchmark_quadtree.py [--output-dir reports/benchmark]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ / "src") not in sys.path:
    sys.path.insert(0, str(_PROJ / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.core.pipeline import run_aldic
from staq_dic.io.image_ops import compute_image_gradient
from staq_dic.mesh.generate_mesh import generate_mesh
from staq_dic.mesh.mesh_setup import mesh_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0
WINSTEPSIZE = 16
MARGIN = WINSTEPSIZE


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    mask_name: str
    refine_name: str
    winsize_min: int
    deform_name: str
    n_nodes: int
    n_elements: int
    rmse_u: float
    rmse_v: float
    time_s: float


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement_lagrangian(ref, u_func, v_func, n_iter=20):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    coords = np.array([Y.ravel(), X.ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest").reshape(h, w)


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------
def make_annular_mask(r_outer=90.0, r_inner=60.0):
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    dist2 = (xx - CX) ** 2 + (yy - CY) ** 2
    return ((dist2 <= r_outer**2) & (dist2 > r_inner**2)).astype(np.float64)


def make_double_hole_mask():
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    outer = ((xx - CX) ** 2 + (yy - CY) ** 2) <= 110.0**2
    hole1 = ((xx - 90) ** 2 + (yy - CY) ** 2) <= 25.0**2
    hole2 = ((xx - 164) ** 2 + (yy - CY) ** 2) <= 25.0**2
    return (outer & ~hole1 & ~hole2).astype(np.float64)


def make_rectangular_notch_mask():
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
    mask[120:136, 0:100] = 0.0
    mask[:10, :] = 0.0
    mask[-10:, :] = 0.0
    mask[:, :10] = 0.0
    mask[:, -10:] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Deformation definitions
# ---------------------------------------------------------------------------
DEFORMATIONS = {
    "Translation\n(2.5, -1.5)": (
        lambda x, y: np.full_like(x, 2.5),
        lambda x, y: np.full_like(x, -1.5),
    ),
    "Expansion\n(2%)": (
        lambda x, y: 0.02 * (x - CX),
        lambda x, y: 0.02 * (y - CY),
    ),
    "Pure shear\n(1.5%)": (
        lambda x, y: 0.015 * (y - CY),
        lambda x, y: np.zeros_like(x),
    ),
    "Rotation\n(2 deg)": (
        lambda x, y: (x - CX) * (np.cos(np.pi / 90) - 1) - (y - CY) * np.sin(np.pi / 90),
        lambda x, y: (x - CX) * np.sin(np.pi / 90) + (y - CY) * (np.cos(np.pi / 90) - 1),
    ),
}

MASKS = {
    "Annular\n(r_in=60)": make_annular_mask(r_outer=90.0, r_inner=60.0),
    "Double\nholes": make_double_hole_mask(),
    "Rectangular\nnotch": make_rectangular_notch_mask(),
}

REFINEMENTS = [
    ("Uniform\n(min=16)", 16),
    ("1-level\n(min=8)", 8),
    ("2-level\n(min=4)", 4),
    ("3-level\n(min=2)", 2),
]


# ---------------------------------------------------------------------------
# RMSE (mask-interior only)
# ---------------------------------------------------------------------------
def compute_rmse(U, coords, gt_u, gt_v, mask):
    u_comp, v_comp = U[0::2], U[1::2]
    h, w = mask.shape
    cx = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
    in_mask = mask[cy, cx] > 0.5
    valid = in_mask & np.isfinite(u_comp) & np.isfinite(v_comp)
    if not np.any(valid):
        return np.inf, np.inf
    err_u = u_comp[valid] - gt_u[valid]
    err_v = v_comp[valid] - gt_v[valid]
    return float(np.sqrt(np.mean(err_u**2))), float(np.sqrt(np.mean(err_v**2)))


# ---------------------------------------------------------------------------
# Core: build mesh + run DIC
# ---------------------------------------------------------------------------
def build_mesh(ref_speckle, mask, winsize_min):
    """Build mesh with given refinement level. Returns (mesh, para)."""
    para = dicpara_default(
        winsize=32,
        winstepsize=WINSTEPSIZE,
        winsize_min=winsize_min,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative",
        admm_max_iter=3,
        admm_tol=1e-2,
        show_plots=False,
        icgn_max_iter=50,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        alpha=0.0,
        img_ref_mask=mask,
    )

    xs = np.arange(MARGIN, IMG_W - MARGIN + 1, WINSTEPSIZE, dtype=np.float64)
    ys = np.arange(MARGIN, IMG_H - MARGIN + 1, WINSTEPSIZE, dtype=np.float64)
    uniform_mesh = mesh_setup(xs, ys, para)

    if winsize_min >= WINSTEPSIZE:
        # No refinement — return uniform mesh directly
        return uniform_mesh, para

    f_img = ref_speckle / ref_speckle.max()
    Df = compute_image_gradient(f_img, mask)
    n_uni = uniform_mesh.coordinates_fem.shape[0]
    U0 = np.zeros(2 * n_uni, dtype=np.float64)
    mesh_qt, _ = generate_mesh(uniform_mesh, para, Df, U0)
    return mesh_qt, para


def run_single(ref_speckle, mask, mesh, para, u_func, v_func):
    """Run DIC and return (U, coords, time_s)."""
    deformed = apply_displacement_lagrangian(ref_speckle, u_func, v_func)
    n_nodes = mesh.coordinates_fem.shape[0]
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)

    t0 = time.perf_counter()
    result = run_aldic(
        para,
        [ref_speckle, deformed],
        [mask, mask],
        mesh=mesh,
        U0=U0,
        compute_strain=False,
    )
    elapsed = time.perf_counter() - t0

    fr = result.result_disp[0]
    U = fr.U_accum if fr.U_accum is not None else fr.U
    coords = result.dic_mesh.coordinates_fem
    return U, coords, elapsed


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_mesh_on_mask(ax, mesh, mask, title=""):
    """Plot mesh elements on mask background."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.3, extent=[0, IMG_W, IMG_H, 0])

    patches = []
    for i in range(elems.shape[0]):
        verts = coords[elems[i, :4]]
        patches.append(MplPolygon(verts, closed=True))
    pc = PatchCollection(patches, facecolor="none", edgecolor="steelblue", linewidth=0.4)
    ax.add_collection(pc)
    ax.plot(coords[:, 0], coords[:, 1], "k.", markersize=0.5, alpha=0.4)

    # Hanging nodes
    midside_idx = elems[:, 4:8]
    hanging = np.unique(midside_idx[midside_idx >= 0])
    if len(hanging) > 0:
        ax.plot(coords[hanging, 0], coords[hanging, 1], "r.", markersize=1.5)

    n_n, n_e = coords.shape[0], elems.shape[0]
    ax.set_title(f"{title}\n{n_n} nodes, {n_e} elem", fontsize=8)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def plot_error_field(ax, coords, values, mask, title="", cmap="RdBu_r",
                     vmin=None, vmax=None):
    """Plot error field with mask overlay."""
    h, w = mask.shape
    ix = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
    iy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
    in_mask = mask[iy, ix] > 0.5
    valid = np.isfinite(values) & in_mask

    if valid.sum() < 3:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontsize=7)
        return

    tri = Triangulation(coords[valid, 0], coords[valid, 1])
    if vmin is None:
        vmin = np.nanpercentile(values[valid], 2)
    if vmax is None:
        vmax = np.nanpercentile(values[valid], 98)
    if abs(vmax - vmin) < 1e-10:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    levels = np.linspace(vmin, vmax, 21)
    tcf = ax.tricontourf(tri, values[valid], levels=levels, cmap=cmap, extend="both")

    # Gray out mask exterior
    overlay = np.ones((h, w, 4), dtype=np.float32)
    overlay[..., :3] = 0.85
    overlay[..., 3] = 0.0
    overlay[mask < 0.5, 3] = 0.9
    ax.imshow(overlay, origin="upper", extent=[0, IMG_W, IMG_H, 0],
              interpolation="nearest", zorder=2)
    ax.contour(mask, levels=[0.5], colors="k", linewidths=0.6,
               extent=[0, IMG_W, IMG_H, 0], origin="upper", zorder=3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.03)
    plt.colorbar(tcf, cax=cax)
    cax.tick_params(labelsize=5)

    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Figure 1: Mesh overview (3 masks x 4 refinements)
# ---------------------------------------------------------------------------
def plot_fig1_meshes(ref_speckle, meshes, output_dir):
    n_masks = len(MASKS)
    n_refs = len(REFINEMENTS)
    fig, axes = plt.subplots(n_masks, n_refs, figsize=(n_refs * 4, n_masks * 3.5),
                             gridspec_kw={"top": 0.90})
    fig.suptitle("Mesh Overview: Mask Shape x Refinement Level", fontsize=13, y=0.96)

    # Column headers
    for c, (ref_name, _) in enumerate(REFINEMENTS):
        bbox = axes[0, c].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.92, ref_name,
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    for r, mask_name in enumerate(MASKS):
        mask = MASKS[mask_name]
        for c, (ref_name, wmin) in enumerate(REFINEMENTS):
            mesh = meshes[(mask_name, wmin)]
            plot_mesh_on_mask(axes[r, c], mesh, mask)
            if c == 0:
                axes[r, c].set_ylabel(mask_name, fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.03, top=0.90,
                        hspace=0.30, wspace=0.20)
    fig.savefig(output_dir / "fig1_meshes.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_meshes.png")


# ---------------------------------------------------------------------------
# Figure 2: RMSE bar chart
# ---------------------------------------------------------------------------
def plot_fig2_rmse(results: list[BenchResult], output_dir):
    mask_names = list(MASKS.keys())
    deform_names = list(DEFORMATIONS.keys())
    ref_names = [r[0] for r in REFINEMENTS]
    n_refs = len(ref_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"top": 0.85})
    fig.suptitle("RMSE Comparison: Refinement Level x Deformation", fontsize=13, y=0.95)

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    bar_w = 0.18

    for mi, mask_name in enumerate(mask_names):
        ax = axes[mi]
        x = np.arange(len(deform_names))

        for ri, ref_name in enumerate(ref_names):
            rmse_total = []
            for di, deform_name in enumerate(deform_names):
                # Find matching result
                match = [r for r in results
                         if r.mask_name == mask_name
                         and r.refine_name == ref_name
                         and r.deform_name == deform_name]
                if match:
                    r = match[0]
                    rmse_total.append(np.sqrt(r.rmse_u**2 + r.rmse_v**2))
                else:
                    rmse_total.append(0)

            offset = (ri - (n_refs - 1) / 2) * bar_w
            bars = ax.bar(x + offset, rmse_total, bar_w, label=ref_name.replace("\n", " "),
                          color=colors[ri], edgecolor="white", linewidth=0.5)
            # Value labels on bars
            for bar, val in zip(bars, rmse_total):
                if val > 0 and val < 2:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=5.5,
                            rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("\n", " ") for d in deform_names], fontsize=8)
        ax.set_ylabel("RMSE_total (px)", fontsize=9)
        ax.set_title(mask_name.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.85,
                        wspace=0.25)
    fig.savefig(output_dir / "fig2_rmse_bars.png", dpi=150)
    plt.close(fig)
    print("  Saved fig2_rmse_bars.png")


# ---------------------------------------------------------------------------
# Figure 3: Timing bar chart
# ---------------------------------------------------------------------------
def plot_fig3_timing(results: list[BenchResult], output_dir):
    mask_names = list(MASKS.keys())
    deform_names = list(DEFORMATIONS.keys())
    ref_names = [r[0] for r in REFINEMENTS]
    n_refs = len(ref_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"top": 0.85})
    fig.suptitle("Computation Time: Refinement Level x Deformation", fontsize=13, y=0.95)

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    bar_w = 0.18

    for mi, mask_name in enumerate(mask_names):
        ax = axes[mi]
        x = np.arange(len(deform_names))

        for ri, ref_name in enumerate(ref_names):
            times = []
            for deform_name in deform_names:
                match = [r for r in results
                         if r.mask_name == mask_name
                         and r.refine_name == ref_name
                         and r.deform_name == deform_name]
                times.append(match[0].time_s if match else 0)

            offset = (ri - (n_refs - 1) / 2) * bar_w
            bars = ax.bar(x + offset, times, bar_w, label=ref_name.replace("\n", " "),
                          color=colors[ri], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, times):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{val:.2f}s", ha="center", va="bottom", fontsize=5.5,
                            rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("\n", " ") for d in deform_names], fontsize=8)
        ax.set_ylabel("Time (s)", fontsize=9)
        ax.set_title(mask_name.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.85,
                        wspace=0.25)
    fig.savefig(output_dir / "fig3_timing_bars.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_timing_bars.png")


# ---------------------------------------------------------------------------
# Figure 4: Error fields for Translation case (representative)
# ---------------------------------------------------------------------------
def plot_fig4_error_fields(ref_speckle, meshes, output_dir):
    """Show error fields for Translation across all mask x refinement combinations."""
    deform_name = "Translation\n(2.5, -1.5)"
    u_func, v_func = DEFORMATIONS[deform_name]

    mask_names = list(MASKS.keys())
    ref_entries = REFINEMENTS
    n_masks = len(mask_names)
    n_refs = len(ref_entries)

    # 2 error components (u, v) side by side for each refinement
    n_cols = n_refs * 2
    fig, axes = plt.subplots(n_masks, n_cols, figsize=(n_cols * 2.8, n_masks * 3.0 + 1.2),
                             gridspec_kw={"top": 0.88})
    fig.suptitle(
        "Error Fields: Translation (2.5, -1.5) — All Masks x Refinements",
        fontsize=13, y=0.96,
    )

    # Column group headers
    for ri, (ref_name, _) in enumerate(ref_entries):
        col_start = ri * 2
        bbox0 = axes[0, col_start].get_position()
        bbox1 = axes[0, col_start + 1].get_position()
        fig.text((bbox0.x0 + bbox1.x1) / 2, 0.905,
                 ref_name.replace("\n", " "), ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    for mi, mask_name in enumerate(mask_names):
        mask = MASKS[mask_name]
        for ri, (ref_name, wmin) in enumerate(ref_entries):
            mesh = meshes[(mask_name, wmin)]
            para = build_mesh.__para_cache__[(mask_name, wmin)]

            U, coords, _ = run_single(ref_speckle, mask, mesh, para, u_func, v_func)

            gt_u = u_func(coords[:, 0], coords[:, 1])
            gt_v = v_func(coords[:, 0], coords[:, 1])
            eu = U[0::2] - gt_u
            ev = U[1::2] - gt_v

            # Shared error scale
            h, w = mask.shape
            ix = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
            ok = (mask[iy, ix] > 0.5) & np.isfinite(eu) & np.isfinite(ev)
            all_err = np.concatenate([eu[ok], ev[ok]])
            err_lim = max(np.nanpercentile(np.abs(all_err), 95), 0.01) if len(all_err) > 0 else 0.01

            col_u = ri * 2
            col_v = ri * 2 + 1
            rmse_u = float(np.sqrt(np.mean(eu[ok]**2))) if ok.any() else np.inf
            rmse_v = float(np.sqrt(np.mean(ev[ok]**2))) if ok.any() else np.inf

            plot_error_field(axes[mi, col_u], coords, eu, mask,
                             f"err_u  RMSE={rmse_u:.4f}",
                             vmin=-err_lim, vmax=err_lim)
            plot_error_field(axes[mi, col_v], coords, ev, mask,
                             f"err_v  RMSE={rmse_v:.4f}",
                             vmin=-err_lim, vmax=err_lim)

        axes[mi, 0].set_ylabel(mask_name.replace("\n", " "), fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.03, top=0.88,
                        hspace=0.35, wspace=0.35)
    fig.savefig(output_dir / "fig4_error_fields.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_error_fields.png")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
def print_table(results: list[BenchResult]):
    hdr = (f"{'Mask':<16s} {'Refinement':<14s} {'Deformation':<22s} "
           f"{'Nodes':>6s} {'Elems':>6s} {'RMSE_u':>8s} {'RMSE_v':>8s} "
           f"{'RMSE_tot':>9s} {'Time(s)':>8s}")
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in results:
        rmse_tot = np.sqrt(r.rmse_u**2 + r.rmse_v**2)
        mask_short = r.mask_name.replace("\n", " ")
        ref_short = r.refine_name.replace("\n", " ")
        deform_short = r.deform_name.replace("\n", " ")
        print(f"{mask_short:<16s} {ref_short:<14s} {deform_short:<22s} "
              f"{r.n_nodes:>6d} {r.n_elements:>6d} {r.rmse_u:>8.5f} {r.rmse_v:>8.5f} "
              f"{rmse_tot:>9.5f} {r.time_s:>8.2f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark quadtree DIC")
    parser.add_argument("--output-dir", type=str, default="reports/benchmark",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reference speckle...")
    ref_speckle = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)

    # ------------------------------------------------------------------
    # Phase 1: Build all meshes (cache for reuse)
    # ------------------------------------------------------------------
    print("\n=== Phase 1: Building meshes ===")
    meshes = {}
    # Attach a para cache to build_mesh for fig4 reuse
    build_mesh.__para_cache__ = {}

    for mask_name, mask in MASKS.items():
        for ref_name, wmin in REFINEMENTS:
            key = (mask_name, wmin)
            mesh, para = build_mesh(ref_speckle, mask, wmin)
            meshes[key] = mesh
            build_mesh.__para_cache__[key] = para
            n, e = mesh.coordinates_fem.shape[0], mesh.elements_fem.shape[0]
            print(f"  {mask_name.replace(chr(10),' '):<18s} {ref_name.replace(chr(10),' '):<14s} "
                  f"-> {n:>5d} nodes, {e:>5d} elements")

    # ------------------------------------------------------------------
    # Phase 2: Run all 48 DIC cases
    # ------------------------------------------------------------------
    print("\n=== Phase 2: Running DIC (48 cases) ===")
    results: list[BenchResult] = []
    total = len(MASKS) * len(REFINEMENTS) * len(DEFORMATIONS)
    count = 0

    for mask_name, mask in MASKS.items():
        for ref_name, wmin in REFINEMENTS:
            mesh = meshes[(mask_name, wmin)]
            para = build_mesh.__para_cache__[(mask_name, wmin)]

            for deform_name, (u_func, v_func) in DEFORMATIONS.items():
                count += 1
                short_mask = mask_name.replace("\n", " ")
                short_ref = ref_name.replace("\n", " ")
                short_def = deform_name.replace("\n", " ")
                print(f"  [{count:>2d}/{total}] {short_mask} | {short_ref} | {short_def}...",
                      end="", flush=True)

                U, coords, elapsed = run_single(ref_speckle, mask, mesh, para, u_func, v_func)

                gt_u = u_func(coords[:, 0], coords[:, 1])
                gt_v = v_func(coords[:, 0], coords[:, 1])
                rmse_u, rmse_v = compute_rmse(U, coords, gt_u, gt_v, mask)

                results.append(BenchResult(
                    mask_name=mask_name,
                    refine_name=ref_name,
                    winsize_min=wmin,
                    deform_name=deform_name,
                    n_nodes=coords.shape[0],
                    n_elements=mesh.elements_fem.shape[0],
                    rmse_u=rmse_u,
                    rmse_v=rmse_v,
                    time_s=elapsed,
                ))
                print(f"  RMSE=({rmse_u:.4f}, {rmse_v:.4f})  {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Phase 3: Generate figures
    # ------------------------------------------------------------------
    print("\n=== Phase 3: Generating figures ===")
    plot_fig1_meshes(ref_speckle, meshes, output_dir)
    plot_fig2_rmse(results, output_dir)
    plot_fig3_timing(results, output_dir)
    plot_fig4_error_fields(ref_speckle, meshes, output_dir)

    # ------------------------------------------------------------------
    # Phase 4: Console summary
    # ------------------------------------------------------------------
    print_table(results)
    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
