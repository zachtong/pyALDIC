#!/usr/bin/env python
"""Visualize quadtree mesh refinement and DIC results on quadtree meshes.

Generates:
    Part 1: Mesh structure comparison — uniform vs quadtree for different
            parameter combinations (winstepsize, winsize_min, mask shapes).
    Part 2: DIC displacement results — affine deformation cases tracked
            on quadtree meshes with various deformation types.

Usage:
    python scripts/visualize_quadtree.py [--output-dir reports/quadtree]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ / "src") not in sys.path:
    sys.path.insert(0, str(_PROJ / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh,
    DICPara,
    GridxyROIRange,
    ImageGradients,
    merge_uv,
)
from al_dic.core.pipeline import run_aldic
from al_dic.io.image_ops import compute_image_gradient
from al_dic.mesh.generate_mesh import generate_mesh
from al_dic.mesh.mesh_setup import mesh_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42):
    from scipy.ndimage import gaussian_filter

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


def make_annular_mask(r_outer=90.0, r_inner=40.0):
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    dist2 = (xx - CX) ** 2 + (yy - CY) ** 2
    return ((dist2 <= r_outer**2) & (dist2 > r_inner**2)).astype(np.float64)


def make_double_hole_mask():
    """Two circular holes side by side."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    outer = ((xx - CX) ** 2 + (yy - CY) ** 2) <= 110.0**2
    hole1 = ((xx - 90) ** 2 + (yy - CY) ** 2) <= 25.0**2
    hole2 = ((xx - 164) ** 2 + (yy - CY) ** 2) <= 25.0**2
    return (outer & ~hole1 & ~hole2).astype(np.float64)


def make_rectangular_notch_mask():
    """Rectangular specimen with a notch from the left."""
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
    # Notch: horizontal slot from left edge
    mask[120:136, 0:100] = 0.0
    # Border
    mask[:10, :] = 0.0
    mask[-10:, :] = 0.0
    mask[:, :10] = 0.0
    mask[:, -10:] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Mesh visualization helpers
# ---------------------------------------------------------------------------


def plot_mesh_on_mask(
    ax,
    mesh: DICMesh,
    mask: NDArray[np.float64],
    title: str = "",
    show_nodes: bool = True,
    show_hanging: bool = True,
):
    """Plot mesh elements overlaid on the mask image."""
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem

    # Show mask as background
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.3, extent=[0, IMG_W, IMG_H, 0])

    # Draw element edges
    patches = []
    for i in range(elems.shape[0]):
        corners = elems[i, :4]
        verts = coords[corners]
        patches.append(MplPolygon(verts, closed=True))

    pc = PatchCollection(
        patches, facecolor="none", edgecolor="steelblue", linewidth=0.5,
    )
    ax.add_collection(pc)

    if show_nodes:
        # Regular nodes
        ax.plot(coords[:, 0], coords[:, 1], "k.", markersize=1, alpha=0.5)

    if show_hanging:
        # Highlight hanging (midside) nodes
        midside_idx = elems[:, 4:8]
        hanging_nodes = np.unique(midside_idx[midside_idx >= 0])
        if len(hanging_nodes) > 0:
            ax.plot(
                coords[hanging_nodes, 0],
                coords[hanging_nodes, 1],
                "r.",
                markersize=3,
                label=f"hanging nodes ({len(hanging_nodes)})",
            )

    # Highlight boundary nodes
    if hasattr(mesh, "mark_coord_hole_edge") and len(mesh.mark_coord_hole_edge) > 0:
        bnd = mesh.mark_coord_hole_edge
        valid = bnd[bnd < len(coords)]
        ax.plot(
            coords[valid, 0],
            coords[valid, 1],
            ".",
            color="orange",
            markersize=2,
            alpha=0.6,
            label=f"boundary nodes ({len(valid)})",
        )

    n_nodes = coords.shape[0]
    n_elems = elems.shape[0]
    ax.set_title(f"{title}\n{n_nodes} nodes, {n_elems} elements", fontsize=9)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    if show_hanging:
        ax.legend(fontsize=6, loc="upper right")


def build_quadtree(
    ref_speckle: NDArray,
    mask: NDArray,
    winstepsize: int = 16,
    winsize_min: int = 8,
    margin: int = 16,
) -> tuple[DICMesh, DICMesh]:
    """Build both uniform and quadtree meshes for comparison."""
    para = dicpara_default(
        winsize=32,
        winstepsize=winstepsize,
        winsize_min=winsize_min,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative",
        show_plots=False,
        img_ref_mask=mask,
    )

    xs = np.arange(margin, IMG_W - margin + 1, winstepsize, dtype=np.float64)
    ys = np.arange(margin, IMG_H - margin + 1, winstepsize, dtype=np.float64)
    uniform_mesh = mesh_setup(xs, ys, para)

    f_img = ref_speckle / ref_speckle.max()
    Df = compute_image_gradient(f_img, mask)

    n_nodes = uniform_mesh.coordinates_fem.shape[0]
    U0 = np.zeros(2 * n_nodes, dtype=np.float64)

    mesh_qt, _ = generate_mesh(uniform_mesh, para, Df, U0)

    return uniform_mesh, mesh_qt


# ---------------------------------------------------------------------------
# Part 1: Mesh structure visualization
# ---------------------------------------------------------------------------


def plot_part1_mesh_comparison(ref_speckle, output_dir: Path):
    """Compare mesh structures for different parameters and mask shapes."""

    # --- Figure 1: Parameter effect (same annular mask, vary winsize_min) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Effect of winsize_min on Quadtree Refinement (Annular Mask, r_inner=40, r_outer=90)",
        fontsize=12,
    )

    mask = make_annular_mask(r_outer=90.0, r_inner=40.0)

    configs = [
        (16, 16, "winstepsize=16, winsize_min=16\n(no refinement possible)"),
        (16, 8, "winstepsize=16, winsize_min=8\n(1 level: 16→8)"),
        (16, 4, "winstepsize=16, winsize_min=4\n(2 levels: 16→8→4)"),
        (32, 16, "winstepsize=32, winsize_min=16\n(1 level: 32→16)"),
        (32, 8, "winstepsize=32, winsize_min=8\n(2 levels: 32→16→8)"),
        (32, 4, "winstepsize=32, winsize_min=4\n(3 levels: 32→16→8→4)"),
    ]

    for idx, (step, wmin, title) in enumerate(configs):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        try:
            _, mesh_qt = build_quadtree(ref_speckle, mask, step, wmin, margin=step)
            plot_mesh_on_mask(ax, mesh_qt, mask, title=title)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_parameter_effect.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_parameter_effect.png")

    # --- Figure 2: Different mask shapes (same parameters) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Quadtree Refinement for Different Mask Shapes (winstepsize=16, winsize_min=8)",
        fontsize=12,
    )

    masks_and_titles = [
        (make_annular_mask(r_outer=90, r_inner=40), "Annular (r_in=40)"),
        (make_annular_mask(r_outer=90, r_inner=20), "Annular (r_in=20)"),
        (make_annular_mask(r_outer=90, r_inner=60), "Annular (r_in=60)"),
        (make_double_hole_mask(), "Double holes"),
        (make_rectangular_notch_mask(), "Rectangular notch"),
        (np.ones((IMG_H, IMG_W), dtype=np.float64), "Full image (no hole)"),
    ]

    for idx, (m, title) in enumerate(masks_and_titles):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        try:
            uniform, mesh_qt = build_quadtree(ref_speckle, m, 16, 8, margin=16)
            plot_mesh_on_mask(ax, mesh_qt, m, title=title)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_mask_shapes.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_mask_shapes.png")

    # --- Figure 3: Uniform vs Quadtree side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Uniform Mesh vs Quadtree Mesh (Annular Mask)", fontsize=12)

    mask = make_annular_mask(r_outer=90, r_inner=40)
    uniform, mesh_qt = build_quadtree(ref_speckle, mask, 16, 8, margin=16)

    plot_mesh_on_mask(axes[0], uniform, mask, title="Uniform Mesh")
    plot_mesh_on_mask(axes[1], mesh_qt, mask, title="Quadtree Mesh")

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_uniform_vs_quadtree.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_uniform_vs_quadtree.png")


# ---------------------------------------------------------------------------
# Part 2: DIC results on quadtree mesh
# ---------------------------------------------------------------------------


def run_dic_on_quadtree(
    ref_speckle: NDArray,
    mask: NDArray,
    u_func,
    v_func,
    winstepsize: int = 16,
    winsize_min: int = 8,
):
    """Run DIC pipeline on a quadtree mesh and return results."""
    from dataclasses import replace

    # Build quadtree mesh
    para = dicpara_default(
        winsize=32,
        winstepsize=winstepsize,
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

    margin = winstepsize
    xs = np.arange(margin, IMG_W - margin + 1, winstepsize, dtype=np.float64)
    ys = np.arange(margin, IMG_H - margin + 1, winstepsize, dtype=np.float64)
    uniform_mesh = mesh_setup(xs, ys, para)

    f_img = ref_speckle / ref_speckle.max()
    Df = compute_image_gradient(f_img, mask)

    # Ground truth U0 on uniform mesh
    node_x = uniform_mesh.coordinates_fem[:, 0]
    node_y = uniform_mesh.coordinates_fem[:, 1]
    gt_u = u_func(node_x, node_y)
    gt_v = v_func(node_x, node_y)
    U0_uniform = merge_uv(gt_u, gt_v)

    # Generate quadtree mesh
    mesh_qt, U0_qt = generate_mesh(uniform_mesh, para, Df, U0_uniform)

    # Generate deformed image
    deformed = apply_displacement_lagrangian(ref_speckle, u_func, v_func)

    # Run pipeline
    result = run_aldic(
        para,
        [ref_speckle, deformed],
        [mask, mask],
        mesh=mesh_qt,
        U0=U0_qt,
        compute_strain=False,
    )

    return result, mesh_qt


def plot_displacement_field(
    ax,
    coords: NDArray,
    values: NDArray,
    title: str,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    mask: NDArray | None = None,
):
    """Plot displacement field using tricontourf with optional mask overlay.

    Args:
        mask: If provided, (H, W) binary mask. Mask contour is drawn and
              regions outside the mask are grayed out.
    """
    x, y = coords[:, 0], coords[:, 1]

    # Filter to mask-interior nodes when mask is provided
    if mask is not None:
        h, w = mask.shape
        ix = np.clip(np.round(x).astype(int), 0, w - 1)
        iy = np.clip(np.round(y).astype(int), 0, h - 1)
        in_mask = mask[iy, ix] > 0.5
        valid = np.isfinite(values) & in_mask
    else:
        valid = np.isfinite(values)

    if valid.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient valid data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title(title, fontsize=9)
        return

    tri = Triangulation(x[valid], y[valid])
    if vmin is None:
        vmin = np.nanpercentile(values[valid], 2)
    if vmax is None:
        vmax = np.nanpercentile(values[valid], 98)
    if abs(vmax - vmin) < 1e-10:
        vmin -= 0.5
        vmax += 0.5

    levels = np.linspace(vmin, vmax, 21)
    tcf = ax.tricontourf(tri, values[valid], levels=levels, cmap=cmap, extend="both")

    # Gray out regions outside mask
    if mask is not None:
        # Create RGBA overlay: gray where mask==0, transparent where mask==1
        overlay = np.ones((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[..., :3] = 0.85  # light gray
        overlay[..., 3] = 0.0    # fully transparent by default
        overlay[mask < 0.5, 3] = 0.9  # opaque gray outside mask
        ax.imshow(overlay, origin="upper", extent=[0, IMG_W, IMG_H, 0],
                  interpolation="nearest", zorder=2)
        # Draw mask contour
        ax.contour(mask, levels=[0.5], colors="k", linewidths=0.8,
                   extent=[0, IMG_W, IMG_H, 0], origin="upper", zorder=3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(tcf, cax=cax)

    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def compute_rmse(U, coords, gt_u, gt_v, mask):
    """Compute RMSE on valid interior nodes."""
    u_comp = U[0::2]
    v_comp = U[1::2]
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


def _run_dic_comparison(ref_speckle, mask, u_func, v_func, winstepsize=16, winsize_min=8):
    """Run DIC on both uniform and quadtree meshes, return comparison data."""
    from dataclasses import replace as dc_replace

    margin = winstepsize
    xs = np.arange(margin, IMG_W - margin + 1, winstepsize, dtype=np.float64)
    ys = np.arange(margin, IMG_H - margin + 1, winstepsize, dtype=np.float64)

    para = dicpara_default(
        winsize=32, winstepsize=winstepsize, winsize_min=winsize_min,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative", admm_max_iter=3, admm_tol=1e-2,
        show_plots=False, icgn_max_iter=50, tol=1e-2,
        mu=1e-3, gauss_pt_order=2, alpha=0.0, img_ref_mask=mask,
    )

    deformed = apply_displacement_lagrangian(ref_speckle, u_func, v_func)

    # Uniform mesh
    mesh_uni = mesh_setup(xs, ys, para)
    n_uni = mesh_uni.coordinates_fem.shape[0]
    r_uni = run_aldic(
        para, [ref_speckle, deformed], [mask, mask],
        mesh=mesh_uni, U0=np.zeros(2 * n_uni), compute_strain=False,
    )

    # Quadtree mesh
    f_img = ref_speckle / ref_speckle.max()
    Df = compute_image_gradient(f_img, mask)
    U0_init = np.zeros(2 * n_uni)
    mesh_qt, U0_qt = generate_mesh(mesh_uni, para, Df, U0_init)
    r_qt = run_aldic(
        para, [ref_speckle, deformed], [mask, mask],
        mesh=mesh_qt, U0=np.zeros(2 * mesh_qt.coordinates_fem.shape[0]),
        compute_strain=False,
    )

    return r_uni, r_qt


def plot_part2_dic_results(ref_speckle, output_dir: Path):
    """Run DIC on uniform and quadtree meshes for various affine cases.

    Figure 4: Full mask (no holes) — 6 cases x 3 columns (measured u, error u uniform, error u quadtree)
    Figure 5: Annular mask — same layout
    """
    mask_full = np.ones((IMG_H, IMG_W), dtype=np.float64)
    mask_ann = make_annular_mask(r_outer=90.0, r_inner=40.0)

    cases = {
        "Translation (2.5, -1.5)": (
            lambda x, y: np.full_like(x, 2.5),
            lambda x, y: np.full_like(x, -1.5),
        ),
        "Expansion (2%)": (
            lambda x, y: 0.02 * (x - CX),
            lambda x, y: 0.02 * (y - CY),
        ),
        "Pure shear (1.5%)": (
            lambda x, y: 0.015 * (y - CY),
            lambda x, y: np.zeros_like(x),
        ),
        "Stretch+shear (2%+1%)": (
            lambda x, y: 0.02 * (x - CX) + 0.01 * (y - CY),
            lambda x, y: 0.01 * (x - CX) + 0.02 * (y - CY),
        ),
        "Rotation (2 deg)": (
            lambda x, y: (x - CX) * (np.cos(np.pi / 90) - 1)
            - (y - CY) * np.sin(np.pi / 90),
            lambda x, y: (x - CX) * np.sin(np.pi / 90)
            + (y - CY) * (np.cos(np.pi / 90) - 1),
        ),
        "Asym. stretch (3%x,1%y)": (
            lambda x, y: 0.03 * (x - CX),
            lambda x, y: 0.01 * (y - CY),
        ),
    }

    for mask, mask_name, fig_name in [
        (mask_full, "Full Mask (no holes)", "fig4_dic_full_mask.png"),
        (mask_ann, "Annular Mask (r_in=40, r_out=90)", "fig5_dic_annular_mask.png"),
    ]:
        n_cases = len(cases)
        # Extra top margin for column headers
        fig, axes = plt.subplots(
            n_cases, 5, figsize=(22, n_cases * 3.2 + 1.0),
            gridspec_kw={"top": 0.91},
        )
        fig.suptitle(
            f"Uniform vs Quadtree DIC ({mask_name}, step=16, min=8)",
            fontsize=13, y=0.97,
        )

        # Fixed column headers above the top row using fig.text
        col_labels = [
            "Ground Truth  u",
            "Uniform Mesh\nError  u",
            "Uniform Mesh\nError  v",
            "Quadtree Mesh\nError  u",
            "Quadtree Mesh\nError  v",
        ]
        for c, label in enumerate(col_labels):
            # Get axis position to center text above each column
            bbox = axes[0, c].get_position()
            fig.text(
                (bbox.x0 + bbox.x1) / 2, 0.935,
                label, ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )

        rmse_rows = []

        for row, (case_name, (u_func, v_func)) in enumerate(cases.items()):
            print(f"  [{mask_name[:4]}] Running: {case_name}...")

            try:
                r_uni, r_qt = _run_dic_comparison(
                    ref_speckle, mask, u_func, v_func,
                )

                # Extract results
                def _extract(result):
                    fr = result.result_disp[0]
                    U = fr.U_accum if fr.U_accum is not None else fr.U
                    c = result.dic_mesh.coordinates_fem
                    return U, c

                U_uni, c_uni = _extract(r_uni)
                U_qt, c_qt = _extract(r_qt)

                # Ground truth
                gt_u_uni = u_func(c_uni[:, 0], c_uni[:, 1])
                gt_u_qt = u_func(c_qt[:, 0], c_qt[:, 1])
                gt_v_uni = v_func(c_uni[:, 0], c_uni[:, 1])
                gt_v_qt = v_func(c_qt[:, 0], c_qt[:, 1])

                # Errors
                eu_uni = U_uni[0::2] - gt_u_uni
                ev_uni = U_uni[1::2] - gt_v_uni
                eu_qt = U_qt[0::2] - gt_u_qt
                ev_qt = U_qt[1::2] - gt_v_qt

                # RMSE (mask-interior nodes only)
                rmse_uni = compute_rmse(U_uni, c_uni, gt_u_uni, gt_v_uni, mask)
                rmse_qt = compute_rmse(U_qt, c_qt, gt_u_qt, gt_v_qt, mask)

                rmse_rows.append((
                    case_name, rmse_uni[0], rmse_uni[1], rmse_qt[0], rmse_qt[1],
                ))

                # Col 0: Ground truth u field (on quadtree mesh)
                plot_displacement_field(
                    axes[row, 0], c_qt, gt_u_qt, "", cmap="jet",
                    mask=mask,
                )

                # Shared error color scale per row (mask-interior only)
                def _mask_filter(err, coords):
                    h, w = mask.shape
                    ix = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
                    iy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
                    ok = (mask[iy, ix] > 0.5) & np.isfinite(err)
                    return err[ok]

                all_err = np.concatenate([
                    _mask_filter(eu_uni, c_uni),
                    _mask_filter(ev_uni, c_uni),
                    _mask_filter(eu_qt, c_qt),
                    _mask_filter(ev_qt, c_qt),
                ])
                err_lim = max(
                    np.nanpercentile(np.abs(all_err), 95), 0.01,
                ) if len(all_err) > 0 else 0.01

                # Col 1-2: Uniform error u, v
                plot_displacement_field(
                    axes[row, 1], c_uni, eu_uni,
                    f"RMSE_u = {rmse_uni[0]:.4f}", cmap="RdBu_r",
                    vmin=-err_lim, vmax=err_lim, mask=mask,
                )
                plot_displacement_field(
                    axes[row, 2], c_uni, ev_uni,
                    f"RMSE_v = {rmse_uni[1]:.4f}", cmap="RdBu_r",
                    vmin=-err_lim, vmax=err_lim, mask=mask,
                )

                # Col 3-4: Quadtree error u, v
                plot_displacement_field(
                    axes[row, 3], c_qt, eu_qt,
                    f"RMSE_u = {rmse_qt[0]:.4f}", cmap="RdBu_r",
                    vmin=-err_lim, vmax=err_lim, mask=mask,
                )
                plot_displacement_field(
                    axes[row, 4], c_qt, ev_qt,
                    f"RMSE_v = {rmse_qt[1]:.4f}", cmap="RdBu_r",
                    vmin=-err_lim, vmax=err_lim, mask=mask,
                )

                axes[row, 0].set_ylabel(
                    case_name, fontsize=9, fontweight="bold",
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                for c in range(5):
                    axes[row, c].text(
                        0.5, 0.5, f"Error:\n{e}",
                        transform=axes[row, c].transAxes,
                        ha="center", va="center", fontsize=7, color="red",
                    )
                axes[row, 0].set_ylabel(case_name, fontsize=9, fontweight="bold")

        fig.subplots_adjust(
            left=0.06, right=0.97, bottom=0.03, top=0.91,
            hspace=0.35, wspace=0.30,
        )
        fig.savefig(output_dir / fig_name, dpi=150)
        plt.close(fig)
        print(f"  Saved {fig_name}")

        # RMSE summary table
        print(f"\n{'=' * 80}")
        print(f"RMSE Summary -- {mask_name}")
        print(f"{'=' * 80}")
        print(f"{'Case':<30s} {'Uni_u':>8s} {'Uni_v':>8s} {'QT_u':>8s} {'QT_v':>8s}")
        print(f"{'-' * 80}")
        for name, uu, uv, qu, qv in rmse_rows:
            print(f"{name:<30s} {uu:>8.5f} {uv:>8.5f} {qu:>8.5f} {qv:>8.5f}")
        print(f"{'=' * 80}\n")

    # (old fig5 overlay and RMSE table removed — integrated into fig4/fig5 above)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize quadtree mesh and DIC results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/quadtree",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reference speckle pattern...")
    ref_speckle = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)

    print("\n=== Part 1: Mesh Structure Comparison ===")
    plot_part1_mesh_comparison(ref_speckle, output_dir)

    print("\n=== Part 2: DIC Results on Quadtree Mesh ===")
    plot_part2_dic_results(ref_speckle, output_dir)

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
