"""Benchmark: window splitting vs. no window splitting for masked DIC.

Creates a synthetic speckle image with a configurable mask (notch or annular),
applies a known displacement field, and runs the full AL-DIC pipeline twice:

- **NEW (window splitting)**: raw image passed to IC-GN precompute; gradients
  computed from raw image; mask used only for pixel validity.
- **OLD (no window splitting)**: pre-masked image passed as "raw"; gradients
  see the artificial speckle->0 edge at mask boundaries.

Usage:
    python bench_window_splitting.py [notch|annular]   (default: annular)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICMesh, GridxyROIRange, merge_uv
from staq_dic.core.pipeline import run_aldic

logging.getLogger("staq_dic").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0
STEP = 8
MARGIN = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_speckle(h=256, w=256, sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    f = gaussian_filter(noise, sigma=sigma, mode="nearest")
    f -= f.min(); f /= f.max()
    return 20.0 + 215.0 * f


def apply_displacement_lagrangian(ref, u_func, v_func, n_iter=20):
    from scipy.ndimage import map_coordinates
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    warped = map_coordinates(ref, [Y.ravel(), X.ravel()], order=5, mode="nearest")
    return warped.reshape(h, w)


def make_mesh(h=256, w=256, step=8, margin=8):
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    ny, nx = len(ys), len(xs)
    elems = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n0 = iy * nx + ix
            elems.append([n0, n0+1, n0+nx+1, n0+nx, -1, -1, -1, -1])
    elems = np.array(elems, dtype=np.int64) if elems else np.empty((0, 8), dtype=np.int64)
    return DICMesh(coordinates_fem=coords, elements_fem=elems, x0=xs, y0=ys)


def make_annular_mask(h=256, w=256, cx=127.0, cy=127.0, r_outer=100.0, r_inner=40.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d2 = (xx - cx)**2 + (yy - cy)**2
    return ((d2 <= r_outer**2) & (d2 > r_inner**2)).astype(np.float64)


def make_notch_mask(h=256, w=256, notch_x=(100, 140), notch_y=(0, 128)):
    mask = np.ones((h, w), dtype=np.float64)
    mask[notch_y[0]:notch_y[1], notch_x[0]:notch_x[1]] = 0.0
    return mask


def classify_boundary_nodes(coords, mask, winsize):
    h, w = mask.shape
    half = winsize // 2
    n = len(coords)
    is_boundary = np.zeros(n, dtype=bool)
    for i in range(n):
        cx_i, cy_i = int(round(coords[i, 0])), int(round(coords[i, 1]))
        if cx_i < 0 or cx_i >= w or cy_i < 0 or cy_i >= h:
            continue
        if mask[cy_i, cx_i] < 0.5:
            continue
        x_lo, x_hi = max(0, cx_i - half), min(w - 1, cx_i + half)
        y_lo, y_hi = max(0, cy_i - half), min(h - 1, cy_i + half)
        if np.any(mask[y_lo:y_hi+1, x_lo:x_hi+1] < 0.5):
            is_boundary[i] = True
    cx_arr = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
    cy_arr = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
    in_mask = mask[cy_arr, cx_arr] > 0.5
    return is_boundary, in_mask & ~is_boundary


def compute_rmse(U, gt_u, gt_v, sel):
    u, v = U[0::2], U[1::2]
    ok = sel & np.isfinite(u) & np.isfinite(v)
    if not np.any(ok):
        return np.inf, np.inf
    return (float(np.sqrt(np.mean((u[ok] - gt_u[ok])**2))),
            float(np.sqrt(np.mean((v[ok] - gt_v[ok])**2))))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(mask_type: str = "annular"):
    # --- Mask ---
    if mask_type == "annular":
        mask = make_annular_mask(IMG_H, IMG_W, CX, CY, r_outer=100.0, r_inner=40.0)
        mask_desc = "Annular mask (r_outer=100, r_inner=40)"
    elif mask_type == "notch":
        mask = make_notch_mask(IMG_H, IMG_W, notch_x=(100, 140), notch_y=(0, 128))
        mask_desc = "Notch mask x=[100,140] y=[0,128]"
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    winsize = 16

    print("=" * 70)
    print(f"  Benchmark: Window Splitting — {mask_desc}")
    print(f"  2% affine stretch, {IMG_H}x{IMG_W}, step={STEP}, winsize={winsize}")
    print("=" * 70)

    # --- Test data ---
    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)
    u_func = lambda x, y: 0.02 * (x - CX)
    v_func = lambda x, y: 0.02 * (y - CY)
    deformed = apply_displacement_lagrangian(ref, u_func, v_func)

    mesh = make_mesh(IMG_H, IMG_W, step=STEP, margin=MARGIN)
    node_x, node_y = mesh.coordinates_fem[:, 0], mesh.coordinates_fem[:, 1]
    gt_u, gt_v = u_func(node_x, node_y), v_func(node_x, node_y)
    U0 = merge_uv(gt_u, gt_v)

    is_boundary, is_interior = classify_boundary_nodes(
        mesh.coordinates_fem, mask, winsize=winsize,
    )
    cx_arr = np.clip(np.round(node_x).astype(int), 0, IMG_W - 1)
    cy_arr = np.clip(np.round(node_y).astype(int), 0, IMG_H - 1)
    in_mask = mask[cy_arr, cx_arr] > 0.5

    n_bdry = int(np.sum(is_boundary))
    print(f"\nNodes: {len(node_x)} total, {int(np.sum(in_mask))} in mask, "
          f"{n_bdry} boundary, {int(np.sum(is_interior))} interior\n")

    # --- DICPara ---
    para = dicpara_default(
        winsize=winsize, winstepsize=STEP, winsize_min=4,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative",
        admm_max_iter=3, admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0, disp_smoothness=0.0, smoothness=0.0,
        show_plots=False, icgn_max_iter=50, tol=1e-2,
        mu=1e-3, gauss_pt_order=2, alpha=0.0, use_masks=True,
    )

    # --- Run ---
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    modes = [
        ("NEW (window splitting)",    [ref, deformed]),
        ("OLD (no window splitting)", [ref * mask, deformed * mask]),
    ]
    results = {}

    for mode_name, images_list in modes:
        timings = []
        for trial in range(3):
            t0 = time.perf_counter()
            result = run_aldic(
                para, images=images_list, masks=[mask, mask],
                compute_strain=False, mesh=mesh, U0=U0.copy(),
            )
            timings.append(time.perf_counter() - t0)

        elapsed = np.median(timings[1:])
        U_final = result.result_disp[0].U
        rmse_all = compute_rmse(U_final, gt_u, gt_v, in_mask)
        rmse_bdry = compute_rmse(U_final, gt_u, gt_v, is_boundary)
        rmse_intr = compute_rmse(U_final, gt_u, gt_v, is_interior)

        results[mode_name] = dict(
            time=elapsed, rmse_all=rmse_all, rmse_bdry=rmse_bdry,
            rmse_intr=rmse_intr, U=U_final,
        )
        t_str = ", ".join(f"{t:.3f}" for t in timings)
        print(f"{mode_name}:")
        print(f"  Time:     {elapsed:.3f}s  (trials: {t_str})")
        for lbl, rm in [("all", rmse_all), ("bdy", rmse_bdry), ("int", rmse_intr)]:
            print(f"  RMSE {lbl}: u={rm[0]:.4f}  v={rm[1]:.4f}  "
                  f"total={np.sqrt(rm[0]**2 + rm[1]**2):.4f} px")
        print()

    # --- Summary table ---
    new, old = results["NEW (window splitting)"], results["OLD (no window splitting)"]

    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'NEW':>14} {'OLD':>14} {'Change':>12}")
    print("-" * 70)
    for label, key in [("All nodes", "rmse_all"),
                       ("Boundary nodes", "rmse_bdry"),
                       ("Interior nodes", "rmse_intr")]:
        nr = np.sqrt(new[key][0]**2 + new[key][1]**2)
        or_ = np.sqrt(old[key][0]**2 + old[key][1]**2)
        chg = (nr - or_) / or_ * 100 if or_ > 1e-10 else 0.0
        print(f"  RMSE {label:<18} {nr:>11.4f} px {or_:>11.4f} px {chg:>+10.1f}%")
    ct = (new["time"] - old["time"]) / old["time"] * 100 if old["time"] > 0 else 0
    print(f"  {'Time':<25} {new['time']:>11.3f} s  {old['time']:>11.3f} s  {ct:>+10.1f}%")
    print()

    # --- Boundary detail (top 20 by OLD error) ---
    bdry_idx = np.where(is_boundary)[0]
    emag_new = np.sqrt((new["U"][0::2] - gt_u)**2 + (new["U"][1::2] - gt_v)**2)
    emag_old = np.sqrt((old["U"][0::2] - gt_u)**2 + (old["U"][1::2] - gt_v)**2)
    sorted_bdry = bdry_idx[np.argsort(-emag_old[bdry_idx])]
    show_n = min(20, len(sorted_bdry))

    print(f"Boundary nodes (top {show_n} by OLD error, of {len(bdry_idx)} total):")
    print(f"  {'Node':>5} {'x':>6} {'y':>6} {'|e_new|':>10} {'|e_old|':>10} {'ratio':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for idx in sorted_bdry[:show_n]:
        en, eo = emag_new[idx], emag_old[idx]
        r = eo / en if en > 1e-10 else float("inf")
        m = " <--" if r > 3 else ""
        print(f"  {idx:>5} {node_x[idx]:>6.0f} {node_y[idx]:>6.0f} "
              f"{en:>10.4f} {eo:>10.4f} {r:>7.1f}x{m}")

    # --- Figures ---
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / f"bench_{mask_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_results(
        ref, mask, mask_type, mesh, gt_u, gt_v,
        new["U"], old["U"], is_boundary, in_mask,
        emag_new, emag_old, out_dir,
    )
    print(f"\nFigures saved to {out_dir}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _grid(vals, nx, ny, in_mask):
    g = vals.reshape(ny, nx).astype(np.float64).copy()
    g[~in_mask.reshape(ny, nx)] = np.nan
    return g


def _draw_mask_contour(ax, mask, color="red", lw=2):
    """Draw mask boundary as contour lines."""
    ax.contour(
        np.arange(mask.shape[1]), np.arange(mask.shape[0]),
        mask, levels=[0.5], colors=[color], linewidths=[lw], linestyles=["--"],
    )


def plot_results(
    ref, mask, mask_type, mesh, gt_u, gt_v,
    U_new, U_old, is_boundary, in_mask,
    emag_new, emag_old, out_dir,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    node_x, node_y = mesh.coordinates_fem[:, 0], mesh.coordinates_fem[:, 1]
    xs, ys = np.unique(node_x), np.unique(node_y)
    nx, ny = len(xs), len(ys)

    u_new, v_new = U_new[0::2], U_new[1::2]
    u_old, v_old = U_old[0::2], U_old[1::2]
    eu_new, ev_new = u_new - gt_u, v_new - gt_v
    eu_old, ev_old = u_old - gt_u, v_old - gt_v

    G = {}
    for k, a in [("gt_u", gt_u), ("gt_v", gt_v),
                 ("u_new", u_new), ("v_new", v_new),
                 ("u_old", u_old), ("v_old", v_old),
                 ("eu_new", eu_new), ("ev_new", ev_new),
                 ("eu_old", eu_old), ("ev_old", ev_old),
                 ("emag_new", emag_new), ("emag_old", emag_old)]:
        G[k] = _grid(a, nx, ny, in_mask)

    bx, by = node_x[is_boundary], node_y[is_boundary]

    def _ax_setup(ax, title):
        ax.imshow(ref, cmap="gray", alpha=0.3, extent=[0, IMG_W, IMG_H, 0])
        _draw_mask_contour(ax, mask)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
        ax.set_aspect("equal")

    def _scatter_bdry(ax):
        ax.scatter(bx, by, s=18, c="none", edgecolors="lime", linewidths=0.8, zorder=5)

    # =================================================================
    # Fig 1: Full-field displacement
    # =================================================================
    fig1, ax1 = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle(
        f"Full-Field Displacement — {mask_type} mask",
        fontsize=14, fontweight="bold",
    )
    for row, (c, gk, nk, ok, lab) in enumerate([
        ("u", "gt_u", "u_new", "u_old", "u [px]"),
        ("v", "gt_v", "v_new", "v_old", "v [px]"),
    ]):
        vmin = np.nanmin([G[gk], G[nk], G[ok]])
        vmax = np.nanmax([G[gk], G[nk], G[ok]])
        for col, (d, t) in enumerate([
            (G[gk], f"Ground Truth {c}"),
            (G[nk], f"NEW (split) {c}"),
            (G[ok], f"OLD (no split) {c}"),
        ]):
            a = ax1[row, col]
            _ax_setup(a, t)
            im = a.pcolormesh(xs - STEP/2, ys - STEP/2, d,
                              cmap="RdBu_r", vmin=vmin, vmax=vmax,
                              shading="auto", alpha=0.85)
            fig1.colorbar(im, ax=a, shrink=0.65, label=lab)
            if col > 0:
                _scatter_bdry(a)
    fig1.tight_layout()
    fig1.savefig(out_dir / "full_field_displacement.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # =================================================================
    # Fig 2: Error maps
    # =================================================================
    fig2, ax2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(
        f"Displacement Error — {mask_type} mask",
        fontsize=14, fontweight="bold",
    )
    for row, (c, nk, ok, lab) in enumerate([
        ("u", "eu_new", "eu_old", "u error [px]"),
        ("v", "ev_new", "ev_old", "v error [px]"),
    ]):
        emax = max(np.nanmax(np.abs(G[nk])), np.nanmax(np.abs(G[ok])))
        norm = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)
        diff = np.abs(G[ok]) - np.abs(G[nk])
        valid_diff = diff[np.isfinite(diff)]
        if len(valid_diff) > 0:
            dlim = float(max(abs(valid_diff.min()), abs(valid_diff.max()), 1e-6))
        else:
            dlim = 1.0
        dnorm = None  # use vmin/vmax directly
        items = [
            (G[nk], f"NEW err {c}", "RdBu_r", dict(norm=norm), lab),
            (G[ok], f"OLD err {c}", "RdBu_r", dict(norm=norm), lab),
            (diff, f"|OLD|-|NEW| {c}", "RdYlGn",
             dict(vmin=-dlim, vmax=dlim), "error reduction [px]"),
        ]
        for col, (d, t, cm, kw, cl) in enumerate(items):
            a = ax2[row, col]
            _ax_setup(a, t)
            im = a.pcolormesh(xs - STEP/2, ys - STEP/2, d,
                              cmap=cm, shading="auto", alpha=0.85, **kw)
            fig2.colorbar(im, ax=a, shrink=0.65, label=cl)
            _scatter_bdry(a)
    fig2.tight_layout()
    fig2.savefig(out_dir / "error_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # =================================================================
    # Fig 3: Error magnitude + bar chart
    # =================================================================
    fig3, ax3 = plt.subplots(2, 2, figsize=(14, 11))
    fig3.suptitle(
        f"Error Magnitude & Boundary Comparison — {mask_type} mask",
        fontsize=14, fontweight="bold",
    )

    emax_mag = max(np.nanmax(G["emag_new"]), np.nanmax(G["emag_old"]))
    for col, (k, t) in enumerate([
        ("emag_new", "NEW |error|"), ("emag_old", "OLD |error|"),
    ]):
        a = ax3[0, col]
        _ax_setup(a, t)
        im = a.pcolormesh(xs - STEP/2, ys - STEP/2, G[k],
                          cmap="hot_r", vmin=0, vmax=emax_mag,
                          shading="auto", alpha=0.85)
        _scatter_bdry(a)
        fig3.colorbar(im, ax=a, shrink=0.65, label="|error| [px]")

    # Bar chart — show top-N boundary nodes sorted by OLD error
    bdry_idx = np.where(is_boundary)[0]
    sorted_bdry = bdry_idx[np.argsort(-emag_old[bdry_idx])]
    show_n = min(30, len(sorted_bdry))
    sel = sorted_bdry[:show_n]

    xp = np.arange(show_n)
    bw_ = 0.35
    ax_b = ax3[1, 0]
    ax_b.bar(xp - bw_/2, emag_new[sel], bw_, label="NEW", color="#2196F3", alpha=0.85)
    ax_b.bar(xp + bw_/2, emag_old[sel], bw_, label="OLD", color="#FF5722", alpha=0.85)
    lbl = [f"({node_x[i]:.0f},{node_y[i]:.0f})" for i in sel]
    ax_b.set_xticks(xp); ax_b.set_xticklabels(lbl, rotation=70, ha="right", fontsize=6)
    ax_b.set_ylabel("|error| [px]"); ax_b.set_title(f"Top-{show_n} Boundary Errors")
    ax_b.legend(fontsize=8); ax_b.grid(axis="y", alpha=0.3)

    ax_r = ax3[1, 1]
    ratio = emag_old[sel] / np.maximum(emag_new[sel], 1e-10)
    colors = ["#4CAF50" if r > 1 else "#FF9800" for r in ratio]
    ax_r.bar(xp, ratio, 0.6, color=colors, alpha=0.85)
    ax_r.axhline(1.0, color="k", lw=1, ls="--", alpha=0.5)
    ax_r.set_xticks(xp); ax_r.set_xticklabels(lbl, rotation=70, ha="right", fontsize=6)
    ax_r.set_ylabel("OLD / NEW ratio"); ax_r.set_title("Improvement Ratio (green=better)")
    ax_r.grid(axis="y", alpha=0.3)
    for i, r in enumerate(ratio):
        if r > 3:
            ax_r.text(i, r + max(ratio)*0.02, f"{r:.0f}x", ha="center",
                      fontsize=6, fontweight="bold", color="#2E7D32")

    fig3.tight_layout()
    fig3.savefig(out_dir / "error_magnitude_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"  Saved: full_field_displacement.png")
    print(f"  Saved: error_maps.png")
    print(f"  Saved: error_magnitude_comparison.png")


if __name__ == "__main__":
    mt = sys.argv[1] if len(sys.argv) > 1 else "annular"
    run_benchmark(mt)
