#!/usr/bin/env python
"""Visual report: refinement modes on the bubble_30_150_30 experimental dataset.

Runs the AL-DIC pipeline four times on a real bubble-expansion image pair
(frame 0 -> frame 12) with the four refinement configurations exposed in the
GUI:
  1. Uniform        (no refinement)
  2. Inner only     (MaskBoundaryCriterion -- refines around the bubble edge)
  3. Outer only     (ROIEdgeCriterion -- refines along the rectangular ROI rim)
  4. Inner + Outer  (both criteria)

Because this is real data (no analytic ground truth), we treat the *uniform*
run as the baseline and report:
  * Mesh statistics  (node / element counts, valid nodes)
  * Displacement field maps for each configuration
  * |delta U| field vs uniform (highlights where refinement actually changes
    things, which is the whole point of the feature)
  * A summary table

Output: al-dic/reports/bubble_refinement_comparison.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.mesh.refinement import build_refinement_policy

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUBBLE_DIR = REPO_ROOT / "examples" / "bubble_30_150_30"
IMG_DIR = BUBBLE_DIR / "images"
MASK_DIR = BUBBLE_DIR / "masks"

REF_INDEX = 0
DEF_INDEX = 12  # mid-sequence -> moderate bubble expansion

# Mesh / subset parameters (match the GUI defaults that the user sees)
SUBSET_SIZE = 40         # internal even value (display 41)
SUBSET_STEP = 16         # power of 2
REFINE_LEVEL = 2         # min element size = max(2, 16/4) = 4 px
HALF_WIN = SUBSET_SIZE // 2

OUTPUT_PDF = (
    REPO_ROOT / "al-dic" / "reports" / "bubble_refinement_comparison.pdf"
)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_grayscale(path: Path) -> np.ndarray:
    """Load an 8-bit image as float64 in the [0, 255] range."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask as float64 with values in {0.0, 1.0}."""
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 127).astype(np.float64)


def compute_min_size(step: int, level: int) -> int:
    """Mirror AppState.compute_refinement_min_size (floor at 2 px)."""
    return max(2, step // (2 ** level))


# ═══════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════

def build_para(img_shape: tuple[int, int], mask: np.ndarray):
    """Build a DICPara matching the bubble dataset + GUI defaults."""
    rows, cols = np.where(mask > 0.5)
    if rows.size == 0:
        raise RuntimeError("Reference mask is empty.")
    roi = GridxyROIRange(
        gridx=(int(cols.min()), int(cols.max())),
        gridy=(int(rows.min()), int(rows.max())),
    )
    return dicpara_default(
        winsize=SUBSET_SIZE,
        winstepsize=SUBSET_STEP,
        winsize_min=min(8, SUBSET_STEP),
        size_of_fft_search_region=20,
        gridxy_roi_range=roi,
        img_size=img_shape,
        reference_mode="accumulative",
        show_plots=False,
    )


def run_one(
    label: str,
    para,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    refine_inner: bool,
    refine_outer: bool,
) -> dict:
    """Run run_aldic with one refinement configuration; collect results."""
    policy = build_refinement_policy(
        refine_inner_boundary=refine_inner,
        refine_outer_boundary=refine_outer,
        min_element_size=compute_min_size(SUBSET_STEP, REFINE_LEVEL),
        half_win=HALF_WIN,
    )
    print(f"  Running [{label}] (inner={refine_inner}, outer={refine_outer}) ...")
    result = run_aldic(
        para=para,
        images=images,
        masks=masks,
        compute_strain=False,
        refinement_policy=policy,
    )

    mesh = result.dic_mesh
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem
    valid_elems = int(np.any(elems >= 0, axis=1).sum())

    U = result.result_disp[0].U
    u_est = U[0::2]
    v_est = U[1::2]
    valid_nodes = int((~np.isnan(u_est) & ~np.isnan(v_est)).sum())

    return {
        "label": label,
        "policy": policy,
        "mesh": mesh,
        "coords": coords,
        "elements": elems,
        "valid_elems": valid_elems,
        "u": u_est,
        "v": v_est,
        "valid_nodes": valid_nodes,
        "n_nodes": coords.shape[0],
    }


# ═══════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════

def draw_mesh(ax, run: dict, mask: np.ndarray, title: str) -> None:
    """Render mesh elements over the mask backdrop."""
    h, w = mask.shape
    coords = run["coords"]
    elems = run["elements"]

    ax.imshow(mask, cmap="gray", alpha=0.35, origin="upper",
              extent=[0, w, h, 0])

    for e in range(elems.shape[0]):
        c4 = elems[e, :4]
        if np.any(c4 < 0):
            continue
        poly = np.vstack([coords[c4], coords[c4[0]]])
        ax.plot(poly[:, 0], poly[:, 1], color="C0", linewidth=0.35)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def draw_field_on_mesh(
    ax, run: dict, values: np.ndarray, mask: np.ndarray, title: str,
    cmap: str = "RdBu_r", vmin: float | None = None, vmax: float | None = None,
) -> None:
    """Draw a per-node scalar field by coloring each element with the
    mean of its 4 corner values."""
    h, w = mask.shape
    coords = run["coords"]
    elems = run["elements"]

    ax.imshow(mask, cmap="gray", alpha=0.2, origin="upper",
              extent=[0, w, h, 0])

    polys: list[np.ndarray] = []
    elem_vals: list[float] = []
    for e in range(elems.shape[0]):
        c4 = elems[e, :4]
        if np.any(c4 < 0):
            continue
        node_vals = values[c4]
        if np.any(np.isnan(node_vals)):
            continue
        polys.append(np.vstack([coords[c4], coords[c4[0]]]))
        elem_vals.append(float(np.mean(node_vals)))

    if not polys:
        ax.set_title(f"{title}\n(no valid elements)", fontsize=8)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        return

    elem_arr = np.array(elem_vals)
    if vmin is None:
        vmin = float(np.nanpercentile(elem_arr, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(elem_arr, 99))
    norm = plt.Normalize(vmin, vmax)
    colors = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(elem_arr)

    for poly, color in zip(polys, colors):
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color,
                edgecolor="gray", linewidth=0.05, alpha=0.9)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)


def interpolate_to_uniform_nodes(
    src_run: dict, dst_run: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor interpolate (u, v) from ``src_run`` nodes to the
    ``dst_run`` nodes so refinement runs can be diffed against the uniform
    baseline.

    NN is good enough for the diff visualization -- we want to highlight
    *where* the field changes, not produce a high-fidelity resampled field.
    """
    src_coords = src_run["coords"]
    dst_coords = dst_run["coords"]
    src_u, src_v = src_run["u"], src_run["v"]

    # Vectorized NN: O(N_dst * N_src), fine for ~1k-5k nodes per side
    dx = dst_coords[:, 0:1] - src_coords[:, 0:1].T
    dy = dst_coords[:, 1:2] - src_coords[:, 1:2].T
    nn_idx = np.argmin(dx * dx + dy * dy, axis=1)

    return src_u[nn_idx], src_v[nn_idx]


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_PDF.parent.mkdir(exist_ok=True)

    # ── Load images & masks ──────────────────────────────────────
    print(f"Loading bubble dataset from: {BUBBLE_DIR}")
    f_img = load_grayscale(IMG_DIR / f"frame_{REF_INDEX:02d}.png")
    g_img = load_grayscale(IMG_DIR / f"frame_{DEF_INDEX:02d}.png")
    f_mask = load_mask(MASK_DIR / f"mask_{REF_INDEX:02d}.png")
    g_mask = load_mask(MASK_DIR / f"mask_{DEF_INDEX:02d}.png")

    # Use the union mask so the analysis covers both states.
    union_mask = np.maximum(f_mask, g_mask)
    print(
        f"  ref={f_img.shape}, def shape ok, "
        f"union mask coverage={union_mask.mean()*100:.1f}%"
    )

    images = [f_img, g_img]
    masks = [union_mask, union_mask]

    # ── Build pipeline para ──────────────────────────────────────
    para = build_para(f_img.shape, union_mask)
    min_size = compute_min_size(SUBSET_STEP, REFINE_LEVEL)
    print(
        f"Subset={SUBSET_SIZE} (display {SUBSET_SIZE+1}), "
        f"step={SUBSET_STEP}, refine_level={REFINE_LEVEL}, "
        f"min_element_size={min_size}px"
    )

    # ── Run all 4 configurations ─────────────────────────────────
    runs: list[dict] = []
    for label, ri, ro in [
        ("Uniform",        False, False),
        ("Inner only",     True,  False),
        ("Outer only",     False, True),
        ("Inner + Outer",  True,  True),
    ]:
        runs.append(
            run_one(label, para, images, masks,
                    refine_inner=ri, refine_outer=ro)
        )

    uniform = runs[0]

    # ── Build PDF ────────────────────────────────────────────────
    print(f"Writing report -> {OUTPUT_PDF}")
    with PdfPages(str(OUTPUT_PDF)) as pdf:
        _page_inputs(pdf, f_img, g_img, f_mask, g_mask, union_mask)
        _page_meshes(pdf, runs, union_mask)
        _page_displacement_fields(pdf, runs, union_mask)
        _page_diff_vs_uniform(pdf, runs, uniform, union_mask)
        _page_summary(pdf, runs, min_size)

    print("Done.")


# ── Pages ────────────────────────────────────────────────────────

def _page_inputs(pdf, f_img, g_img, f_mask, g_mask, union_mask) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        f"Inputs: bubble_30_150_30 frames {REF_INDEX:02d} -> {DEF_INDEX:02d}",
        fontsize=13, y=0.98,
    )

    axes[0].imshow(f_img, cmap="gray", origin="upper")
    axes[0].set_title(f"Reference frame {REF_INDEX:02d}", fontsize=9)

    axes[1].imshow(g_img, cmap="gray", origin="upper")
    axes[1].set_title(f"Deformed frame {DEF_INDEX:02d}", fontsize=9)

    axes[2].imshow(f_mask, cmap="gray", origin="upper")
    axes[2].set_title(
        f"Mask {REF_INDEX:02d} ({f_mask.mean()*100:.1f}%)", fontsize=9
    )

    axes[3].imshow(g_mask, cmap="gray", origin="upper")
    axes[3].set_title(
        f"Mask {DEF_INDEX:02d} ({g_mask.mean()*100:.1f}%)", fontsize=9
    )

    axes[4].imshow(union_mask, cmap="gray", origin="upper")
    axes[4].set_title(
        f"Union mask ({union_mask.mean()*100:.1f}%)", fontsize=9
    )

    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_meshes(pdf, runs: list[dict], mask: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle("Quadtree mesh per refinement configuration",
                 fontsize=13, y=0.98)
    flat = axes.flatten()
    for ax, run in zip(flat, runs):
        draw_mesh(ax, run, mask,
                  f"{run['label']}\n{run['n_nodes']} nodes, "
                  f"{run['valid_elems']} elems")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_displacement_fields(pdf, runs: list[dict], mask: np.ndarray) -> None:
    # Shared color ranges from the uniform run for visual comparability
    uni = runs[0]
    valid = ~np.isnan(uni["u"]) & ~np.isnan(uni["v"])
    if not valid.any():
        return
    u_lim = max(abs(np.nanpercentile(uni["u"][valid], 1)),
                abs(np.nanpercentile(uni["u"][valid], 99)))
    v_lim = max(abs(np.nanpercentile(uni["v"][valid], 1)),
                abs(np.nanpercentile(uni["v"][valid], 99)))

    # u field
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle("u displacement (px) -- color range fixed from Uniform",
                 fontsize=13, y=0.98)
    for ax, run in zip(axes.flatten(), runs):
        draw_field_on_mesh(
            ax, run, run["u"], mask,
            f"{run['label']}: u",
            cmap="RdBu_r", vmin=-u_lim, vmax=u_lim,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # v field
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle("v displacement (px) -- color range fixed from Uniform",
                 fontsize=13, y=0.98)
    for ax, run in zip(axes.flatten(), runs):
        draw_field_on_mesh(
            ax, run, run["v"], mask,
            f"{run['label']}: v",
            cmap="RdBu_r", vmin=-v_lim, vmax=v_lim,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_diff_vs_uniform(
    pdf, runs: list[dict], uniform: dict, mask: np.ndarray,
) -> None:
    """For each refined run, plot |U_refined - NN(U_uniform)|."""
    refined = [r for r in runs if r["label"] != "Uniform"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "|U_refined - U_uniform| (NN-resampled) -- where refinement changes the field",
        fontsize=12, y=0.98,
    )

    diffs_for_scale: list[np.ndarray] = []
    per_run_diff: list[np.ndarray] = []
    for run in refined:
        u_uni_nn, v_uni_nn = interpolate_to_uniform_nodes(uniform, run)
        diff = np.sqrt((run["u"] - u_uni_nn) ** 2 + (run["v"] - v_uni_nn) ** 2)
        per_run_diff.append(diff)
        valid = ~np.isnan(diff)
        if valid.any():
            diffs_for_scale.append(diff[valid])

    if diffs_for_scale:
        all_d = np.concatenate(diffs_for_scale)
        vmax = float(np.nanpercentile(all_d, 99))
        vmax = max(vmax, 1e-3)
    else:
        vmax = 1.0

    for ax, run, diff in zip(axes, refined, per_run_diff):
        draw_field_on_mesh(
            ax, run, diff, mask,
            f"{run['label']}\nmax={np.nanmax(diff):.3f}px, "
            f"mean={np.nanmean(diff):.3f}px",
            cmap="magma", vmin=0.0, vmax=vmax,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_summary(pdf, runs: list[dict], min_size: int) -> None:
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Refinement comparison summary", fontsize=13, y=0.98)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    col_labels = [
        "Configuration", "Inner", "Outer", "Nodes",
        "Valid elements", "Valid nodes",
        "Nodes vs Uniform", "Min elem size (px)",
    ]
    uni_nodes = runs[0]["n_nodes"]
    rows: list[list[str]] = []
    for run in runs:
        ri = "yes" if run["policy"] is not None and any(
            "MaskBoundary" in type(c).__name__ for c in run["policy"].pre_solve
        ) else "no"
        ro = "yes" if run["policy"] is not None and any(
            "ROIEdge" in type(c).__name__ for c in run["policy"].pre_solve
        ) else "no"
        delta = run["n_nodes"] - uni_nodes
        pct = 100 * delta / max(1, uni_nodes)
        rows.append([
            run["label"], ri, ro,
            str(run["n_nodes"]),
            str(run["valid_elems"]),
            str(run["valid_nodes"]),
            f"{delta:+d} ({pct:+.1f}%)",
            str(min_size),
        ])

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.9)
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        color = "#D6E4F0" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
