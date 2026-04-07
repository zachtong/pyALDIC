#!/usr/bin/env python
"""Visual report: brush refinement mask follows material points across ref frames.

End-to-end PDF report demonstrating the auto-warp behaviour added in Task 7
of the brush-refinement plan: when AL-DIC switches to a non-zero reference
frame K, the user-painted brush mask must track the **same material points**
as they translate / deform across the image, *not* stay glued to frame-0
pixel locations.

Setup:
    - 256x256 synthetic speckle, full-image mask.
    - 5 frames with a known +3 px / frame translation in x.
    - FrameSchedule = (0, 0, 2, 2) — frames 1,2 reference frame 0,
      frames 3,4 reference frame 2.  This is the simplest schedule
      that exercises both the "raw mask" path (ref=0) and the
      "warped mask" path (ref=2, parent=0).
    - Brush mask: a 40x40 box painted on frame 0.

For each frame the report shows:
    1. The frame's image with the brush mask overlay (warped or raw).
    2. The refined Q4 mesh element edges, highlighting elements marked
       for refinement by the BrushRegionCriterion.
    3. A diagnostic table with painted-px count and refined-element count
       per frame.

Output: reports/brush_refinement_warp.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from conftest import apply_displacement_lagrangian, generate_speckle  # noqa: E402

from staq_dic.core._brush_warp import warp_brush_mask_to_ref  # noqa: E402
from staq_dic.core.config import dicpara_default  # noqa: E402
from staq_dic.core.data_structures import FrameSchedule, GridxyROIRange  # noqa: E402
from staq_dic.core.pipeline import run_aldic  # noqa: E402
from staq_dic.mesh.criteria.brush_region import BrushRegionCriterion  # noqa: E402
from staq_dic.mesh.refinement import build_refinement_policy  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

IMG_H, IMG_W = 256, 256
STEP = 16
WINSIZE = 32
WINSIZE_MIN = 4

N_FRAMES = 5
TX_PER_FRAME = 3.0  # +3 px / frame in x

BRUSH_BOX = (110, 60, 150, 100)  # (x_min, y_min, x_max, y_max) on frame 0


# ─────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────────────────────────────


def build_translation_dataset(
    n_frames: int,
    tx_per_frame: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build n_frames speckle images with cumulative +tx_per_frame in x."""
    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)

    images = [ref]
    for k in range(1, n_frames):
        tx = tx_per_frame * k
        deformed = apply_displacement_lagrangian(
            ref,
            lambda x, y, _tx=tx: np.full_like(x, _tx),
            lambda x, y: np.zeros_like(x),
        )
        images.append(deformed)

    masks = [mask] * n_frames
    return images, masks


def make_brush_mask() -> np.ndarray:
    """Paint a 40x40 box brush region on frame 0."""
    brush = np.zeros((IMG_H, IMG_W), dtype=bool)
    x_min, y_min, x_max, y_max = BRUSH_BOX
    brush[y_min:y_max, x_min:x_max] = True
    return brush


# ─────────────────────────────────────────────────────────────────────
# Pipeline run
# ─────────────────────────────────────────────────────────────────────


def run_pipeline(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    brush_mask: np.ndarray,
):
    """Run AL-DIC with the (0,0,2,2) schedule and the brush refinement on."""
    schedule = FrameSchedule(ref_indices=(0, 0, 2, 2))

    para = dicpara_default(
        winsize=WINSIZE,
        winstepsize=STEP,
        winsize_min=WINSIZE_MIN,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, IMG_W - 1), gridy=(0, IMG_H - 1)),
        frame_schedule=schedule,
        reference_mode="incremental",
        admm_max_iter=3,
        admm_tol=1e-2,
        method_to_compute_strain=3,
        smoothness=0.0,
        disp_smoothness=0.0,
        strain_smoothness=0.0,
        show_plots=False,
        icgn_max_iter=50,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        alpha=0.0,
    )

    policy = build_refinement_policy(
        refinement_mask=brush_mask.astype(np.float64),
        min_element_size=4,
        half_win=WINSIZE // 2,
    )

    result = run_aldic(
        para,
        images,
        masks,
        compute_strain=False,
        refinement_policy=policy,
    )
    return result, schedule, policy


# ─────────────────────────────────────────────────────────────────────
# Analysis: warp brush mask for each frame and count refined elements
# ─────────────────────────────────────────────────────────────────────


def expected_brush_at_frame(frame_idx: int, brush0: np.ndarray) -> np.ndarray:
    """Closed-form ground-truth: shift the painted box by +tx_per_frame*frame_idx."""
    shift = int(round(TX_PER_FRAME * frame_idx))
    out = np.zeros_like(brush0)
    if shift >= IMG_W:
        return out
    out[:, shift:] = brush0[:, : IMG_W - shift]
    return out


def warp_brush_for_pipeline_frame(
    brush0: np.ndarray,
    result,
    schedule: FrameSchedule,
    frame_idx: int,
) -> tuple[np.ndarray, str]:
    """Reproduce the in-pipeline warp for visualization purposes.

    Returns (warped_mask, label) where label describes which path was used.
    """
    if frame_idx == 0:
        return brush0.copy(), "frame 0 (reference)"

    ref_idx = schedule.parent(frame_idx)
    if ref_idx == 0:
        return brush0.copy(), f"raw mask (ref={ref_idx})"

    parent_of_ref = schedule.parent(ref_idx)
    if parent_of_ref != 0:
        return brush0.copy(), f"raw mask (parent({ref_idx})={parent_of_ref}!=0)"

    disp = result.result_disp[ref_idx - 1]
    mesh = result.result_fe_mesh_each_frame[ref_idx - 1]
    if disp is None or mesh is None:
        return brush0.copy(), "raw mask (no prior result)"

    warped = warp_brush_mask_to_ref(
        brush0, disp.U, mesh.coordinates_fem, (IMG_H, IMG_W)
    )
    return warped, f"warped to ref={ref_idx}"


def count_refined_elements(mesh, brush_for_ref: np.ndarray) -> tuple[int, int]:
    """Count (elements_in_brush, smallest_size_in_brush).

    Counts elements whose centroid lies inside the (warped) brush region.
    Also reports the smallest such element's edge length, which should be
    close to ``min_element_size`` if the auto-warp + refinement kicked in.
    """
    if mesh is None or mesh.elements_fem.shape[0] == 0:
        return 0, 0
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem[:, :4]
    cx = coords[elems, 0].mean(axis=1)
    cy = coords[elems, 1].mean(axis=1)
    # Sample brush mask at element centroids
    ix = np.clip(np.round(cx).astype(int), 0, IMG_W - 1)
    iy = np.clip(np.round(cy).astype(int), 0, IMG_H - 1)
    inside = brush_for_ref[iy, ix]
    n_inside = int(inside.sum())
    if n_inside == 0:
        return 0, 0
    # Smallest element edge length inside the brush region
    x_extent = coords[elems, 0].max(axis=1) - coords[elems, 0].min(axis=1)
    y_extent = coords[elems, 1].max(axis=1) - coords[elems, 1].min(axis=1)
    elem_size = np.minimum(x_extent, y_extent)
    smallest = int(elem_size[inside].min()) if n_inside > 0 else 0
    return n_inside, smallest


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────


def plot_frame(
    ax,
    image: np.ndarray,
    brush_for_frame: np.ndarray,
    mesh,
    title: str,
) -> None:
    """One panel: image + cyan brush overlay + mesh edges."""
    ax.imshow(image, cmap="gray", origin="upper", vmin=0, vmax=255)

    if brush_for_frame.any():
        cyan = np.zeros((IMG_H, IMG_W, 4), dtype=np.float64)
        cyan[..., 0] = 0.08  # R
        cyan[..., 1] = 0.86  # G
        cyan[..., 2] = 0.78  # B
        cyan[..., 3] = brush_for_frame.astype(np.float64) * 0.45
        ax.imshow(cyan, origin="upper")

    if mesh is not None and mesh.elements_fem.shape[0] > 0:
        coords = mesh.coordinates_fem
        elems = mesh.elements_fem[:, :4]  # Q4 corners
        segs = []
        for e in elems:
            for k in range(4):
                p0 = coords[e[k]]
                p1 = coords[e[(k + 1) % 4]]
                segs.append([p0, p1])
        lc = LineCollection(
            segs, colors="#ffd24a", linewidths=0.4, alpha=0.8,
        )
        ax.add_collection(lc)

    ax.set_xlim(0, IMG_W - 1)
    ax.set_ylim(IMG_H - 1, 0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_diagnostic_table(ax, rows: list[tuple]) -> None:
    """Render a small text table with per-frame metrics."""
    ax.axis("off")
    headers = (
        "frame", "ref", "path", "painted px", "elems in brush", "min edge (px)",
    )
    cell_text = [list(map(str, r)) for r in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.08, 0.06, 0.36, 0.16, 0.18, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Building synthetic translation dataset...")
    images, masks = build_translation_dataset(N_FRAMES, TX_PER_FRAME)
    brush0 = make_brush_mask()

    print("Running AL-DIC with brush refinement...")
    result, schedule, policy = run_pipeline(images, masks, brush0)

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "brush_refinement_warp.pdf"
    print(f"Writing report to {out_path}")

    rows: list[tuple] = []

    with PdfPages(out_path) as pdf:
        # Page 1: title + dataset overview
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(images[0], cmap="gray", origin="upper", vmin=0, vmax=255)
        cyan = np.zeros((IMG_H, IMG_W, 4))
        cyan[..., 0] = 0.08
        cyan[..., 1] = 0.86
        cyan[..., 2] = 0.78
        cyan[..., 3] = brush0.astype(float) * 0.5
        axes[0].imshow(cyan, origin="upper")
        axes[0].set_title("Frame 0 (reference) + painted brush", fontsize=10)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].axis("off")
        info_lines = [
            "Brush refinement material-point auto-warp",
            "",
            f"Image size:        {IMG_H} x {IMG_W}",
            f"Frames:            {N_FRAMES}",
            f"Translation:       +{TX_PER_FRAME} px / frame in x",
            f"Schedule:          {schedule.ref_indices} (incremental)",
            f"Brush box:         {BRUSH_BOX}",
            f"Win / step / min:  {WINSIZE} / {STEP} / {WINSIZE_MIN}",
            "",
            "Frames 1,2 use ref=0 -> raw mask path",
            "Frames 3,4 use ref=2 -> warped mask path",
        ]
        axes[1].text(
            0.0, 0.95, "\n".join(info_lines),
            family="monospace", fontsize=10,
            va="top", ha="left",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2-N: per-frame visualization
        n_cols = 2
        n_rows = (N_FRAMES + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 5 * n_rows))
        axes = axes.flatten()

        for f_idx in range(N_FRAMES):
            ax = axes[f_idx]
            mesh_for_f: object | None
            if f_idx == 0:
                # Frame 0 has no result; use the dataset image and the
                # painted brush as-is.  No mesh shown for the bare reference.
                brush_show = brush0
                mesh_for_f = None
                ref_label = "-"
                path_label = "frame 0 (reference)"
            else:
                ref_idx = schedule.parent(f_idx)
                ref_label = str(ref_idx)
                brush_show, path_label = warp_brush_for_pipeline_frame(
                    brush0, result, schedule, f_idx,
                )
                mesh_for_f = result.result_fe_mesh_each_frame[f_idx - 1]

            n_painted = int(brush_show.sum())
            n_in_brush, smallest = count_refined_elements(mesh_for_f, brush_show)
            rows.append(
                (f_idx, ref_label, path_label, n_painted, n_in_brush, smallest)
            )

            title = (
                f"Frame {f_idx}  (ref={ref_label})\n"
                f"{path_label}  |  {n_painted} px  |  "
                f"{n_in_brush} elems in brush (min edge {smallest} px)"
            )
            plot_frame(ax, images[f_idx], brush_show, mesh_for_f, title)

        # Hide unused panels
        for j in range(N_FRAMES, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            "Brush refinement mask following material points across frames",
            fontsize=12, y=0.995,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Final page: diagnostic table
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.set_title("Per-frame brush warp metrics", fontsize=11, pad=12)
        plot_diagnostic_table(ax, rows)
        pdf.savefig(fig)
        plt.close(fig)

    print("Done.")
    for r in rows:
        print(
            f"  frame={r[0]} ref={r[1]} path={r[2]:<28s} "
            f"px={r[3]:>5d} elems_in_brush={r[4]:>4d} min_edge={r[5]:>3d}"
        )


if __name__ == "__main__":
    main()
