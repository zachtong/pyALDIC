"""Diagnostic report: mesh creation and trimming behavior across tracking modes.

Monkey-patches mesh_setup and mark_inside to trace WHEN the mesh is
created/re-trimmed, with WHICH mask, and whether it changes across frames.

Runs 4 test scenarios:
  A) Accumulative — same mask all frames
  B) Accumulative — different masks per frame (should NOT matter, only ref mask used)
  C) Incremental (every_frame) — different masks per reference frame
  D) Custom schedule (every_2) — ref switch mid-sequence with new mask

Outputs: reports/report_mesh_behavior.pdf
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

# --- Setup paths ---
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "tests"))

from conftest import generate_speckle, apply_displacement_lagrangian
from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh, DICPara, FrameSchedule, GridxyROIRange,
)
from al_dic.core.pipeline import run_aldic
from al_dic.mesh.mark_inside import mark_inside as _real_mark_inside
from al_dic.mesh.mesh_setup import mesh_setup as _real_mesh_setup

# --- Report output ---
REPORT_DIR = PROJ / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# =====================================================================
# Synthetic data
# =====================================================================
IMG_H, IMG_W = 256, 256
N_FRAMES = 5  # 0=ref, 1..4=deformed
STEP = 16


def make_images(n: int = N_FRAMES) -> list[NDArray[np.float64]]:
    """Generate n synthetic images with progressive translation."""
    ref = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)
    images = [ref]
    for i in range(1, n):
        tx = 2.0 * i
        images.append(apply_displacement_lagrangian(
            ref,
            lambda x, y, _t=tx: np.full_like(x, _t),
            lambda x, y: np.zeros_like(x),
        ))
    return images


def make_mask_full() -> NDArray[np.float64]:
    """Full-image mask (no holes)."""
    return np.ones((IMG_H, IMG_W), dtype=np.float64)


def make_mask_with_hole(
    cx: float = 128.0, cy: float = 128.0, radius: float = 40.0,
) -> NDArray[np.float64]:
    """Mask with a circular hole."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
    mask[dist2 < radius ** 2] = 0.0
    return mask


# =====================================================================
# Instrumentation
# =====================================================================
class MeshTracker:
    """Records mesh_setup and mark_inside calls during a pipeline run."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.mesh_setup_calls: list[dict] = []
        self.mark_inside_calls: list[dict] = []
        self._call_order: list[str] = []

    def wrap_mesh_setup(self, x0, y0, para):
        result = _real_mesh_setup(x0, y0, para)
        self.mesh_setup_calls.append({
            "n_nodes": result.coordinates_fem.shape[0],
            "n_elements": result.elements_fem.shape[0],
            "x0_range": (float(x0.min()), float(x0.max())),
            "y0_range": (float(y0.min()), float(y0.max())),
        })
        self._call_order.append("mesh_setup")
        return result

    def wrap_mark_inside(self, coords, elems, mask):
        inside_idx, outside_idx = _real_mark_inside(coords, elems, mask)
        # Compute a hash of the mask to detect changes
        mask_hash = hash(mask.tobytes())
        n_hole_pixels = int((mask < 0.5).sum())
        self.mark_inside_calls.append({
            "n_elements_in": elems.shape[0],
            "n_inside": len(inside_idx),
            "n_outside": len(outside_idx),
            "mask_hash": mask_hash,
            "n_hole_pixels": n_hole_pixels,
        })
        self._call_order.append("mark_inside")
        return inside_idx, outside_idx

    def summary(self) -> str:
        lines = [f"=== {self.label} ==="]
        lines.append(f"  mesh_setup calls: {len(self.mesh_setup_calls)}")
        for i, c in enumerate(self.mesh_setup_calls):
            lines.append(
                f"    [{i}] {c['n_nodes']} nodes, {c['n_elements']} elements, "
                f"x={c['x0_range']}, y={c['y0_range']}"
            )
        lines.append(f"  mark_inside calls: {len(self.mark_inside_calls)}")
        seen_hashes: dict[int, int] = {}
        for i, c in enumerate(self.mark_inside_calls):
            h = c["mask_hash"]
            if h not in seen_hashes:
                seen_hashes[h] = len(seen_hashes)
            mask_id = seen_hashes[h]
            lines.append(
                f"    [{i}] {c['n_elements_in']} elems -> "
                f"{c['n_inside']} inside (trimmed), "
                f"{c['n_outside']} outside (kept)  "
                f"| mask_id={mask_id} hole_px={c['n_hole_pixels']}"
            )
        lines.append(f"  Call order: {' -> '.join(self._call_order)}")
        return "\n".join(lines)


def run_with_tracking(
    label: str,
    para: DICPara,
    images: list[NDArray[np.float64]],
    masks: list[NDArray[np.float64]],
) -> tuple[MeshTracker, object]:
    """Run pipeline with monkey-patched mesh tracking."""
    tracker = MeshTracker(label)
    with (
        patch(
            "al_dic.core.pipeline.mesh_setup",
            side_effect=tracker.wrap_mesh_setup,
        ),
        patch(
            "al_dic.core.pipeline.mark_inside",
            side_effect=tracker.wrap_mark_inside,
        ),
    ):
        result = run_aldic(
            para, images, masks,
            compute_strain=False,
        )
    return tracker, result


# =====================================================================
# Test scenarios
# =====================================================================
def make_base_para(**overrides) -> DICPara:
    return dicpara_default(
        winsize=32,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(
            gridx=(STEP, IMG_W - STEP),
            gridy=(STEP, IMG_H - STEP),
        ),
        admm_max_iter=2,
        show_plots=False,
        **overrides,
    )


def scenario_a() -> tuple[MeshTracker, object]:
    """Accumulative, same mask (full) for all frames."""
    images = make_images()
    masks = [make_mask_full()] * N_FRAMES
    para = make_base_para(reference_mode="accumulative")
    return run_with_tracking("A: Acc + same mask", para, images, masks)


def scenario_b() -> tuple[MeshTracker, object]:
    """Accumulative, DIFFERENT masks per frame (only ref mask should matter)."""
    images = make_images()
    masks = [
        make_mask_with_hole(128, 128, 40),  # frame 0 (ref): center hole
        make_mask_full(),                    # frame 1: no hole
        make_mask_with_hole(60, 60, 30),     # frame 2: different hole
        make_mask_full(),                    # frame 3: no hole
        make_mask_with_hole(200, 200, 50),   # frame 4: another hole
    ]
    para = make_base_para(reference_mode="accumulative")
    return run_with_tracking("B: Acc + different masks", para, images, masks)


def scenario_c() -> tuple[MeshTracker, object]:
    """Incremental (every_frame), different masks per reference."""
    images = make_images()
    masks = [
        make_mask_with_hole(128, 128, 40),  # frame 0 (ref for frame 1)
        make_mask_with_hole(100, 100, 30),  # frame 1 (ref for frame 2)
        make_mask_full(),                    # frame 2 (ref for frame 3) — hole disappears!
        make_mask_with_hole(180, 180, 35),  # frame 3 (ref for frame 4)
        make_mask_full(),                    # frame 4 (deformed only)
    ]
    # Incremental: ref_indices = (0, 1, 2, 3)
    para = make_base_para(reference_mode="incremental")
    return run_with_tracking("C: Inc (every_frame) + different masks", para, images, masks)


def scenario_d() -> tuple[MeshTracker, object]:
    """Custom schedule (every_2): ref switches at frame 2."""
    images = make_images()
    masks = [
        make_mask_with_hole(128, 128, 40),  # frame 0: center hole
        make_mask_full(),                    # frame 1
        make_mask_full(),                    # frame 2: NEW ref, NO hole
        make_mask_full(),                    # frame 3
        make_mask_full(),                    # frame 4
    ]
    # ref_indices[i] = ref for deformed frame i+1
    # frames: 0(ref), 1->0, 2->0, 3->2, 4->2
    schedule = FrameSchedule(ref_indices=(0, 0, 2, 2))
    para = make_base_para(
        reference_mode="incremental",  # overridden by schedule
        frame_schedule=schedule,
    )
    return run_with_tracking("D: Custom (every_2) + ref switch", para, images, masks)


# =====================================================================
# Report generation
# =====================================================================
def generate_report(results: list[tuple[MeshTracker, object]]) -> None:
    """Generate PDF report with mesh behavior analysis."""
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = REPORT_DIR / "report_mesh_behavior.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        # --- Page 1: Text summary ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Mesh Behavior Analysis: Accumulative vs Incremental", fontsize=14, fontweight="bold")

        text_parts = []
        for tracker, result in results:
            text_parts.append(tracker.summary())
            text_parts.append("")

        full_text = "\n".join(text_parts)
        ax.text(
            0.02, 0.95, full_text,
            transform=ax.transAxes, fontsize=8, fontfamily="monospace",
            verticalalignment="top",
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Visual comparison of mesh_setup / mark_inside counts ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        labels = [r[0].label for r in results]
        mesh_counts = [len(r[0].mesh_setup_calls) for r in results]
        mark_counts = [len(r[0].mark_inside_calls) for r in results]

        ax1 = axes[0]
        bars1 = ax1.barh(labels, mesh_counts, color="#3b82f6")
        ax1.set_xlabel("Number of mesh_setup() calls")
        ax1.set_title("Mesh Creation Frequency")
        for bar, count in zip(bars1, mesh_counts):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(count), va="center", fontsize=10, fontweight="bold")

        ax2 = axes[1]
        bars2 = ax2.barh(labels, mark_counts, color="#ef4444")
        ax2.set_xlabel("Number of mark_inside() calls")
        ax2.set_title("Mesh Trimming Frequency")
        for bar, count in zip(bars2, mark_counts):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(count), va="center", fontsize=10, fontweight="bold")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: Per-scenario mask hash timeline ---
        fig, axes = plt.subplots(len(results), 1, figsize=(11, 3 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for ax, (tracker, result) in zip(axes, results):
            if not tracker.mark_inside_calls:
                ax.text(0.5, 0.5, "No mark_inside calls", ha="center", va="center")
                ax.set_title(tracker.label)
                continue

            # Collect unique mask hashes
            hashes = [c["mask_hash"] for c in tracker.mark_inside_calls]
            unique_hashes = list(dict.fromkeys(hashes))
            hash_to_id = {h: i for i, h in enumerate(unique_hashes)}

            call_indices = list(range(len(tracker.mark_inside_calls)))
            mask_ids = [hash_to_id[h] for h in hashes]
            hole_sizes = [c["n_hole_pixels"] for c in tracker.mark_inside_calls]
            trimmed = [c["n_inside"] for c in tracker.mark_inside_calls]

            colors = plt.cm.Set2(np.array(mask_ids) / max(len(unique_hashes), 1))
            ax.bar(call_indices, trimmed, color=colors, edgecolor="black", linewidth=0.5)
            for i, (mid, hs, tr) in enumerate(zip(mask_ids, hole_sizes, trimmed)):
                ax.text(i, tr + 0.5, f"mask={mid}\nhole={hs}px\ntrim={tr}",
                        ha="center", va="bottom", fontsize=7)

            ax.set_xlabel("mark_inside() call index")
            ax.set_ylabel("Elements trimmed")
            ax.set_title(f"{tracker.label} — mask identity per trimming call")
            ax.set_xticks(call_indices)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: Analysis and conclusions ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Analysis & Conclusions", fontsize=14, fontweight="bold")

        # Build analysis text from actual data
        analysis_lines = []
        for tracker, result in results:
            analysis_lines.append(f"--- {tracker.label} ---")
            n_setup = len(tracker.mesh_setup_calls)
            n_mark = len(tracker.mark_inside_calls)
            analysis_lines.append(f"  mesh_setup called {n_setup}x, mark_inside called {n_mark}x")

            if n_mark > 0:
                unique_masks = len(set(c["mask_hash"] for c in tracker.mark_inside_calls))
                analysis_lines.append(f"  Unique masks used for trimming: {unique_masks}")

                if n_mark == 1:
                    analysis_lines.append("  -> Mesh built and trimmed ONCE (first frame only)")
                elif n_mark > 1 and unique_masks == 1:
                    analysis_lines.append("  -> Mesh re-trimmed multiple times but with SAME mask")
                elif n_mark > 1 and unique_masks > 1:
                    analysis_lines.append("  -> Mesh re-trimmed with DIFFERENT masks!")
                    trims = [c["n_inside"] for c in tracker.mark_inside_calls]
                    if len(set(trims)) > 1:
                        analysis_lines.append(
                            f"     Trim counts vary: {trims} -> "
                            f"element topology CHANGES between frames"
                        )
                    else:
                        analysis_lines.append(
                            f"     Trim counts identical ({trims[0]}) despite different masks"
                        )
            else:
                analysis_lines.append("  -> No trimming at all (mesh externally provided?)")

            # Check final mesh in result
            dic_mesh = result.dic_mesh
            analysis_lines.append(
                f"  Final mesh: {dic_mesh.coordinates_fem.shape[0]} nodes, "
                f"{dic_mesh.elements_fem.shape[0]} elements"
            )
            analysis_lines.append("")

        analysis_lines.append("=" * 60)
        analysis_lines.append("KEY FINDING: Does mesh change when reference frame changes?")
        analysis_lines.append("=" * 60)

        # Compare scenario A vs C
        tracker_a = results[0][0]
        tracker_c = results[2][0]
        analysis_lines.append(
            f"  Scenario A (acc, same mask):      "
            f"mesh_setup={len(tracker_a.mesh_setup_calls)}x, "
            f"mark_inside={len(tracker_a.mark_inside_calls)}x"
        )
        analysis_lines.append(
            f"  Scenario C (inc, different masks): "
            f"mesh_setup={len(tracker_c.mesh_setup_calls)}x, "
            f"mark_inside={len(tracker_c.mark_inside_calls)}x"
        )

        if len(tracker_c.mesh_setup_calls) > 1:
            analysis_lines.append(
                "  -> INCREMENTAL MODE REBUILDS MESH on ref switch!"
            )
        else:
            analysis_lines.append(
                "  -> INCREMENTAL MODE DOES NOT REBUILD MESH."
            )
            analysis_lines.append(
                "     Even with different masks per reference frame,"
            )
            analysis_lines.append(
                "     the mesh is created once and never re-trimmed."
            )

        # Check scenario D
        tracker_d = results[3][0]
        analysis_lines.append("")
        analysis_lines.append(
            f"  Scenario D (custom schedule, ref switch at frame 2): "
            f"mesh_setup={len(tracker_d.mesh_setup_calls)}x, "
            f"mark_inside={len(tracker_d.mark_inside_calls)}x"
        )
        if len(tracker_d.mark_inside_calls) > 0:
            unique_masks_d = len(set(c["mask_hash"] for c in tracker_d.mark_inside_calls))
            analysis_lines.append(f"     Unique masks in trimming: {unique_masks_d}")

        analysis_text = "\n".join(analysis_lines)
        ax.text(
            0.02, 0.95, analysis_text,
            transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
            verticalalignment="top",
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport saved to: {pdf_path}")


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    print("Running mesh behavior diagnostics...")
    print(f"  Image size: {IMG_H}x{IMG_W}, step={STEP}, {N_FRAMES} frames\n")

    all_results = []

    for name, scenario_fn in [
        ("A", scenario_a),
        ("B", scenario_b),
        ("C", scenario_c),
        ("D", scenario_d),
    ]:
        print(f"--- Running scenario {name} ---")
        tracker, result = scenario_fn()
        all_results.append((tracker, result))
        print(tracker.summary())
        print()

    generate_report(all_results)


if __name__ == "__main__":
    main()
