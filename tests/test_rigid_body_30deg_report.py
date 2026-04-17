#!/usr/bin/env python
"""Large rotation (30 deg) focus test: accumulative vs incremental.

Tests whether accumulative mode with LARGE FFT search region can match
incremental mode accuracy for 30 deg rigid body rotation.

Scope (deliberately narrow):
  - Only one motion: 30 deg rotation (6 frames x 5 deg each)
  - Only Green-Lagrangian strain (exact zero for rigid rotation)
  - Two strain methods: Plane fitting (VSG=41), FEM nodal
  - Two modes with different FFT search settings:
      * accumulative: FFT=130 (enough for ~100 px max displacement)
      * incremental:  FFT=30  (only needs ~17 px per step)

Report saved to al-dic/reports/rigid_body_30deg_test.pdf
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tests"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from conftest import apply_displacement_lagrangian, generate_speckle
from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import DICPara, GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.strain.compute_strain import compute_strain
from al_dic.utils.region_analysis import precompute_node_regions

# ── Geometry ─────────────────────────────────────────────────────────
IMG_H = IMG_W = 512
STEP = 8
WINSIZE = 32
MARGIN = 24
MASK_RADIUS = 200
CX, CY = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0
SIGMA = 3.0
SEED = 42

# FFT search: accumulative needs ~100 px for 30 deg at r=200
FFT_ACCUM = 130
FFT_INCR = 30

# Strain methods (Green-Lagrangian only)
METHODS = [
    ("PF-GL", 2, "Plane fitting (VSG=41)"),
    ("FEM-GL", 3, "FEM nodal"),
]

REPORT_DIR = BASE / "al-dic" / "reports"


# ── Helpers ──────────────────────────────────────────────────────────

def circular_mask(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.float64)


def rigid_rot_funcs(angle_deg, cx, cy):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    def u(x, y):
        return (x - cx) * (c - 1) - (y - cy) * s
    def v(x, y):
        return (x - cx) * s + (y - cy) * (c - 1)
    return u, v


def nodes_in_mask(coords, mask):
    x = np.clip(np.round(coords[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.round(coords[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[y, x] > 0


# ── Pipeline runner ──────────────────────────────────────────────────

def make_para(mode, fft_search):
    return dicpara_default(
        winsize=WINSIZE,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(
            gridx=(MARGIN, IMG_W - 1 - MARGIN),
            gridy=(MARGIN, IMG_H - 1 - MARGIN),
        ),
        reference_mode=mode,
        size_of_fft_search_region=fft_search,
        show_plots=False,
        icgn_max_iter=100,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        strain_smoothness=0.0,
        smoothness=0.0,
    )


def generate_images(ref, angle_schedule):
    images = [ref.copy()]
    for angle in angle_schedule:
        u, v = rigid_rot_funcs(angle, CX, CY)
        images.append(apply_displacement_lagrangian(ref, u, v))
    return images


def run_pipeline(ref, mask, angle_schedule, mode, fft_search):
    images = generate_images(ref, angle_schedule)
    masks = [mask.copy() for _ in images]
    para = make_para(mode, fft_search)
    print(f"    mode={mode:13s} FFT={fft_search:3d}  ({len(images)} frames)...", flush=True)
    t0 = time.perf_counter()
    try:
        result = run_aldic(para, images, masks,
                           compute_strain=False,
                           progress_fn=lambda f, m: None)
        dt = time.perf_counter() - t0
        print(f"      done in {dt:.1f}s", flush=True)
        return result
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"      FAILED ({dt:.1f}s): {type(exc).__name__}: {exc}", flush=True)
        return None


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_frame(result, frame_idx, mask, angle_deg):
    """Return metrics for one frame, both strain methods."""
    fd = result.result_disp[frame_idx]
    if fd is None:
        return None
    U = fd.U_accum if fd.U_accum is not None else fd.U
    coords = result.dic_mesh.coordinates_fem
    inside = nodes_in_mask(coords, mask)
    region_map = precompute_node_regions(coords, mask, (IMG_H, IMG_W))

    # Ground truth displacement
    u_func, v_func = rigid_rot_funcs(angle_deg, CX, CY)
    gt_u = u_func(coords[:, 0], coords[:, 1])
    gt_v = v_func(coords[:, 0], coords[:, 1])
    u_comp = U[0::2]
    v_comp = U[1::2]
    disp_err = np.sqrt((u_comp[inside] - gt_u[inside]) ** 2 +
                        (v_comp[inside] - gt_v[inside]) ** 2)
    disp_rmse = float(np.sqrt(np.mean(disp_err ** 2)))

    results = {"disp_rmse": disp_rmse, "angle_deg": angle_deg}
    for label, method, desc in METHODS:
        para_s = replace(
            result.dic_para,
            method_to_compute_strain=method,
            strain_plane_fit_rad=20.0,
            strain_smoothness=0.0,
            strain_type=2,  # Green-Lagrangian
            img_ref_mask=mask,
        )
        sr = compute_strain(result.dic_mesh, para_s, U, region_map)
        valid = inside & np.isfinite(sr.strain_exx) & np.isfinite(sr.strain_rotation)

        # For G-L, expected strain is exactly zero for rigid rotation
        exx_rmse = float(np.sqrt(np.mean(sr.strain_exx[valid] ** 2))) if valid.any() else np.nan
        eyy_rmse = float(np.sqrt(np.mean(sr.strain_eyy[valid] ** 2))) if valid.any() else np.nan
        exy_rmse = float(np.sqrt(np.mean(sr.strain_exy[valid] ** 2))) if valid.any() else np.nan
        rot_mean = float(np.mean(sr.strain_rotation[valid])) if valid.any() else np.nan
        rot_std = float(np.std(sr.strain_rotation[valid])) if valid.any() else np.nan
        rot_err = rot_mean - angle_deg if np.isfinite(rot_mean) else np.nan

        # For field maps
        exx_field = np.full_like(sr.strain_exx, np.nan)
        exx_field[valid] = sr.strain_exx[valid]
        rot_field = np.full_like(sr.strain_rotation, np.nan)
        rot_field[valid] = sr.strain_rotation[valid]

        results[label] = {
            "desc": desc,
            "exx_rmse": exx_rmse,
            "eyy_rmse": eyy_rmse,
            "exy_rmse": exy_rmse,
            "rot_mean": rot_mean,
            "rot_std": rot_std,
            "rot_err": rot_err,
            "exx_field": exx_field,
            "rot_field": rot_field,
        }
    return results


# ── Report pages ─────────────────────────────────────────────────────

def _page_title(pdf, params):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.78, "Large Rotation (30\u00b0) Focus Test",
             fontsize=24, ha="center", weight="bold")
    fig.text(0.5, 0.72, "Accumulative (FFT=130) vs Incremental (FFT=30)",
             fontsize=14, ha="center", color="0.3")
    fig.text(0.5, 0.66, "Green-Lagrangian strain only",
             fontsize=14, ha="center", color="0.3")
    txt = (
        f"Image:           {IMG_H} x {IMG_W}\n"
        f"Mask radius:     {MASK_RADIUS} px  (centered)\n"
        f"Mesh step:       {STEP} px\n"
        f"Winsize:         {WINSIZE} px\n"
        f"Speckle sigma:   {SIGMA}\n\n"
        f"Angle schedule:  5, 10, 15, 20, 25, 30 deg  (6 frames)\n"
        f"Max edge displacement at 30 deg: ~{MASK_RADIUS * np.sin(np.radians(30)):.0f} px\n\n"
        f"Accumulative FFT search: {FFT_ACCUM} px  (safe for ~{FFT_ACCUM} px displacement)\n"
        f"Incremental  FFT search: {FFT_INCR} px  (only needs ~17 px per 5-deg step)\n\n"
        f"Strain methods tested (both with Green-Lagrangian):\n"
        f"  1. Plane fitting (VSG=41, radius=20 px)\n"
        f"  2. FEM nodal (shape function derivatives)\n\n"
        f"For rigid rotation, Green-Lagrangian strain is EXACTLY ZERO:\n"
        f"  E = (F\u1d40 F - I) / 2 = (R\u1d40 R - I) / 2 = 0\n"
        f"Any nonzero strain RMSE is purely numerical / IC-GN noise."
    )
    fig.text(0.12, 0.42, txt, fontsize=11, va="top", family="monospace", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _page_rotation(pdf, results_by_mode, angle_schedule):
    """Rotation angle tracking."""
    frame_nums = list(range(1, len(angle_schedule) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame_nums, angle_schedule, "k--", lw=2, label="Ground truth", zorder=10)

    styles = {"accumulative": "o-", "incremental": "s-"}
    colors = {"PF-GL": "tab:blue", "FEM-GL": "tab:red"}

    for mode, frames in results_by_mode.items():
        if frames is None:
            continue
        for label, _, _ in METHODS:
            rot_vals = [fr[label]["rot_mean"] if fr is not None else np.nan
                        for fr in frames]
            ax.plot(frame_nums, rot_vals, styles[mode],
                    color=colors[label],
                    label=f"{label} ({mode})",
                    markersize=6, lw=1.5, alpha=0.85)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Rotation angle (\u00b0)")
    ax.set_title("Recovered Rotation Angle vs Frame", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_strain_rmse(pdf, results_by_mode, angle_schedule):
    """Strain RMSE per frame (log scale)."""
    frame_nums = list(range(1, len(angle_schedule) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Green-Lagrangian Strain RMSE vs Frame (expected: 0)",
                 fontsize=13, weight="bold")

    metric_axes = [
        ("exx_rmse", "exx RMSE (should be 0)"),
        ("rot_err", "Rotation angle error (\u00b0)"),
    ]

    styles = {"accumulative": "o-", "incremental": "s-"}
    colors = {"PF-GL": "tab:blue", "FEM-GL": "tab:red"}

    for ax_idx, (key, ylabel) in enumerate(metric_axes):
        ax = axes[ax_idx]
        for mode, frames in results_by_mode.items():
            if frames is None:
                continue
            for label, _, _ in METHODS:
                vals = []
                for fr in frames:
                    if fr is None:
                        vals.append(np.nan)
                    else:
                        v = fr[label][key]
                        if key == "rot_err":
                            v = abs(v)
                        vals.append(v)
                ax.plot(frame_nums, vals, styles[mode],
                        color=colors[label],
                        label=f"{label} ({mode})",
                        markersize=5, lw=1.3, alpha=0.85)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _page_fields_last_frame(pdf, results_by_mode, pipeline_results, mask):
    """Field maps for the last frame (30 deg) across 4 combinations."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Strain & Rotation Fields at Frame 6 (30\u00b0)",
                 fontsize=14, weight="bold")

    for col_idx, (mode, frames) in enumerate([
        ("accumulative", results_by_mode["accumulative"]),
        ("incremental", results_by_mode["incremental"]),
    ]):
        result = pipeline_results.get(mode)
        if result is None or frames is None:
            continue
        fr = frames[-1]
        if fr is None:
            continue
        coords = result.dic_mesh.coordinates_fem
        inside = nodes_in_mask(coords, mask)
        x = coords[inside, 0]
        y = coords[inside, 1]
        angle = fr["angle_deg"]

        for row_idx, (label, _, desc) in enumerate(METHODS):
            sm = fr[label]

            # Column layout: mode (2) x method (2) = 4 cols
            # Row 0: exx, Row 1: rotation
            col = col_idx * 2 + row_idx

            # exx field
            ax_e = axes[0, col]
            exx = sm["exx_field"][inside]
            vlim = max(abs(np.nanpercentile(exx, 1)),
                       abs(np.nanpercentile(exx, 99)),
                       1e-6)
            sc = ax_e.scatter(x, y, c=exx, cmap="RdBu_r",
                              vmin=-vlim, vmax=vlim,
                              s=2, edgecolors="none")
            ax_e.set_title(f"{mode[:5].capitalize()} / {label}\n"
                           f"exx  RMSE={sm['exx_rmse']:.2e}",
                           fontsize=9)
            ax_e.set_aspect("equal")
            ax_e.invert_yaxis()
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            plt.colorbar(sc, ax=ax_e, shrink=0.7, format="%.1e")

            # rotation field
            ax_r = axes[1, col]
            rot = sm["rot_field"][inside]
            sc = ax_r.scatter(x, y, c=rot, cmap="coolwarm",
                              vmin=angle - 1.5, vmax=angle + 1.5,
                              s=2, edgecolors="none")
            ax_r.set_title(f"Rotation  mean={sm['rot_mean']:.3f}\u00b0\n"
                           f"err={sm['rot_err']:+.4f}\u00b0",
                           fontsize=9)
            ax_r.set_aspect("equal")
            ax_r.invert_yaxis()
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            plt.colorbar(sc, ax=ax_r, shrink=0.7, format="%.2f")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_summary_table(pdf, results_by_mode):
    """Summary of last-frame metrics."""
    fig = plt.figure(figsize=(11, 4.5))
    fig.text(0.5, 0.93, "Summary: Frame 6 (30\u00b0) Metrics",
             fontsize=14, ha="center", weight="bold")

    headers = ["Mode", "Method", "Disp RMSE\n(px)",
               "exx RMSE\n(strain)", "eyy RMSE\n(strain)", "exy RMSE\n(strain)",
               "Rot mean\n(\u00b0)", "Rot err\n(\u00b0)"]
    rows = []
    for mode in ["accumulative", "incremental"]:
        frames = results_by_mode.get(mode)
        if frames is None:
            for label, _, _ in METHODS:
                rows.append([mode, label, "FAIL", "-", "-", "-", "-", "-"])
            continue
        fr = frames[-1]
        if fr is None:
            for label, _, _ in METHODS:
                rows.append([mode, label, "FAIL", "-", "-", "-", "-", "-"])
            continue
        for label, _, _ in METHODS:
            sm = fr[label]
            rows.append([
                mode, label,
                f"{fr['disp_rmse']:.4f}",
                f"{sm['exx_rmse']:.2e}",
                f"{sm['eyy_rmse']:.2e}",
                f"{sm['exy_rmse']:.2e}",
                f"{sm['rot_mean']:+.4f}",
                f"{sm['rot_err']:+.4f}",
            ])

    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#d5e8f0")
        table[0, j].set_text_props(weight="bold")
    for i, row in enumerate(rows, start=1):
        if "FAIL" in row:
            for j in range(len(headers)):
                table[i, j].set_facecolor("#fdd")
    pdf.savefig(fig)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Large Rotation (30 deg) Focus Test")
    print("=" * 60)

    print("\n[1/3] Generating reference speckle...")
    ref = generate_speckle(IMG_H, IMG_W, sigma=SIGMA, seed=SEED)
    mask = circular_mask(IMG_H, IMG_W, CX, CY, MASK_RADIUS)

    angle_schedule = [5, 10, 15, 20, 25, 30]  # 6 frames

    print(f"\n[2/3] Running pipelines...")
    results_by_mode = {}
    pipeline_results = {}

    for mode, fft in [("accumulative", FFT_ACCUM), ("incremental", FFT_INCR)]:
        res = run_pipeline(ref, mask, angle_schedule, mode, fft)
        pipeline_results[mode] = res
        if res is None:
            results_by_mode[mode] = None
            continue

        frames = []
        for i, angle in enumerate(angle_schedule):
            fr = analyze_frame(res, i, mask, angle)
            frames.append(fr)
            if fr is not None:
                pf = fr["PF-GL"]
                fem = fr["FEM-GL"]
                print(f"      frame {i+1} ({angle:2d}\u00b0): "
                      f"disp={fr['disp_rmse']:.3f}px  "
                      f"PF rot={pf['rot_mean']:+.3f}\u00b0 (err {pf['rot_err']:+.4f}) exx={pf['exx_rmse']:.2e}  |  "
                      f"FEM rot={fem['rot_mean']:+.3f}\u00b0 (err {fem['rot_err']:+.4f}) exx={fem['exx_rmse']:.2e}",
                      flush=True)
        results_by_mode[mode] = frames

    print(f"\n[3/3] Generating PDF report...")
    report_path = REPORT_DIR / "rigid_body_30deg_test.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(report_path)) as pdf:
        _page_title(pdf, None)
        _page_rotation(pdf, results_by_mode, angle_schedule)
        _page_strain_rmse(pdf, results_by_mode, angle_schedule)
        _page_fields_last_frame(pdf, results_by_mode, pipeline_results, mask)
        _page_summary_table(pdf, results_by_mode)

    print(f"\n[DONE] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
