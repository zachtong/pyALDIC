#!/usr/bin/env python
"""Rigid body motion test: strain and rotation angle accuracy.

Generates synthetic speckle images with known rigid body transformations,
runs pyALDIC in both modes (accumulative, incremental) with both strain
methods (plane fitting, FEM) and both strain measures (infinitesimal,
Green-Lagrangian).

Verifies:
  - Strain approx 0 for all rigid body motions
  - Rotation angle matches ground truth
  - Large rotation (30 deg) requires incremental mode
  - Green-Lagrangian strain gives exact zero for rigid rotation
    (infinitesimal gives cos(theta)-1 != 0)

PDF report saved to al-dic/reports/
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

# ── Constants ────────────────────────────────────────────────────────
IMG_H = IMG_W = 512
STEP = 8
WINSIZE = 32
MARGIN = 24
MASK_RADIUS = 200
CX, CY = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0
FFT_SEARCH = 30
SIGMA = 3.0
SEED = 42
REPORT_DIR = BASE / "al-dic" / "reports"

# Strain method / type combos
STRAIN_COMBOS = [
    ("PF-Inf", 2, 0, "Plane fitting, Infinitesimal"),
    ("PF-GL",  2, 2, "Plane fitting, Green-Lagrangian"),
    ("FEM-Inf", 3, 0, "FEM nodal, Infinitesimal"),
    ("FEM-GL", 3, 2, "FEM nodal, Green-Lagrangian"),
]


# ── Helpers ──────────────────────────────────────────────────────────

def circular_mask(h: int, w: int, cx: float, cy: float, r: float):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.float64)


def rigid_body_funcs(angle_deg: float, tx: float, ty: float, cx: float, cy: float):
    """Return (u_func, v_func) for rigid translation + rotation."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    def u(x, y):
        return (x - cx) * (cos_t - 1) - (y - cy) * sin_t + tx
    def v(x, y):
        return (x - cx) * sin_t + (y - cy) * (cos_t - 1) + ty
    return u, v


def nodes_in_mask(coords, mask):
    """Boolean array of nodes falling inside the mask."""
    x = np.clip(np.round(coords[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.round(coords[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[y, x] > 0


def expected_strain_inf(angle_deg: float):
    """Expected infinitesimal strain for rigid rotation."""
    theta = np.radians(angle_deg)
    return np.cos(theta) - 1.0  # exx = eyy = cos(theta) - 1, exy = 0


# ── Test case definitions ────────────────────────────────────────────

def build_cases():
    """Each case: name, label, list of (angle_deg, tx, ty) for each deformed frame."""
    return [
        {
            "name": "translation",
            "label": "Pure Translation (10, 5) px",
            "frames": [(0, 10, 5), (0, 20, 10)],
        },
        {
            "name": "rotation_2deg",
            "label": "Pure Rotation 2\u00b0/frame",
            "frames": [(2, 0, 0), (4, 0, 0)],
        },
        {
            "name": "rotation_5deg",
            "label": "Pure Rotation 5\u00b0/frame",
            "frames": [(5, 0, 0), (10, 0, 0)],
        },
        {
            "name": "rotation_30deg",
            "label": "Pure Rotation 30\u00b0 (5\u00b0/step \u00d7 6)",
            "frames": [(5 * i, 0, 0) for i in range(1, 7)],
        },
    ]


# ── Pipeline runner ──────────────────────────────────────────────────

def make_para(mode: str) -> DICPara:
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
        size_of_fft_search_region=FFT_SEARCH,
        show_plots=False,
        icgn_max_iter=50,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        strain_smoothness=0.0,
        smoothness=0.0,
    )


def generate_images(ref, case):
    """Generate deformed images via Lagrangian warp."""
    images = [ref.copy()]
    for angle, tx, ty in case["frames"]:
        u_func, v_func = rigid_body_funcs(angle, tx, ty, CX, CY)
        images.append(apply_displacement_lagrangian(ref, u_func, v_func))
    return images


def run_pipeline(case, mode, ref, mask):
    """Run pipeline for one case x mode, return result or None on failure."""
    images = generate_images(ref, case)
    masks = [mask.copy() for _ in images]
    para = make_para(mode)
    print(f"    Running pipeline ({len(images)} frames)...", flush=True)
    t0 = time.perf_counter()
    try:
        result = run_aldic(
            para, images, masks,
            compute_strain=False,
            progress_fn=lambda f, m: None,
        )
        dt = time.perf_counter() - t0
        print(f"    Pipeline done in {dt:.1f}s", flush=True)
        return result
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"    Pipeline FAILED ({dt:.1f}s): {type(exc).__name__}: {exc}", flush=True)
        return None


# ── Strain analysis ──────────────────────────────────────────────────

def analyze_frame(result, frame_idx, para_base, mask, angle_deg):
    """Compute strain with all 4 combos and return metrics dict."""
    fd = result.result_disp[frame_idx]
    U = fd.U_accum if fd.U_accum is not None else fd.U
    coords = result.dic_mesh.coordinates_fem
    mesh = result.dic_mesh
    inside = nodes_in_mask(coords, mask)

    region_map = precompute_node_regions(coords, mask, (IMG_H, IMG_W))

    # Ground truth displacement
    u_func, v_func = rigid_body_funcs(angle_deg, 0, 0, CX, CY)
    # Also add translation if present (we'll pass angle and tx/ty from case)
    # Actually, ground truth comes from the case frame definition
    gt_u_vals = u_func(coords[:, 0], coords[:, 1])
    gt_v_vals = v_func(coords[:, 0], coords[:, 1])

    u_comp = U[0::2]
    v_comp = U[1::2]
    disp_err = np.sqrt((u_comp[inside] - gt_u_vals[inside]) ** 2 +
                        (v_comp[inside] - gt_v_vals[inside]) ** 2)
    disp_rmse = float(np.sqrt(np.mean(disp_err ** 2)))

    strain_metrics = {}
    for label, method, stype, desc in STRAIN_COMBOS:
        para_s = replace(
            para_base,
            method_to_compute_strain=method,
            strain_plane_fit_rad=20.0,
            strain_smoothness=0.0,
            strain_type=stype,
            img_ref_mask=mask,
        )
        sr = compute_strain(mesh, para_s, U, region_map)

        # Expected strain
        if stype == 0:  # infinitesimal
            exp_exx = expected_strain_inf(angle_deg)
            exp_eyy = exp_exx
        else:  # Green-Lagrangian
            exp_exx = 0.0
            exp_eyy = 0.0

        valid = inside & np.isfinite(sr.strain_exx) & np.isfinite(sr.strain_rotation)

        exx_err = float(np.sqrt(np.mean((sr.strain_exx[valid] - exp_exx) ** 2)))
        eyy_err = float(np.sqrt(np.mean((sr.strain_eyy[valid] - exp_eyy) ** 2)))
        exy_err = float(np.sqrt(np.mean(sr.strain_exy[valid] ** 2)))
        rot_mean = float(np.mean(sr.strain_rotation[valid]))
        rot_std = float(np.std(sr.strain_rotation[valid]))
        rot_err = rot_mean - angle_deg

        strain_metrics[label] = {
            "desc": desc,
            "exx_rmse": exx_err,
            "eyy_rmse": eyy_err,
            "exy_rmse": exy_err,
            "rot_mean": rot_mean,
            "rot_std": rot_std,
            "rot_err": rot_err,
            "exx_field": sr.strain_exx.copy(),
            "rot_field": sr.strain_rotation.copy(),
        }

    return {
        "disp_rmse": disp_rmse,
        "angle_deg": angle_deg,
        "strain": strain_metrics,
    }


def analyze_result(result, case, para_base, mask):
    """Analyze all frames in a pipeline result."""
    frames = []
    for i, (angle, tx, ty) in enumerate(case["frames"]):
        if i >= len(result.result_disp) or result.result_disp[i] is None:
            frames.append(None)
            continue
        # For ground truth, we need the full (angle, tx, ty) for displacement
        # but analyze_frame uses angle_deg for strain/rotation expectation
        # We need a custom GT func that includes translation
        fd = result.result_disp[i]
        U = fd.U_accum if fd.U_accum is not None else fd.U
        coords = result.dic_mesh.coordinates_fem
        inside = nodes_in_mask(coords, mask)
        region_map = precompute_node_regions(coords, mask, (IMG_H, IMG_W))

        # Full ground truth (rotation + translation)
        u_func, v_func = rigid_body_funcs(angle, tx, ty, CX, CY)
        gt_u = u_func(coords[:, 0], coords[:, 1])
        gt_v = v_func(coords[:, 0], coords[:, 1])
        u_comp = U[0::2]
        v_comp = U[1::2]
        disp_err = np.sqrt((u_comp[inside] - gt_u[inside]) ** 2 +
                            (v_comp[inside] - gt_v[inside]) ** 2)
        disp_rmse = float(np.sqrt(np.mean(disp_err ** 2)))

        strain_metrics = {}
        for label, method, stype, desc in STRAIN_COMBOS:
            para_s = replace(
                para_base,
                method_to_compute_strain=method,
                strain_plane_fit_rad=20.0,
                strain_smoothness=0.0,
                strain_type=stype,
                img_ref_mask=mask,
            )
            sr = compute_strain(result.dic_mesh, para_s, U, region_map)
            valid = inside & np.isfinite(sr.strain_exx) & np.isfinite(sr.strain_rotation)

            if stype == 0:
                exp_exx = expected_strain_inf(angle)
                exp_eyy = exp_exx
            else:
                exp_exx = 0.0
                exp_eyy = 0.0

            exx_err = float(np.sqrt(np.mean((sr.strain_exx[valid] - exp_exx) ** 2))) if valid.any() else np.nan
            eyy_err = float(np.sqrt(np.mean((sr.strain_eyy[valid] - exp_eyy) ** 2))) if valid.any() else np.nan
            exy_err = float(np.sqrt(np.mean(sr.strain_exy[valid] ** 2))) if valid.any() else np.nan
            rot_mean = float(np.mean(sr.strain_rotation[valid])) if valid.any() else np.nan
            rot_std = float(np.std(sr.strain_rotation[valid])) if valid.any() else np.nan
            rot_err = rot_mean - angle if np.isfinite(rot_mean) else np.nan

            strain_metrics[label] = {
                "desc": desc,
                "exx_rmse": exx_err,
                "eyy_rmse": eyy_err,
                "exy_rmse": exy_err,
                "rot_mean": rot_mean,
                "rot_std": rot_std,
                "rot_err": rot_err,
                "exx_field": sr.strain_exx.copy(),
                "rot_field": sr.strain_rotation.copy(),
            }

        frames.append({
            "disp_rmse": disp_rmse,
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
            "strain": strain_metrics,
        })
    return frames


# ── Report generation ────────────────────────────────────────────────

def _add_title_page(pdf, all_data):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.75, "Rigid Body Motion", fontsize=28, ha="center", weight="bold")
    fig.text(0.5, 0.67, "Strain & Rotation Angle Accuracy Test", fontsize=20, ha="center")
    fig.text(0.5, 0.55, f"Image: {IMG_H}\u00d7{IMG_W},  step={STEP},  winsize={WINSIZE},  "
             f"mask r={MASK_RADIUS} px", fontsize=12, ha="center", color="0.4")
    fig.text(0.5, 0.50, f"FFT search={FFT_SEARCH},  speckle \u03c3={SIGMA}",
             fontsize=12, ha="center", color="0.4")

    cases_text = "Test Cases:\n"
    for cd in all_data:
        cases_text += f"  \u2022 {cd['case']['label']}\n"
    cases_text += "\nModes: Accumulative, Incremental"
    cases_text += "\nStrain Methods: Plane Fitting, FEM Nodal"
    cases_text += "\nStrain Types: Infinitesimal, Green-Lagrangian"
    fig.text(0.15, 0.15, cases_text, fontsize=11, va="bottom", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _add_case_rotation_page(pdf, case_data):
    """Rotation angle vs frame index for incremental vs accumulative."""
    case = case_data["case"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"{case['label']}  \u2014  Rotation Angle Tracking", fontsize=14, weight="bold")

    angles_gt = [f[0] for f in case["frames"]]
    frame_nums = list(range(1, len(angles_gt) + 1))

    for ax_idx, mode in enumerate(["accumulative", "incremental"]):
        ax = axes[ax_idx]
        ax.set_title(f"{mode.capitalize()} mode", fontsize=12)
        ax.plot(frame_nums, angles_gt, "k--", label="Ground truth", lw=2, zorder=10)

        key = f"{case['name']}_{mode}"
        frames = case_data.get(key)
        if frames is None:
            ax.text(0.5, 0.5, "Pipeline failed", transform=ax.transAxes,
                    ha="center", fontsize=14, color="red")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Rotation angle (\u00b0)")
            continue

        for label, _, _, _ in STRAIN_COMBOS:
            rot_vals = []
            for fr in frames:
                if fr is None:
                    rot_vals.append(np.nan)
                else:
                    rot_vals.append(fr["strain"][label]["rot_mean"])
            ax.plot(frame_nums, rot_vals, "o-", label=label, markersize=4, lw=1.5)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Rotation angle (\u00b0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_case_strain_page(pdf, case_data):
    """Strain RMSE vs frame for all combos."""
    case = case_data["case"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{case['label']}  \u2014  Strain RMSE vs Frame", fontsize=14, weight="bold")

    angles_gt = [f[0] for f in case["frames"]]
    frame_nums = list(range(1, len(angles_gt) + 1))

    plot_items = [
        ("exx_rmse", "exx RMSE"),
        ("eyy_rmse", "eyy RMSE"),
        ("exy_rmse", "exy RMSE"),
        ("rot_err", "Rotation Error (\u00b0)"),
    ]

    for mode_idx, mode in enumerate(["accumulative", "incremental"]):
        key = f"{case['name']}_{mode}"
        frames = case_data.get(key)

        for pi, (metric_key, metric_label) in enumerate(plot_items):
            ax = axes[pi // 2, pi % 2]
            if mode_idx == 0:
                ax.set_title(metric_label, fontsize=11)

            if frames is None:
                continue

            for label, _, _, _ in STRAIN_COMBOS:
                vals = []
                for fr in frames:
                    if fr is None:
                        vals.append(np.nan)
                    else:
                        v = fr["strain"][label][metric_key]
                        if metric_key == "rot_err":
                            v = abs(v)
                        vals.append(v)

                style = "-" if mode == "incremental" else "--"
                ax.plot(frame_nums, vals, f"o{style}", label=f"{label} ({mode[:3]})",
                        markersize=3, lw=1.2)

            ax.set_xlabel("Frame")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _add_case_field_page(pdf, case_data, mode):
    """Show exx and rotation field maps for the last valid frame."""
    case = case_data["case"]
    key = f"{case['name']}_{mode}"
    frames = case_data.get(key)
    if frames is None:
        return
    result = case_data.get(f"{key}_result")
    if result is None:
        return

    # Find last valid frame
    last_valid = None
    for i in range(len(frames) - 1, -1, -1):
        if frames[i] is not None and frames[i]["disp_rmse"] < 5.0:
            last_valid = i
            break
    if last_valid is None:
        return

    fr = frames[last_valid]
    coords = result.dic_mesh.coordinates_fem
    mask = circular_mask(IMG_H, IMG_W, CX, CY, MASK_RADIUS)
    inside = nodes_in_mask(coords, mask)
    x, y = coords[inside, 0], coords[inside, 1]
    angle = fr["angle_deg"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"{case['label']}  \u2014  {mode.capitalize()} mode, "
                 f"Frame {last_valid + 1} ({angle:.0f}\u00b0)",
                 fontsize=13, weight="bold")

    for col, (label, _, _, desc) in enumerate(STRAIN_COMBOS):
        sm = fr["strain"][label]
        exx = sm["exx_field"][inside]
        rot = sm["rot_field"][inside]

        # exx field
        ax = axes[0, col]
        vmax_exx = max(abs(np.nanpercentile(exx, 1)), abs(np.nanpercentile(exx, 99)), 1e-6)
        sc = ax.scatter(x, y, c=exx, cmap="RdBu_r", vmin=-vmax_exx, vmax=vmax_exx,
                        s=2, edgecolors="none")
        ax.set_title(f"{label}\nexx (RMSE={sm['exx_rmse']:.2e})", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.6, format="%.1e")

        # rotation field
        ax = axes[1, col]
        sc = ax.scatter(x, y, c=rot, cmap="coolwarm",
                        vmin=angle - 1, vmax=angle + 1,
                        s=2, edgecolors="none")
        ax.set_title(f"Rotation (err={sm['rot_err']:+.3f}\u00b0)", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.6, format="%.2f")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_summary_table(pdf, all_data):
    """Final summary table: last-frame metrics for all cases x modes."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.96, "Summary: Last-Frame Metrics", fontsize=16, ha="center", weight="bold")

    # Collect rows
    headers = ["Case", "Mode", "Frame", "Disp\nRMSE (px)",
               "PF-Inf\nexx RMSE", "PF-GL\nexx RMSE",
               "FEM-Inf\nexx RMSE", "FEM-GL\nexx RMSE",
               "PF-GL\nRot Err (\u00b0)"]
    rows = []
    for cd in all_data:
        case = cd["case"]
        for mode in ["accumulative", "incremental"]:
            key = f"{case['name']}_{mode}"
            frames = cd.get(key)
            if frames is None:
                rows.append([case["label"], mode, "-", "FAIL",
                             "-", "-", "-", "-", "-"])
                continue

            # Find last frame with reasonable displacement
            last = None
            for i in range(len(frames) - 1, -1, -1):
                if frames[i] is not None and frames[i]["disp_rmse"] < 5.0:
                    last = i
                    break
            if last is None:
                rows.append([case["label"], mode, "-", "FAIL",
                             "-", "-", "-", "-", "-"])
                continue

            fr = frames[last]
            angle = fr["angle_deg"]
            rows.append([
                case["label"][:25],
                mode[:5],
                f"{last + 1} ({angle:.0f}\u00b0)" if angle != 0 else f"{last + 1}",
                f"{fr['disp_rmse']:.4f}",
                f"{fr['strain']['PF-Inf']['exx_rmse']:.2e}",
                f"{fr['strain']['PF-GL']['exx_rmse']:.2e}",
                f"{fr['strain']['FEM-Inf']['exx_rmse']:.2e}",
                f"{fr['strain']['FEM-GL']['exx_rmse']:.2e}",
                f"{fr['strain']['PF-GL']['rot_err']:+.4f}",
            ])

    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#d5e8f0")
        table[0, j].set_text_props(weight="bold")
    # Color FAIL rows
    for i, row in enumerate(rows, start=1):
        if "FAIL" in row:
            for j in range(len(headers)):
                table[i, j].set_facecolor("#fdd")

    pdf.savefig(fig)
    plt.close(fig)


def _add_theory_page(pdf):
    """Page explaining the expected results."""
    fig = plt.figure(figsize=(11, 8.5))
    text = (
        "Theoretical Background\n"
        "=" * 50 + "\n\n"
        "For rigid rotation by angle \u03b8 around the image center:\n\n"
        "  Displacement gradient:\n"
        "    du/dx = cos\u03b8 \u2212 1    du/dy = \u2212sin\u03b8\n"
        "    dv/dx = sin\u03b8       dv/dy = cos\u03b8 \u2212 1\n\n"
        "  Infinitesimal strain (symmetric part of grad u):\n"
        "    exx = eyy = cos\u03b8 \u2212 1   (NOT zero for finite \u03b8!)\n"
        "    exy = 0\n"
        "    For 2\u00b0:  exx = \u22126.1e\u22124\n"
        "    For 5\u00b0:  exx = \u22123.8e\u22123\n"
        "    For 30\u00b0: exx = \u22120.134  (13.4% apparent strain!)\n\n"
        "  Green-Lagrangian strain E = (F\u1d40F \u2212 I) / 2:\n"
        "    Since F = R (rotation), F\u1d40F = R\u1d40R = I\n"
        "    \u2192 E = 0 exactly, regardless of \u03b8\n\n"
        "  \u2192 Green-Lagrangian is the correct strain measure\n"
        "    for large-rotation rigid body motion.\n\n"
        "Accumulative vs Incremental mode:\n"
        "  Accumulative: every frame vs reference.\n"
        "    Fails when inter-frame displacement > FFT search range.\n"
        "    For rotation at edge (r=200 px):\n"
        "      5\u00b0  \u2192 ~17 px displacement  \u2192 OK (FFT=30)\n"
        "      15\u00b0 \u2192 ~52 px displacement  \u2192 FAIL\n\n"
        "  Incremental: each frame vs previous.\n"
        "    Inter-frame displacement stays small (5\u00b0/step = ~17 px).\n"
        "    Works for arbitrarily large cumulative rotation.\n"
    )
    fig.text(0.08, 0.92, text, fontsize=11, va="top", family="monospace",
             linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(all_data, report_path: Path):
    """Build full PDF report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(report_path)) as pdf:
        _add_title_page(pdf, all_data)
        _add_theory_page(pdf)

        for cd in all_data:
            _add_case_rotation_page(pdf, cd)
            _add_case_strain_page(pdf, cd)
            for mode in ["accumulative", "incremental"]:
                _add_case_field_page(pdf, cd, mode)

        _add_summary_table(pdf, all_data)

    print(f"\n[REPORT] Saved to: {report_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Rigid Body Motion: Strain & Rotation Accuracy Test")
    print("=" * 60)

    # Generate reference speckle
    print("\n[1/4] Generating reference speckle image...", flush=True)
    ref = generate_speckle(IMG_H, IMG_W, sigma=SIGMA, seed=SEED)
    mask = circular_mask(IMG_H, IMG_W, CX, CY, MASK_RADIUS)
    print(f"  Image: {IMG_H}x{IMG_W}, mask radius: {MASK_RADIUS} px")

    cases = build_cases()
    all_data = []

    # Run all cases
    for ci, case in enumerate(cases):
        print(f"\n[2/4] Case {ci + 1}/{len(cases)}: {case['label']}", flush=True)
        case_data = {"case": case}

        for mode in ["accumulative", "incremental"]:
            key = f"{case['name']}_{mode}"
            print(f"\n  Mode: {mode}", flush=True)

            result = run_pipeline(case, mode, ref, mask)
            if result is None:
                case_data[key] = None
                continue

            case_data[f"{key}_result"] = result

            # Build a base para for strain (using pipeline's actual para)
            para_base = result.dic_para

            print(f"    Analyzing {len(result.result_disp)} frames...", flush=True)
            frames_analysis = analyze_result(result, case, para_base, mask)
            case_data[key] = frames_analysis

            # Print summary for last frame
            last = frames_analysis[-1] if frames_analysis else None
            if last is not None:
                a = last["angle_deg"]
                print(f"    Last frame: angle={a:.0f}\u00b0, disp_rmse={last['disp_rmse']:.4f} px")
                for label, _, _, _ in STRAIN_COMBOS:
                    sm = last["strain"][label]
                    print(f"      {label:8s}: exx_rmse={sm['exx_rmse']:.2e}  "
                          f"rot={sm['rot_mean']:+.3f}\u00b0 (err={sm['rot_err']:+.4f}\u00b0)")

        all_data.append(case_data)

    # Generate report
    print(f"\n[3/4] Generating PDF report...", flush=True)
    report_path = REPORT_DIR / "rigid_body_strain_test.pdf"
    generate_report(all_data, report_path)

    print(f"\n[4/4] Done!")


if __name__ == "__main__":
    main()
