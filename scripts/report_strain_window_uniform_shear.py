#!/usr/bin/env python
"""Visual report: StrainController on uniform pure shear.

Drives :class:`StrainController` over a synthetic ``PipelineResult`` whose
displacement field is exactly ``u = shear * y, v = 0`` so that the
ground-truth strain is analytically known:

    epsilon_xx = 0
    epsilon_yy = 0
    epsilon_xy = shear / 2

The script renders a multi-page PDF that lets the user visually verify the
strain post-processing window's accuracy, method comparison (plane-fit vs
FEM), strain-type comparison (infinitesimal vs Green-Lagrangian), and
residual distributions.

Output: reports/strain_window_uniform_shear.pdf
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from al_dic.gui.app_state import AppState  # noqa: E402
from al_dic.gui.controllers.strain_controller import StrainController  # noqa: E402

from test_gui._helpers import make_synthetic_pipeline_result  # noqa: E402

# ---- Synthetic configuration -------------------------------------------------

SHEAR: float = 0.01
N_FRAMES: int = 4  # frame 0 reference + 3 deformed frames
IMG_SHAPE: tuple[int, int] = (256, 256)
STEP: int = 16
PLANE_FIT_RAD: float = 20.0
SMOOTHNESS: float = 1e-5
PREVIEW_FRAME: int = 2  # 0-indexed within result_disp -> cumulative shear 3*0.01

# ---- Helpers -----------------------------------------------------------------


def _build_state() -> tuple[AppState, np.ndarray]:
    """Construct a fresh AppState pre-populated with the synthetic result."""
    AppState._instance = None
    state = AppState()
    result, mask = make_synthetic_pipeline_result(
        n_frames=N_FRAMES, shear=SHEAR, img_shape=IMG_SHAPE, step=STEP,
    )
    state.results = result
    state.per_frame_rois[0] = mask.astype(bool)
    return state, mask


def _run_strain(
    state: AppState, method: int, strain_type: int,
) -> list:
    """Recompute strain in-place with a given method/type and return list."""
    ctrl = StrainController(state)
    override = {
        "method_to_compute_strain": method,
        "strain_plane_fit_rad": PLANE_FIT_RAD,
        "strain_smoothness": SMOOTHNESS,
        "strain_type": strain_type,
    }
    return ctrl.compute_all_frames(override)


def _ground_truth(frame_idx: int, n_nodes: int) -> dict[str, np.ndarray]:
    """Analytic ground-truth strain for cumulative pure shear at ``frame_idx``."""
    cumulative = (frame_idx + 1) * SHEAR
    exy = cumulative / 2.0
    return {
        "strain_exx": np.zeros(n_nodes),
        "strain_eyy": np.zeros(n_nodes),
        "strain_exy": np.full(n_nodes, exy),
        "strain_principal_max": np.full(n_nodes, exy),
        "strain_principal_min": np.full(n_nodes, -exy),
        "strain_maxshear": np.full(n_nodes, exy),
    }


def _scatter_field(
    ax, coords: np.ndarray, values: np.ndarray, title: str,
    vmin: float | None = None, vmax: float | None = None,
    cmap: str = "RdBu_r",
) -> None:
    """Render a per-node strain field as a scatter coloured by value."""
    valid = ~np.isnan(values)
    sc = ax.scatter(
        coords[valid, 0],
        coords[valid, 1],
        c=values[valid],
        s=18,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, IMG_SHAPE[1])
    ax.set_ylim(IMG_SHAPE[0], 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Root mean square error ignoring NaNs in either array."""
    diff = predicted - truth
    valid = ~np.isnan(diff)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(diff[valid] ** 2)))


# ---- Report generation -------------------------------------------------------


def generate_report(out_dir: Path | None = None) -> Path:
    """Build the PDF and return its absolute path."""
    if out_dir is None:
        out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "strain_window_uniform_shear.pdf"

    state, mask = _build_state()
    coords = state.results.result_fe_mesh_each_frame[0].coordinates_fem
    n_nodes = coords.shape[0]

    # Run all four (method, type) combinations once.
    runs: dict[tuple[int, int], list] = {
        (2, 0): _run_strain(state, method=2, strain_type=0),
        (3, 0): _run_strain(state, method=3, strain_type=0),
        (2, 2): _run_strain(state, method=2, strain_type=2),
        (3, 2): _run_strain(state, method=3, strain_type=2),
    }

    gt = _ground_truth(PREVIEW_FRAME, n_nodes)
    gt_exy = gt["strain_exy"][0]
    diverging_lim = max(0.005, abs(gt_exy) * 1.5)

    with PdfPages(str(pdf_path)) as pdf:
        # ── Page 1: Title + parameter table ──────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle(
            "Strain Post-Processing Window — Uniform Pure Shear Validation",
            fontsize=15, y=0.96,
        )
        ax.axis("off")

        intro = (
            "Synthetic ground truth: u = shear * y, v = 0\n"
            "Analytic strain: epsilon_xx = 0, epsilon_yy = 0, "
            "epsilon_xy = shear / 2\n\n"
            "This report drives StrainController.compute_all_frames over a\n"
            "PipelineResult whose result_disp encodes the cumulative pure-shear\n"
            "field exactly, then renders the resulting strain fields with\n"
            "method 2 (plane fitting) and method 3 (FEM nodal), under both\n"
            "infinitesimal (type 0) and Green-Lagrangian (type 2) measures."
        )
        ax.text(
            0.05, 0.85, intro, fontsize=11, verticalalignment="top",
            family="monospace",
        )

        param_rows = [
            ["Image size", f"{IMG_SHAPE[0]} x {IMG_SHAPE[1]} px"],
            ["Mesh step", f"{STEP} px"],
            ["Frames", f"{N_FRAMES} ({N_FRAMES - 1} deformed)"],
            ["Per-frame shear", f"{SHEAR}"],
            [
                "Cumulative shear (preview frame)",
                f"{(PREVIEW_FRAME + 1) * SHEAR}",
            ],
            ["Plane-fit radius", f"{PLANE_FIT_RAD} px"],
            ["Strain smoothness", f"{SMOOTHNESS}"],
            ["GT epsilon_xy (preview frame)", f"{gt_exy}"],
            ["Methods exercised", "2 (plane fit), 3 (FEM nodal)"],
            ["Strain types exercised", "0 (infinitesimal), 2 (Green-Lagrangian)"],
        ]
        table = ax.table(
            cellText=param_rows,
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            colWidths=[0.45, 0.45],
            loc="lower center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)
        for j in range(2):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(len(param_rows)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            for j in range(2):
                table[i + 1, j].set_facecolor(color)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Ground-truth field + ROI mask ────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(
            f"Ground Truth — Cumulative shear at frame {PREVIEW_FRAME + 1}",
            fontsize=13, y=0.97,
        )

        axes[0].imshow(mask, cmap="gray", origin="upper")
        axes[0].set_title("ROI mask (full image)", fontsize=10)
        axes[0].tick_params(labelsize=7)

        _scatter_field(
            axes[1], coords, gt["strain_exy"],
            title=f"Analytic epsilon_xy = {gt_exy:.4f}",
            vmin=-diverging_lim, vmax=diverging_lim,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Method 2 (plane fitting) ─────────────────────────
        sr_m2 = runs[(2, 0)][PREVIEW_FRAME]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(
            "Method 2 — Plane Fitting (infinitesimal strain)",
            fontsize=13, y=0.98,
        )
        _scatter_field(
            axes[0, 0], coords, sr_m2.strain_exx,
            "epsilon_xx (computed)", -diverging_lim, diverging_lim,
        )
        _scatter_field(
            axes[0, 1], coords, sr_m2.strain_eyy,
            "epsilon_yy (computed)", -diverging_lim, diverging_lim,
        )
        _scatter_field(
            axes[1, 0], coords, sr_m2.strain_exy,
            "epsilon_xy (computed)", -diverging_lim, diverging_lim,
        )
        err_m2 = sr_m2.strain_exy - gt["strain_exy"]
        err_lim = max(1e-4, np.nanmax(np.abs(err_m2)) * 1.1)
        _scatter_field(
            axes[1, 1], coords, err_m2,
            "epsilon_xy error vs GT", -err_lim, err_lim, cmap="seismic",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Method 3 (FEM nodal) ─────────────────────────────
        sr_m3 = runs[(3, 0)][PREVIEW_FRAME]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(
            "Method 3 — FEM Nodal (infinitesimal strain)",
            fontsize=13, y=0.98,
        )
        _scatter_field(
            axes[0, 0], coords, sr_m3.strain_exx,
            "epsilon_xx (computed)", -diverging_lim, diverging_lim,
        )
        _scatter_field(
            axes[0, 1], coords, sr_m3.strain_eyy,
            "epsilon_yy (computed)", -diverging_lim, diverging_lim,
        )
        _scatter_field(
            axes[1, 0], coords, sr_m3.strain_exy,
            "epsilon_xy (computed)", -diverging_lim, diverging_lim,
        )
        err_m3 = sr_m3.strain_exy - gt["strain_exy"]
        err_lim = max(1e-4, np.nanmax(np.abs(err_m3)) * 1.1)
        _scatter_field(
            axes[1, 1], coords, err_m3,
            "epsilon_xy error vs GT", -err_lim, err_lim, cmap="seismic",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 5: Principal + von Mises (method 2) ─────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle(
            "Method 2 — Principal & Maximum-Shear strains "
            f"(frame {PREVIEW_FRAME + 1})",
            fontsize=13, y=0.98,
        )
        principal_lim = diverging_lim
        _scatter_field(
            axes[0], coords, sr_m2.strain_principal_max,
            f"principal_max  (GT={gt_exy:.4f})",
            -principal_lim, principal_lim,
        )
        _scatter_field(
            axes[1], coords, sr_m2.strain_principal_min,
            f"principal_min  (GT={-gt_exy:.4f})",
            -principal_lim, principal_lim,
        )
        _scatter_field(
            axes[2], coords, sr_m2.strain_maxshear,
            f"max_shear  (GT={gt_exy:.4f})",
            -principal_lim, principal_lim,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 6: Residual histograms (method 2 vs method 3) ───────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle(
            "Residual distributions (computed - ground truth)",
            fontsize=13, y=0.97,
        )
        for idx, field in enumerate(("strain_exx", "strain_eyy", "strain_exy")):
            ax = axes[idx]
            r2 = (
                getattr(runs[(2, 0)][PREVIEW_FRAME], field) - gt[field]
            )
            r3 = (
                getattr(runs[(3, 0)][PREVIEW_FRAME], field) - gt[field]
            )
            r2 = r2[~np.isnan(r2)]
            r3 = r3[~np.isnan(r3)]
            bins = 40
            ax.hist(
                r2, bins=bins, alpha=0.55, label="method 2",
                color="#4472C4", edgecolor="black",
            )
            ax.hist(
                r3, bins=bins, alpha=0.55, label="method 3",
                color="#ED7D31", edgecolor="black",
            )
            ax.set_title(field, fontsize=10)
            ax.set_xlabel("residual", fontsize=9)
            ax.set_ylabel("count", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8)
            ax.axvline(0.0, color="k", linewidth=0.7, linestyle="--")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 7: RMSE summary table ───────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle(
            f"RMSE Summary — preview frame {PREVIEW_FRAME + 1} "
            f"(cumulative shear = {(PREVIEW_FRAME + 1) * SHEAR})",
            fontsize=13, y=0.95,
        )
        ax.axis("off")

        rmse_rows = []
        for (method, stype), strain_list in runs.items():
            sr = strain_list[PREVIEW_FRAME]
            row = [
                f"method {method} / type {stype}",
                f"{_rmse(sr.strain_exx, gt['strain_exx']):.2e}",
                f"{_rmse(sr.strain_eyy, gt['strain_eyy']):.2e}",
                f"{_rmse(sr.strain_exy, gt['strain_exy']):.2e}",
                f"{_rmse(sr.strain_principal_max, gt['strain_principal_max']):.2e}",
                f"{_rmse(sr.strain_principal_min, gt['strain_principal_min']):.2e}",
                f"{_rmse(sr.strain_maxshear, gt['strain_maxshear']):.2e}",
            ]
            rmse_rows.append(row)

        col_labels = [
            "Configuration",
            "exx",
            "eyy",
            "exy",
            "p_max",
            "p_min",
            "max_shear",
        ]
        table = ax.table(
            cellText=rmse_rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.7)
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(len(rmse_rows)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

        ax.text(
            0.5, 0.08,
            "Lower is better. Type 0 (infinitesimal) and type 2 "
            "(Green-Lagrangian) coincide for pure shear because the\n"
            "non-linear correction terms (du/dx * dv/dx, du/dy * dv/dy) "
            "vanish identically.",
            ha="center", fontsize=9, style="italic", transform=ax.transAxes,
        )

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    return pdf_path


def main() -> None:
    pdf_path = generate_report()
    print(f"Report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
