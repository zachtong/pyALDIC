"""Comprehensive FFT / NCC initial search benchmark.

Tests ``integer_search`` (direct) and ``integer_search_pyramid`` (pyramid)
across multiple deformation types and image sizes.

Deformation cases
-----------------
1. Zero displacement
2. Small translation  (2.5, -1.3) — sub-pixel
3. Medium translation (8.0, -5.0) — within search region
4. Large translation  (30.0, -20.0) — exceeds search region
5. Affine  — stretch + shear + rotation + translation
6. Quadratic — second-order spatially varying field

Metrics
-------
- RMSE (u, v) vs ground truth  (excluding edge margin)
- Max absolute error
- Execution time (wall clock)
- Valid node ratio

Produces a multi-page PDF: ``reports/fft_benchmark.pdf``
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
_proj = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj / "src"))
sys.path.insert(0, str(_proj / "tests"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import DICPara, GridxyROIRange
from al_dic.solver.integer_search import integer_search, integer_search_pyramid
from conftest import generate_speckle, apply_displacement_lagrangian

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINSIZE = 32
STEP = 16
SEARCH = 10
EDGE_MARGIN = 2 * STEP  # exclude edge nodes from RMSE
SEED = 42
REPORT_DIR = _proj / "reports"
REPORT_DIR.mkdir(exist_ok=True)
PDF_PATH = REPORT_DIR / "fft_benchmark.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_para(h: int, w: int) -> DICPara:
    """Build a DICPara for the given image size."""
    return dicpara_default(
        winsize=WINSIZE,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(h, w),
        gridxy_roi_range=GridxyROIRange(
            gridx=(0, w - 1), gridy=(0, h - 1),
        ),
        reference_mode="accumulative",
        show_plots=False,
    )


def _gt_on_grid(
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    u_func,
    v_func,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate ground truth displacement at grid nodes."""
    XX, YY = np.meshgrid(x0, y0)
    return u_func(XX, YY), v_func(XX, YY)


def _grid_rmse(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    gt_u: NDArray[np.float64],
    gt_v: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    img_size: tuple[int, int],
    margin: int = EDGE_MARGIN,
) -> dict:
    """Compute RMSE, max error, valid fraction on interior nodes."""
    h, w = img_size
    XX, YY = np.meshgrid(x0, y0)
    interior = (
        (XX >= margin) & (XX <= w - 1 - margin)
        & (YY >= margin) & (YY <= h - 1 - margin)
    )
    valid = interior & np.isfinite(u) & np.isfinite(v)
    n_interior = int(interior.sum())
    n_valid = int(valid.sum())

    if n_valid == 0:
        return dict(
            rmse_u=np.inf, rmse_v=np.inf,
            max_err_u=np.inf, max_err_v=np.inf,
            valid_ratio=0.0, n_valid=0, n_interior=n_interior,
        )

    eu = u[valid] - gt_u[valid]
    ev = v[valid] - gt_v[valid]
    return dict(
        rmse_u=float(np.sqrt(np.mean(eu ** 2))),
        rmse_v=float(np.sqrt(np.mean(ev ** 2))),
        max_err_u=float(np.max(np.abs(eu))),
        max_err_v=float(np.max(np.abs(ev))),
        valid_ratio=n_valid / max(1, n_interior),
        n_valid=n_valid,
        n_interior=n_interior,
    )


# ---------------------------------------------------------------------------
# Deformation definitions
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    dict(
        name="Zero displacement",
        u_func=lambda x, y: np.zeros_like(x),
        v_func=lambda x, y: np.zeros_like(x),
        desc="Identical images — baseline noise floor",
    ),
    dict(
        name="Small translation (2.5, -1.3)",
        u_func=lambda x, y: np.full_like(x, 2.5),
        v_func=lambda x, y: np.full_like(x, -1.3),
        desc="Sub-pixel rigid shift — tests sub-pixel accuracy",
    ),
    dict(
        name="Medium translation (8, -5)",
        u_func=lambda x, y: np.full_like(x, 8.0),
        v_func=lambda x, y: np.full_like(x, -5.0),
        desc="Within search region (search=10) — should work for both methods",
    ),
    dict(
        name="Large translation (30, -20)",
        u_func=lambda x, y: np.full_like(x, 30.0),
        v_func=lambda x, y: np.full_like(x, -20.0),
        desc="Exceeds search region — direct search will FAIL, pyramid required",
    ),
    dict(
        name="Affine (stretch+shear+rotation)",
        # u = 0.03x - 0.01y + 3,  v = 0.01x + 0.02y - 2
        u_func=lambda x, y: 0.03 * (x - 128) - 0.01 * (y - 128) + 3.0,
        v_func=lambda x, y: 0.01 * (x - 128) + 0.02 * (y - 128) - 2.0,
        desc="1st-order: du/dx=0.03, du/dy=-0.01, dv/dx=0.01, dv/dy=0.02",
    ),
    dict(
        name="Quadratic",
        # u = 1e-4 * x^2 + 5e-5 * x*y + 2,  v = -5e-5 * y^2 + 3e-5 * x*y + 1
        u_func=lambda x, y: 1e-4 * (x - 128) ** 2 + 5e-5 * (x - 128) * (y - 128) + 2.0,
        v_func=lambda x, y: -5e-5 * (y - 128) ** 2 + 3e-5 * (x - 128) * (y - 128) + 1.0,
        desc="2nd-order spatially varying — tests limits of NCC integer search",
    ),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_one_case(
    case: dict,
    ref_img: NDArray[np.float64],
    para: DICPara,
    n_pyr_levels: int = 3,
    refine_search: int = 3,
) -> dict:
    """Run both direct and pyramid search, return metrics."""
    h, w = ref_img.shape

    # Generate deformed image
    def_img = apply_displacement_lagrangian(
        ref_img, case["u_func"], case["v_func"],
    )

    results = {}

    # --- Direct search ---
    t0 = time.perf_counter()
    x0_d, y0_d, u_d, v_d, info_d = integer_search(ref_img, def_img, para)
    t_direct = time.perf_counter() - t0

    gt_u_d, gt_v_d = _gt_on_grid(x0_d, y0_d, case["u_func"], case["v_func"])
    m_d = _grid_rmse(u_d, v_d, gt_u_d, gt_v_d, x0_d, y0_d, (h, w))
    results["direct"] = dict(
        x0=x0_d, y0=y0_d, u=u_d, v=v_d,
        gt_u=gt_u_d, gt_v=gt_v_d,
        time_s=t_direct, **m_d,
    )

    # --- Pyramid search ---
    t0 = time.perf_counter()
    x0_p, y0_p, u_p, v_p, info_p = integer_search_pyramid(
        ref_img, def_img, para,
        n_levels=n_pyr_levels, refine_search=refine_search,
    )
    t_pyramid = time.perf_counter() - t0

    gt_u_p, gt_v_p = _gt_on_grid(x0_p, y0_p, case["u_func"], case["v_func"])
    m_p = _grid_rmse(u_p, v_p, gt_u_p, gt_v_p, x0_p, y0_p, (h, w))
    results["pyramid"] = dict(
        x0=x0_p, y0=y0_p, u=u_p, v=v_p,
        gt_u=gt_u_p, gt_v=gt_v_p,
        time_s=t_pyramid, **m_p,
    )

    return results


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------

def _add_field_page(
    pdf: PdfPages,
    case_name: str,
    method: str,
    r: dict,
):
    """One page: u/v computed, u/v error maps."""
    x0, y0 = r["x0"], r["y0"]
    u, v = r["u"], r["v"]
    gt_u, gt_v = r["gt_u"], r["gt_v"]
    err_u = u - gt_u
    err_v = v - gt_v

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{case_name} — {method}\n"
                 f"RMSE: u={r['rmse_u']:.4f} px, v={r['rmse_v']:.4f} px  |  "
                 f"MaxErr: u={r['max_err_u']:.4f}, v={r['max_err_v']:.4f}  |  "
                 f"Time: {r['time_s']*1000:.1f} ms  |  "
                 f"Valid: {r['valid_ratio']*100:.0f}%",
                 fontsize=10, y=0.98)

    extent = [x0[0], x0[-1], y0[-1], y0[0]]

    for ax, data, title, cmap in [
        (axes[0, 0], u, "u (computed)", "RdBu_r"),
        (axes[0, 1], gt_u, "u (ground truth)", "RdBu_r"),
        (axes[0, 2], err_u, "u error", "coolwarm"),
        (axes[1, 0], v, "v (computed)", "RdBu_r"),
        (axes[1, 1], gt_v, "v (ground truth)", "RdBu_r"),
        (axes[1, 2], err_v, "v error", "coolwarm"),
    ]:
        im = ax.imshow(data, extent=extent, origin="upper",
                       cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def _add_summary_table(pdf: PdfPages, all_results: list[dict]):
    """Summary table page comparing all cases and methods."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title("FFT / NCC Search Benchmark Summary", fontsize=14, pad=20)

    col_labels = [
        "Case", "Method",
        "RMSE u (px)", "RMSE v (px)",
        "MaxErr u", "MaxErr v",
        "Valid %", "Time (ms)",
    ]

    rows = []
    colors = []
    for entry in all_results:
        case_name = entry["case"]
        for method in ["direct", "pyramid"]:
            r = entry[method]
            failed = r["rmse_u"] > 5.0 or r["valid_ratio"] < 0.3
            rows.append([
                case_name if method == "direct" else "",
                method.upper(),
                f"{r['rmse_u']:.4f}" if not failed else "FAIL",
                f"{r['rmse_v']:.4f}" if not failed else "FAIL",
                f"{r['max_err_u']:.3f}" if not failed else "—",
                f"{r['max_err_v']:.3f}" if not failed else "—",
                f"{r['valid_ratio']*100:.0f}",
                f"{r['time_s']*1000:.1f}",
            ])
            if failed:
                colors.append(["#ffcccc"] * len(col_labels))
            elif method == "pyramid":
                colors.append(["#e6f2ff"] * len(col_labels))
            else:
                colors.append(["#ffffff"] * len(col_labels))

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_speed_chart(pdf: PdfPages, all_results: list[dict]):
    """Bar chart comparing direct vs pyramid speed."""
    names = [e["case"] for e in all_results]
    t_direct = [e["direct"]["time_s"] * 1000 for e in all_results]
    t_pyramid = [e["pyramid"]["time_s"] * 1000 for e in all_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, t_direct, width, label="Direct", color="#4477AA")
    ax.bar(x + width / 2, t_pyramid, width, label="Pyramid", color="#CC6677")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Execution Time: Direct vs Pyramid Search")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_accuracy_chart(pdf: PdfPages, all_results: list[dict]):
    """Bar chart comparing direct vs pyramid RMSE."""
    names = [e["case"] for e in all_results]
    rmse_d = [max(e["direct"]["rmse_u"], e["direct"]["rmse_v"])
              for e in all_results]
    rmse_p = [max(e["pyramid"]["rmse_u"], e["pyramid"]["rmse_v"])
              for e in all_results]
    # Cap for display
    cap = 10.0
    rmse_d_c = [min(r, cap) for r in rmse_d]
    rmse_p_c = [min(r, cap) for r in rmse_p]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    bars_d = ax.bar(x - width / 2, rmse_d_c, width, label="Direct",
                    color="#4477AA")
    bars_p = ax.bar(x + width / 2, rmse_p_c, width, label="Pyramid",
                    color="#CC6677")

    # Mark capped bars
    for i, (rd, rp) in enumerate(zip(rmse_d, rmse_p)):
        if rd >= cap:
            bars_d[i].set_hatch("//")
            ax.text(i - width / 2, cap + 0.2, "FAIL", ha="center",
                    fontsize=7, color="red")
        if rp >= cap:
            bars_p[i].set_hatch("//")
            ax.text(i + width / 2, cap + 0.2, "FAIL", ha="center",
                    fontsize=7, color="red")

    ax.set_ylabel("max(RMSE_u, RMSE_v) [px]")
    ax.set_title("Accuracy: Direct vs Pyramid Search (capped at 10 px)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_size_scaling_page(pdf: PdfPages, scaling_results: list[dict]):
    """Speed scaling across image sizes."""
    sizes = [r["size"] for r in scaling_results]
    t_d = [r["direct_ms"] for r in scaling_results]
    t_p = [r["pyramid_ms"] for r in scaling_results]
    n_nodes = [r["n_nodes"] for r in scaling_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(sizes, t_d, "o-", label="Direct", color="#4477AA", linewidth=2)
    ax1.plot(sizes, t_p, "s-", label="Pyramid", color="#CC6677", linewidth=2)
    ax1.set_xlabel("Image size (pixels)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Search Time vs Image Size (translation=8px)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Throughput: nodes / second
    tp_d = [n / (t / 1000) if t > 0 else 0 for n, t in zip(n_nodes, t_d)]
    tp_p = [n / (t / 1000) if t > 0 else 0 for n, t in zip(n_nodes, t_p)]
    ax2.plot(sizes, [t / 1000 for t in tp_d], "o-", label="Direct",
             color="#4477AA", linewidth=2)
    ax2.plot(sizes, [t / 1000 for t in tp_p], "s-", label="Pyramid",
             color="#CC6677", linewidth=2)
    ax2.set_xlabel("Image size (pixels)")
    ax2.set_ylabel("Throughput (k nodes/s)")
    ax2.set_title("Search Throughput vs Image Size")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FFT / NCC Search Benchmark")
    print("=" * 60)

    # Generate reference image (512x512 for accuracy tests)
    IMG_H, IMG_W = 512, 512
    ref_img = generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=SEED)
    para = _make_para(IMG_H, IMG_W)

    # Warm-up (first call might be slower due to module imports, JIT etc.)
    _ = integer_search(ref_img, ref_img, para)

    # ---- Run all deformation cases ----
    all_results: list[dict] = []

    for case in CASES:
        print(f"\n--- {case['name']} ---")
        print(f"    {case['desc']}")

        r = run_one_case(case, ref_img, para)
        entry = dict(case=case["name"], desc=case["desc"],
                     direct=r["direct"], pyramid=r["pyramid"])
        all_results.append(entry)

        for method in ["direct", "pyramid"]:
            m = r[method]
            status = "OK" if m["rmse_u"] < 5.0 and m["valid_ratio"] > 0.3 else "FAIL"
            print(f"  [{method.upper():7s}] RMSE u={m['rmse_u']:.4f} v={m['rmse_v']:.4f}  "
                  f"MaxErr u={m['max_err_u']:.3f} v={m['max_err_v']:.3f}  "
                  f"Valid={m['valid_ratio']*100:.0f}%  "
                  f"Time={m['time_s']*1000:.1f}ms  {status}")

    # ---- Speed scaling across image sizes ----
    print("\n--- Speed scaling ---")
    scaling_results = []
    for sz in [128, 256, 512, 1024]:
        ref_sz = generate_speckle(sz, sz, sigma=3.0, seed=SEED)
        para_sz = _make_para(sz, sz)

        # Medium translation (within search region)
        u_func = lambda x, y: np.full_like(x, 8.0)
        v_func = lambda x, y: np.full_like(x, -5.0)
        def_sz = apply_displacement_lagrangian(ref_sz, u_func, v_func)

        # Warm up
        _ = integer_search(ref_sz, def_sz, para_sz)

        # Time direct
        n_runs = max(1, 5 if sz <= 512 else 2)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            x0_d, y0_d, _, _, _ = integer_search(ref_sz, def_sz, para_sz)
        t_d = (time.perf_counter() - t0) / n_runs * 1000

        # Time pyramid
        t0 = time.perf_counter()
        for _ in range(n_runs):
            x0_p, _, _, _, _ = integer_search_pyramid(ref_sz, def_sz, para_sz)
        t_p = (time.perf_counter() - t0) / n_runs * 1000

        n_nodes = len(x0_d) * len(y0_d)
        scaling_results.append(dict(
            size=sz, direct_ms=t_d, pyramid_ms=t_p, n_nodes=n_nodes,
        ))
        print(f"  {sz}x{sz}: direct={t_d:.1f}ms  pyramid={t_p:.1f}ms  "
              f"nodes={n_nodes}")

    # ---- Generate PDF ----
    print(f"\nGenerating PDF report: {PDF_PATH}")
    with PdfPages(str(PDF_PATH)) as pdf:
        # Page 1: Summary table
        _add_summary_table(pdf, all_results)

        # Page 2: Accuracy bar chart
        _add_accuracy_chart(pdf, all_results)

        # Page 3: Speed bar chart
        _add_speed_chart(pdf, all_results)

        # Page 4: Speed scaling
        _add_size_scaling_page(pdf, scaling_results)

        # Per-case field maps
        for entry in all_results:
            for method in ["direct", "pyramid"]:
                _add_field_page(pdf, entry["case"], method, entry[method])

    print(f"\nReport saved to: {PDF_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
