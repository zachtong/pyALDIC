"""Comprehensive synthetic validation: affine + quadratic deformation fields.

Generates multi-frame 1024x1024 synthetic tests with known ground truth,
runs the full AL-DIC pipeline, and produces:
  - Per-section timing breakdown
  - ADMM convergence history
  - Per-frame displacement accuracy (RMSE, max error)
  - Full-field result plots + error maps

Usage:
    python scripts/validate_synthetic.py
"""

import sys
import time
import warnings
import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter, map_coordinates
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange, PipelineResult
from staq_dic.core.pipeline import run_aldic

# Enable ADMM convergence logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("staq_dic.core.pipeline")
logger.setLevel(logging.INFO)

OUT_DIR = Path("outputs/validate_synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Image generation
# ============================================================

def generate_speckle(h, w, sigma=3.0, seed=42):
    """Synthetic speckle pattern, normalized to [20, 235]."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement(ref, u_field, v_field):
    """Inverse warp: warped(y, x) = ref(y - v, x - u). Order=5 to avoid
    double-interpolation artifact with IC-GN's order=3."""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="nearest", cval=0.0).reshape(h, w)


# ============================================================
# Deformation field definitions (pixel-domain)
# ============================================================

def make_fields(h, w):
    """Return dict of (label, u_field, v_field, desc) for each test case.
    All fields are (H, W) float64 in pixel units."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    # Normalized coordinates [-1, 1]
    xn = (xx - cx) / cx
    yn = (yy - cy) / cy

    cases = {}

    # Case 1: Pure translation (0.8 px in x, 0.3 px in y)
    cases["translation"] = {
        "u": np.full((h, w), 0.8),
        "v": np.full((h, w), 0.3),
        "desc": "Translation: u=0.8px, v=0.3px",
    }

    # Case 2: Affine stretch + shear
    # u = 0.02*x + 0.005*y + 0.5
    # v = 0.005*x + 0.01*y + 0.2
    cases["affine"] = {
        "u": 0.02 * (xx - cx) + 0.005 * (yy - cy) + 0.5,
        "v": 0.005 * (xx - cx) + 0.01 * (yy - cy) + 0.2,
        "desc": "Affine: 2% stretch_x, 1% stretch_y, 0.5% shear, translation",
    }

    # Case 3: Larger affine with rotation component
    # u = 0.03*x - 0.015*y + 1.0
    # v = 0.015*x + 0.02*y - 0.5
    cases["affine_large"] = {
        "u": 0.03 * (xx - cx) - 0.015 * (yy - cy) + 1.0,
        "v": 0.015 * (xx - cx) + 0.02 * (yy - cy) - 0.5,
        "desc": "Large affine: 3% stretch_x + 1.5% rotation + translation",
    }

    # Case 4: Quadratic (parabolic bending)
    # u = 0.5 + 2e-6 * (x-cx)^2
    # v = 0.2 + 3e-6 * (x-cx) * (y-cy)
    cases["quadratic"] = {
        "u": 0.5 + 2e-6 * (xx - cx) ** 2,
        "v": 0.2 + 3e-6 * (xx - cx) * (yy - cy),
        "desc": "Quadratic: parabolic u + bilinear v",
    }

    # Case 5: Complex quadratic (barrel distortion-like)
    # u = 1e-6 * ((x-cx)^2 - (y-cy)^2) + 0.3
    # v = 2e-6 * (x-cx)*(y-cy) + 0.1
    cases["quadratic_complex"] = {
        "u": 1e-6 * ((xx - cx) ** 2 - (yy - cy) ** 2) + 0.3,
        "v": 2e-6 * (xx - cx) * (yy - cy) + 0.1,
        "desc": "Complex quadratic: barrel distortion + translation",
    }

    return cases


# ============================================================
# Ground truth extraction at mesh nodes
# ============================================================

def extract_gt_at_nodes(coords, u_field, v_field):
    """Interpolate pixel-domain displacement fields to mesh node locations.
    coords: (n_nodes, 2) with columns [x, y].
    Returns (u_gt, v_gt) each (n_nodes,)."""
    # coords[:, 0] = x, coords[:, 1] = y
    # fields are indexed as [row=y, col=x]
    from scipy.interpolate import RectBivariateSpline
    h, w = u_field.shape
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    spl_u = RectBivariateSpline(ys, xs, u_field, kx=3, ky=3)
    spl_v = RectBivariateSpline(ys, xs, v_field, kx=3, ky=3)
    u_gt = spl_u.ev(coords[:, 1], coords[:, 0])
    v_gt = spl_v.ev(coords[:, 1], coords[:, 0])
    return u_gt, v_gt


# ============================================================
# Plotting helpers
# ============================================================

def plot_field(ax, coords, values, title, cmap="RdBu_r", symmetric=False):
    """Scatter plot of a nodal field on mesh coordinates."""
    x, y = coords[:, 0], coords[:, 1]
    vmin, vmax = np.nanpercentile(values, [2, 98])
    if symmetric:
        vlim = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    else:
        norm = None
        if vmin == vmax:
            vmin -= 0.01
            vmax += 0.01
    sc = ax.scatter(x, y, c=values, s=1, cmap=cmap, norm=norm,
                    vmin=None if symmetric else vmin,
                    vmax=None if symmetric else vmax,
                    rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, shrink=0.7)


def generate_frame_plots(coords, u_py, v_py, u_gt, v_gt, case_name, frame_label):
    """Generate displacement + error maps for one frame."""
    u_err = u_py - u_gt
    v_err = v_py - v_gt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{case_name} — {frame_label}", fontsize=12)

    plot_field(axes[0, 0], coords, u_gt, "u ground truth (px)")
    plot_field(axes[0, 1], coords, u_py, "u computed (px)")
    plot_field(axes[0, 2], coords, u_err, "u error (px)", symmetric=True)

    plot_field(axes[1, 0], coords, v_gt, "v ground truth (px)")
    plot_field(axes[1, 1], coords, v_py, "v computed (px)")
    plot_field(axes[1, 2], coords, v_err, "v error (px)", symmetric=True)

    plt.tight_layout()
    fname = OUT_DIR / f"{case_name}_{frame_label}_disp.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def generate_strain_plots(coords, strain_result, case_name, frame_label):
    """Generate strain field plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{case_name} — {frame_label} — Strain", fontsize=12)

    plot_field(axes[0], coords, strain_result.strain_exx, "exx")
    plot_field(axes[1], coords, strain_result.strain_exy, "exy")
    plot_field(axes[2], coords, strain_result.strain_eyy, "eyy")

    plt.tight_layout()
    fname = OUT_DIR / f"{case_name}_{frame_label}_strain.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ============================================================
# Main validation
# ============================================================

def run_validation():
    H, W = 1024, 1024
    step = 4
    ws = 16

    print("=" * 75)
    print(f"STAQ-DIC Synthetic Validation: {H}x{W}, step={step}, ws={ws}")
    print("=" * 75)

    # Generate reference speckle
    print("\n[Setup] Generating speckle pattern...")
    ref = generate_speckle(H, W, sigma=3.0, seed=42)

    # Define deformation cases
    cases = make_fields(H, W)

    # Create images: ref + 5 deformed frames
    print("[Setup] Warping deformed images...")
    images = [ref]
    case_names = []
    gt_fields = []
    for name, c in cases.items():
        img = apply_displacement(ref, c["u"], c["v"])
        images.append(img)
        case_names.append(name)
        gt_fields.append((c["u"], c["v"]))
        print(f"  Frame {len(images)}: {c['desc']}")
        u_range = c["u"].max() - c["u"].min()
        v_range = c["v"].max() - c["v"].min()
        print(f"    u range: [{c['u'].min():.3f}, {c['u'].max():.3f}] ({u_range:.3f}px span)")
        print(f"    v range: [{c['v'].min():.3f}, {c['v'].max():.3f}] ({v_range:.3f}px span)")

    n_frames = len(images)
    masks = [np.ones((H, W), dtype=np.float64)] * n_frames

    # DIC parameters
    roi = GridxyROIRange(gridx=(ws, W - ws - 1), gridy=(ws, H - ws - 1))
    para = dicpara_default(
        winsize=ws, winstepsize=step, winsize_min=step,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
        reference_mode="accumulative",
        show_plots=False,
    )

    # Progress callback for timing
    section_times = {}
    last_time = [time.perf_counter()]
    last_msg = ["start"]

    def progress_fn(frac, msg):
        now = time.perf_counter()
        elapsed = now - last_time[0]
        if elapsed > 0.01:
            section_times[last_msg[0]] = elapsed
        last_time[0] = now
        last_msg[0] = msg
        print(f"  [{frac*100:5.1f}%] {msg}")

    # Run pipeline
    print(f"\n{'='*75}")
    print(f"Running pipeline ({n_frames} images, {n_frames-1} deformed frames)...")
    print(f"{'='*75}")

    t_total_start = time.perf_counter()
    result = run_aldic(
        para, images, masks,
        progress_fn=progress_fn,
        compute_strain=True,
    )
    t_total = time.perf_counter() - t_total_start
    # Capture last section
    section_times[last_msg[0]] = time.perf_counter() - last_time[0]

    print(f"\n  TOTAL pipeline time: {t_total:.3f}s")

    # Extract mesh info
    coords = result.dic_mesh.coordinates_fem
    n_nodes = coords.shape[0]
    n_ele = result.dic_mesh.elements_fem.shape[0]
    print(f"  Mesh: {n_nodes} nodes, {n_ele} elements")

    # ============================================================
    # Section timing summary
    # ============================================================
    print(f"\n{'='*75}")
    print("SECTION TIMING")
    print(f"{'='*75}")
    for msg, t in section_times.items():
        if t > 0.01:
            print(f"  {msg:<55s} {t:7.3f}s")
    print(f"  {'TOTAL':<55s} {t_total:7.3f}s")

    # ============================================================
    # Per-frame accuracy analysis
    # ============================================================
    print(f"\n{'='*75}")
    print("PER-FRAME ACCURACY")
    print(f"{'='*75}")

    header = f"{'Frame':<25s} {'u_RMSE':>8s} {'v_RMSE':>8s} {'u_max':>8s} {'v_max':>8s} {'disp_RMSE':>10s}"
    print(header)
    print("-" * 75)

    all_metrics = []

    for i in range(n_frames - 1):
        name = case_names[i]
        u_field_gt, v_field_gt = gt_fields[i]

        # Python results (accumulative displacement)
        U = result.result_disp[i].U_accum
        if U is None:
            U = result.result_disp[i].U
        u_py = U[0::2]
        v_py = U[1::2]

        # Ground truth at mesh nodes
        u_gt, v_gt = extract_gt_at_nodes(coords, u_field_gt, v_field_gt)

        # Metrics
        u_err = u_py - u_gt
        v_err = v_py - v_gt
        u_rmse = np.sqrt(np.nanmean(u_err ** 2))
        v_rmse = np.sqrt(np.nanmean(v_err ** 2))
        u_maxe = np.nanmax(np.abs(u_err))
        v_maxe = np.nanmax(np.abs(v_err))
        disp_rmse = np.sqrt(np.nanmean(u_err ** 2 + v_err ** 2))

        metrics = {
            "name": name, "u_rmse": u_rmse, "v_rmse": v_rmse,
            "u_max": u_maxe, "v_max": v_maxe, "disp_rmse": disp_rmse,
        }
        all_metrics.append(metrics)

        print(f"  {name:<23s} {u_rmse:8.4f} {v_rmse:8.4f} {u_maxe:8.4f} {v_maxe:8.4f} {disp_rmse:10.4f}")

        # Generate plots
        fname_disp = generate_frame_plots(
            coords, u_py, v_py, u_gt, v_gt, name, f"frame{i+2}",
        )

        if result.result_strain and result.result_strain[i] is not None:
            fname_strain = generate_strain_plots(
                coords, result.result_strain[i], name, f"frame{i+2}",
            )

    print("-" * 75)

    # ============================================================
    # Strain accuracy (for cases with known analytical strain)
    # ============================================================
    print(f"\n{'='*75}")
    print("STRAIN ACCURACY (analytical ground truth)")
    print(f"{'='*75}")

    # Affine cases have constant strain
    strain_gt = {
        "translation": {"exx": 0.0, "eyy": 0.0, "exy": 0.0},
        "affine": {"exx": 0.02, "eyy": 0.01, "exy": 0.5 * (0.005 + 0.005)},
        "affine_large": {"exx": 0.03, "eyy": 0.02, "exy": 0.5 * (-0.015 + 0.015)},
    }

    header2 = f"{'Frame':<25s} {'exx_RMSE':>10s} {'exy_RMSE':>10s} {'eyy_RMSE':>10s}"
    print(header2)
    print("-" * 60)

    for i, name in enumerate(case_names):
        if name not in strain_gt:
            continue
        gt = strain_gt[name]
        sr = result.result_strain[i]
        if sr is None:
            continue

        # For small strain, dudx ≈ exx, etc.
        exx_err = sr.dudx - gt["exx"]
        eyy_err = sr.dvdy - gt["eyy"]
        exy_err = 0.5 * (sr.dudy + sr.dvdx) - gt["exy"]

        # Trim boundary nodes (strain less accurate at edges)
        margin = ws
        inner = (
            (coords[:, 0] > margin) & (coords[:, 0] < W - margin) &
            (coords[:, 1] > margin) & (coords[:, 1] < H - margin)
        )

        exx_rmse = np.sqrt(np.nanmean(exx_err[inner] ** 2))
        exy_rmse = np.sqrt(np.nanmean(exy_err[inner] ** 2))
        eyy_rmse = np.sqrt(np.nanmean(eyy_err[inner] ** 2))

        print(f"  {name:<23s} {exx_rmse:10.6f} {exy_rmse:10.6f} {eyy_rmse:10.6f}")

    print("-" * 60)

    # ============================================================
    # PASS / FAIL summary
    # ============================================================
    print(f"\n{'='*75}")
    print("VALIDATION SUMMARY")
    print(f"{'='*75}")

    disp_tol = 0.5  # px
    all_pass = True
    for m in all_metrics:
        passed = m["disp_rmse"] < disp_tol
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {m['name']:<23s} disp_RMSE={m['disp_rmse']:.4f}px  [{status}] (tol={disp_tol}px)")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  Output dir: {OUT_DIR.resolve()}")
    print(f"  Total time: {t_total:.3f}s")

    return all_pass


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
