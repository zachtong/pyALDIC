"""Benchmark pipeline scaling across different mesh sizes.

Tests multiple (image_size, step) combinations to profile how each
pipeline component scales with node count.
"""

import sys
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange
from al_dic.io.image_ops import compute_image_gradient, normalize_images
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.solver.local_icgn import local_icgn
from al_dic.solver.subpb1_solver import precompute_subpb1, subpb1_solver
from al_dic.solver.subpb2_solver import precompute_subpb2, subpb2_solver
from al_dic.strain.nodal_strain_fem import global_nodal_strain_fem
from dataclasses import replace


def make_speckle(h, w, seed=42):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((h, w)), sigma=2.0)


# ============================================================
# Configurations: (image_size, step) -> different node counts
# ============================================================
configs = [
    # (H, W, step, winsize, label)
    (256,  256,  4, 16, "256x256  step=4"),
    (512,  512,  4, 16, "512x512  step=4"),
    (512,  512,  2, 16, "512x512  step=2"),
    (1024, 1024, 4, 16, "1024x1024 step=4"),
    (1024, 1024, 2, 16, "1024x1024 step=2"),
]

# Numba warmup
print("Warming up Numba JIT...")
h0, w0 = 128, 128
f0 = make_speckle(h0, w0, seed=99)
g0 = np.roll(f0, 1, axis=1)
roi0 = GridxyROIRange(gridx=(16, w0 - 17), gridy=(16, h0 - 17))
p0 = dicpara_default(
    winsize=16, winstepsize=4, winsize_min=4,
    gridxy_roi_range=roi0, img_size=(h0, w0),
    tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
)
imgs0, cr0 = normalize_images([f0, g0], roi0)
p0 = replace(p0, gridxy_roi_range=cr0)
Df0 = compute_image_gradient(imgs0[0], np.ones((h0, w0)))
x00 = np.arange(roi0.gridx[0], roi0.gridx[1] + 1, 4, dtype=np.float64)
y00 = np.arange(roi0.gridy[0], roi0.gridy[1] + 1, 4, dtype=np.float64)
m0 = mesh_setup(x00, y00, p0)
U00 = np.zeros(2 * m0.coordinates_fem.shape[0])
U00[0::2] = 1.0
_ = local_icgn(U00, m0.coordinates_fem, Df0, imgs0[0], imgs0[1], p0, p0.tol)
print("  Warmup done.\n")

# ============================================================
# Run benchmarks
# ============================================================
results = []

for H, W, step, ws, label in configs:
    print("=" * 75)
    print(f"  {label}  ({H}x{W}, step={step}, ws={ws})")
    print("=" * 75)

    f_img = make_speckle(H, W, seed=42)
    g_img = np.roll(f_img, 1, axis=1)

    roi = GridxyROIRange(gridx=(ws, W - ws - 1), gridy=(ws, H - ws - 1))
    para = dicpara_default(
        winsize=ws, winstepsize=step, winsize_min=step,
        gridxy_roi_range=roi, img_size=(H, W),
        tol=1e-3, icgn_max_iter=50, mu=1e-3, alpha=0.0,
    )
    imgs, clamped = normalize_images([f_img, g_img], roi)
    para = replace(para, gridxy_roi_range=clamped)
    f_mask = np.ones((H, W))
    Df = compute_image_gradient(imgs[0], f_mask)

    x0 = np.arange(roi.gridx[0], roi.gridx[1] + 1, step, dtype=np.float64)
    y0 = np.arange(roi.gridy[0], roi.gridy[1] + 1, step, dtype=np.float64)
    mesh = mesh_setup(x0, y0, para)
    coords = mesh.coordinates_fem
    n_nodes = coords.shape[0]
    n_ele = mesh.elements_fem.shape[0]
    print(f"  Nodes: {n_nodes:,}   Elements: {n_ele:,}")

    beta = 1e-3 * step ** 2 * para.mu
    U0 = np.zeros(2 * n_nodes)
    U0[0::2] = 1.0

    timings = {}

    # S4: Local IC-GN
    t0 = time.perf_counter()
    U1, F1, _, _, _, _ = local_icgn(U0, coords, Df, imgs[0], imgs[1], para, para.tol)
    timings["local_icgn"] = time.perf_counter() - t0

    # Precompute subpb1
    t0 = time.perf_counter()
    cache_sp1 = precompute_subpb1(coords, Df, imgs[0], para)
    timings["pre_subpb1"] = time.perf_counter() - t0

    # Precompute subpb2
    t0 = time.perf_counter()
    cache_sp2 = precompute_subpb2(mesh, para.gauss_pt_order, beta, para.mu, 0.0)
    timings["pre_subpb2"] = time.perf_counter() - t0

    # S5: first subpb2 + strain
    gd = np.zeros(4 * n_nodes)
    dd = np.zeros(2 * n_nodes)
    t0 = time.perf_counter()
    U2 = subpb2_solver(mesh, para.gauss_pt_order, beta, para.mu,
                        U1, F1, gd, dd, 0.0, step, precomputed=cache_sp2)
    timings["subpb2_solve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    F2 = global_nodal_strain_fem(mesh, para, U2)
    timings["fem_strain"] = time.perf_counter() - t0

    # ADMM x3
    gd = F2 - F1
    dd = U2 - U1
    admm_times = []
    for admm_iter in range(3):
        t0 = time.perf_counter()
        U1_new, _, _, _ = subpb1_solver(
            U2, F2, dd, gd, coords, Df, imgs[0], imgs[1],
            para.mu, beta, para, para.tol, precomputed=cache_sp1,
        )
        F1_new = F2.copy()
        U2_new = subpb2_solver(mesh, para.gauss_pt_order, beta, para.mu,
                                U1_new, F1_new, gd, dd, 0.0, step, precomputed=cache_sp2)
        F2_new = global_nodal_strain_fem(mesh, para, U2_new)
        gd = F2_new - F1_new
        dd = U2_new - U1_new
        U1, F1, U2, F2 = U1_new, F1_new, U2_new, F2_new
        admm_times.append(time.perf_counter() - t0)

    timings["admm_avg"] = np.mean(admm_times)
    timings["admm_total"] = sum(admm_times)

    total = (timings["local_icgn"] + timings["pre_subpb1"] + timings["pre_subpb2"]
             + timings["subpb2_solve"] + timings["fem_strain"] + timings["admm_total"])
    timings["total"] = total

    print(f"  Local IC-GN:       {timings['local_icgn']:7.3f}s")
    print(f"  Precompute subpb1: {timings['pre_subpb1']:7.3f}s")
    print(f"  Precompute subpb2: {timings['pre_subpb2']:7.3f}s")
    print(f"  Subpb2 solve (1x): {timings['subpb2_solve']:7.3f}s")
    print(f"  FEM strain (1x):   {timings['fem_strain']:7.3f}s")
    print(f"  ADMM x3:           {timings['admm_total']:7.3f}s  (avg {timings['admm_avg']:.3f}s)")
    print(f"  TOTAL:             {total:7.3f}s")
    print()

    results.append({
        "label": label, "n_nodes": n_nodes, "n_ele": n_ele,
        **timings,
    })

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 120)
print("SCALING SUMMARY")
print("=" * 120)

header = f"{'Config':<20s} {'Nodes':>8s} {'Elems':>8s} | {'IC-GN':>7s} {'PreSp1':>7s} {'PreSp2':>7s} {'Sp2Sol':>7s} {'Strain':>7s} {'ADMMx3':>7s} | {'TOTAL':>7s}"
print(header)
print("-" * 120)

for r in results:
    row = (
        f"{r['label']:<20s} {r['n_nodes']:>8,} {r['n_ele']:>8,} | "
        f"{r['local_icgn']:>7.3f} {r['pre_subpb1']:>7.3f} {r['pre_subpb2']:>7.3f} "
        f"{r['subpb2_solve']:>7.3f} {r['fem_strain']:>7.3f} {r['admm_total']:>7.3f} | "
        f"{r['total']:>7.3f}"
    )
    print(row)

print("-" * 120)

# Per-node scaling
print("\n" + "=" * 100)
print("PER-NODE COST (microseconds)")
print("=" * 100)

header2 = f"{'Config':<20s} {'Nodes':>8s} | {'IC-GN':>8s} {'PreSp1':>8s} {'PreSp2':>8s} {'Sp2Sol':>8s} {'Strain':>8s} {'ADMM/it':>8s} | {'Total':>8s}"
print(header2)
print("-" * 100)

for r in results:
    n = r["n_nodes"]
    row = (
        f"{r['label']:<20s} {n:>8,} | "
        f"{r['local_icgn']/n*1e6:>8.1f} {r['pre_subpb1']/n*1e6:>8.1f} {r['pre_subpb2']/n*1e6:>8.1f} "
        f"{r['subpb2_solve']/n*1e6:>8.1f} {r['fem_strain']/n*1e6:>8.1f} {r['admm_avg']/n*1e6:>8.1f} | "
        f"{r['total']/n*1e6:>8.1f}"
    )
    print(row)

print("-" * 100)

# Percentage breakdown
print("\n" + "=" * 100)
print("PERCENTAGE BREAKDOWN")
print("=" * 100)

header3 = f"{'Config':<20s} {'Nodes':>8s} | {'IC-GN':>6s} {'PreSp1':>6s} {'PreSp2':>6s} {'Sp2Sol':>6s} {'Strain':>6s} {'ADMMx3':>6s} | {'Total':>7s}"
print(header3)
print("-" * 100)

for r in results:
    t = r["total"]
    row = (
        f"{r['label']:<20s} {r['n_nodes']:>8,} | "
        f"{r['local_icgn']/t*100:>5.1f}% {r['pre_subpb1']/t*100:>5.1f}% {r['pre_subpb2']/t*100:>5.1f}% "
        f"{r['subpb2_solve']/t*100:>5.1f}% {r['fem_strain']/t*100:>5.1f}% {r['admm_total']/t*100:>5.1f}% | "
        f"{t:>7.3f}s"
    )
    print(row)

print("-" * 100)
