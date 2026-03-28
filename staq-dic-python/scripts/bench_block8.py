"""Benchmark 8x8 block assembly optimization for precompute_subpb2.

Measures precompute_subpb2 + subpb2_solver + full pipeline timing.
"""

import sys
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.io.image_ops import compute_image_gradient, normalize_images
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.solver.local_icgn import local_icgn
from staq_dic.solver.subpb2_solver import precompute_subpb2, subpb2_solver
from staq_dic.solver.subpb1_solver import precompute_subpb1, subpb1_solver
from staq_dic.strain.nodal_strain_fem import global_nodal_strain_fem
from dataclasses import replace

# ============================================================
# Setup
# ============================================================
H, W, step, ws = 1024, 1024, 2, 16
print(f"Benchmark: {H}x{W}, step={step}, winsize={ws}")

rng = np.random.default_rng(42)
f_img = gaussian_filter(rng.standard_normal((H, W)), sigma=2.0)
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
print(f"Nodes: {n_nodes}, Elements: {n_ele}")

# Initial IC-GN
print("\nRunning Local IC-GN (warmup + timing)...")
U0 = np.zeros(2 * n_nodes)
U0[0::2] = 1.0
t0 = time.perf_counter()
U1, F1, _, _, _, _ = local_icgn(U0, coords, Df, imgs[0], imgs[1], para, para.tol)
t_icgn = time.perf_counter() - t0
print(f"  Local IC-GN: {t_icgn:.3f}s")

# ============================================================
# Benchmark: precompute_subpb2
# ============================================================
beta = 1e-3 * step ** 2 * para.mu
print(f"\n{'='*65}")
print("BENCHMARK: precompute_subpb2 (8x8 block assembly)")
print(f"{'='*65}")

times = []
for i in range(3):
    t0 = time.perf_counter()
    cache = precompute_subpb2(mesh, para.gauss_pt_order, beta, para.mu, 0.0)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Run {i+1}: {times[-1]:.3f}s")

print(f"  Average: {np.mean(times):.3f}s  (min: {min(times):.3f}s)")

# ============================================================
# Benchmark: subpb2_solver (one call)
# ============================================================
print(f"\n{'='*65}")
print("BENCHMARK: subpb2_solver (one cached solve)")
print(f"{'='*65}")

gd = np.zeros(4 * n_nodes)
dd = np.zeros(2 * n_nodes)
times_solve = []
for i in range(3):
    t0 = time.perf_counter()
    U2 = subpb2_solver(mesh, para.gauss_pt_order, beta, para.mu,
                        U1, F1, gd, dd, 0.0, step, precomputed=cache)
    t1 = time.perf_counter()
    times_solve.append(t1 - t0)
    print(f"  Run {i+1}: {times_solve[-1]:.3f}s")

print(f"  Average: {np.mean(times_solve):.3f}s  (min: {min(times_solve):.3f}s)")

# ============================================================
# Benchmark: Full pipeline (S4 + S5 + 3x ADMM)
# ============================================================
print(f"\n{'='*65}")
print("BENCHMARK: Full pipeline (S4 + precompute + S5 + 3x ADMM)")
print(f"{'='*65}")

t_total_start = time.perf_counter()

# S4: local_icgn (already done)
from staq_dic.io.image_ops import ImageGradients
# Precompute subpb1
t0 = time.perf_counter()
cache_sp1 = precompute_subpb1(coords, Df, imgs[0], para)
t_pre_sp1 = time.perf_counter() - t0
print(f"  Precompute subpb1:   {t_pre_sp1:.3f}s")

# Precompute subpb2
t0 = time.perf_counter()
cache_sp2 = precompute_subpb2(mesh, para.gauss_pt_order, beta, para.mu, 0.0)
t_pre_sp2 = time.perf_counter() - t0
print(f"  Precompute subpb2:   {t_pre_sp2:.3f}s")

# S5: first subpb2 + strain
gd = np.zeros(4 * n_nodes)
dd = np.zeros(2 * n_nodes)
t0 = time.perf_counter()
U2 = subpb2_solver(mesh, para.gauss_pt_order, beta, para.mu,
                    U1, F1, gd, dd, 0.0, step, precomputed=cache_sp2)
F2 = global_nodal_strain_fem(mesh, para, U2)
t_s5 = time.perf_counter() - t0
print(f"  S5 (subpb2+strain):  {t_s5:.3f}s")

# ADMM x3
gd = F2 - F1
dd = U2 - U1
admm_times = []
for admm_iter in range(3):
    t0 = time.perf_counter()
    # Subpb1
    U1_new, _, _, _ = subpb1_solver(
        U2, F2, dd, gd, coords, Df, imgs[0], imgs[1],
        para.mu, beta, para, para.tol, precomputed=cache_sp1,
    )
    F1_new = F2.copy()
    # Subpb2
    U2_new = subpb2_solver(mesh, para.gauss_pt_order, beta, para.mu,
                            U1_new, F1_new, gd, dd, 0.0, step, precomputed=cache_sp2)
    F2_new = global_nodal_strain_fem(mesh, para, U2_new)
    # Dual update
    gd = F2_new - F1_new
    dd = U2_new - U1_new
    U1, F1, U2, F2 = U1_new, F1_new, U2_new, F2_new
    t_admm = time.perf_counter() - t0
    admm_times.append(t_admm)
    print(f"  ADMM iter {admm_iter+1}:        {t_admm:.3f}s")

t_total = time.perf_counter() - t_total_start + t_icgn

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  Local IC-GN:         {t_icgn:.3f}s")
print(f"  Precompute subpb1:   {t_pre_sp1:.3f}s")
print(f"  Precompute subpb2:   {t_pre_sp2:.3f}s")
print(f"  S5 (subpb2+strain):  {t_s5:.3f}s")
print(f"  ADMM x3:             {sum(admm_times):.3f}s  (avg {np.mean(admm_times):.3f}s)")
print(f"  TOTAL:               {t_total:.3f}s")
