"""Benchmark alternative fill_nan interpolation methods."""

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
from staq_dic.strain.nodal_strain_fem import _detect_outliers_movmedian
from staq_dic.solver.outlier_detection import fill_nan_rbf
from staq_dic.solver.fem_assembly import compute_all_elements_gp
from dataclasses import replace
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import LinearNDInterpolator

# ============================================================
# Setup: get real NaN data from subpb2 -> FEM strain
# ============================================================
H, W, step, ws = 1024, 1024, 2, 16
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
elems = mesh.elements_fem
n_nodes = coords.shape[0]
n_ele = elems.shape[0]

print("Setting up test data...")
U0 = np.zeros(2 * n_nodes)
U0[0::2] = 1.0
U1, F1, _, _, _, _ = local_icgn(U0, coords, Df, imgs[0], imgs[1], para, para.tol)
beta = 1e-3 * step ** 2 * para.mu
cache = precompute_subpb2(mesh, para.gauss_pt_order, beta, para.mu, 0.0)
gd = np.zeros(4 * n_nodes)
dd = np.zeros(2 * n_nodes)
U2 = subpb2_solver(
    mesh, para.gauss_pt_order, beta, para.mu,
    U1, F1, gd, dd, 0.0, step, precomputed=cache,
)

# Reproduce FEM strain up to the NaN stage
ptx = np.zeros((n_ele, 8))
pty = np.zeros((n_ele, 8))
for k in range(8):
    v = elems[:, k] >= 0
    ptx[v, k] = coords[elems[v, k], 0]
    pty[v, k] = coords[elems[v, k], 1]
delta = (elems[:, 4:8] >= 0).astype(np.float64)
dummy = n_nodes
aiu = np.zeros((n_ele, 16), dtype=np.int64)
for k in range(8):
    nids = elems[:, k].copy()
    nids[nids < 0] = dummy
    aiu[:, 2 * k] = 2 * nids
    aiu[:, 2 * k + 1] = 2 * nids + 1
U_ext = np.concatenate([U2, np.zeros(2)])
U_ele = U_ext[aiu]
_, DN, Jdet = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, n_ele)
dNdx = DN[:, 0, 0::2]
dNdy = DN[:, 1, 0::2]
u_e = U_ele[:, 0::2]
v_e = U_ele[:, 1::2]
dudx = np.sum(dNdx * u_e, axis=1)
dudy = np.sum(dNdy * u_e, axis=1)
dvdx = np.sum(dNdx * v_e, axis=1)
dvdy = np.sum(dNdy * v_e, axis=1)
ea = np.abs(Jdet)
F_n = np.zeros((n_nodes, 4))
nw = np.zeros(n_nodes)
for k in range(8):
    ni = elems[:, k]
    v = ni >= 0
    n = ni[v]
    w = ea[v]
    np.add.at(F_n[:, 0], n, w * dudx[v])
    np.add.at(F_n[:, 1], n, w * dvdx[v])
    np.add.at(F_n[:, 2], n, w * dudy[v])
    np.add.at(F_n[:, 3], n, w * dvdy[v])
    np.add.at(nw, n, w)
orph = nw < 1e-15
nw[orph] = 1
F_n /= nw[:, None]
F_n[orph] = np.nan
F_out = np.empty(4 * n_nodes)
F_out[0::4] = F_n[:, 0]
F_out[1::4] = F_n[:, 1]
F_out[2::4] = F_n[:, 2]
F_out[3::4] = F_n[:, 3]
w_sz = 1 + para.winstepsize
o1 = _detect_outliers_movmedian(F_out[0::4], w_sz)
o2 = _detect_outliers_movmedian(F_out[1::4], w_sz)
o3 = _detect_outliers_movmedian(F_out[2::4], w_sz)
o4 = _detect_outliers_movmedian(F_out[3::4], w_sz)
oa = o1 | o2 | o3 | o4
oi = np.where(oa)[0]
for c in range(4):
    F_out[4 * oi + c] = np.nan

# Extract test data
nan_idx = np.where(np.isnan(F_out[0::4]))[0]
not_nan_idx = np.setdiff1d(np.arange(n_nodes), nan_idx)
src_xy = coords[not_nan_idx]
dst_xy = coords[nan_idx]
n_nan = len(nan_idx)
n_good = len(not_nan_idx)
src_vals = F_out[4 * not_nan_idx + 0]  # F11 component

print(f"Good nodes: {n_good}, NaN nodes: {n_nan}")
print()

# ============================================================
# Method 0: Current implementation (full Delaunay)
# ============================================================
print("=" * 65)
print("Method 0: Full Delaunay + LinearNDInterpolator (CURRENT)")
print("=" * 65)
t0 = time.perf_counter()
tri = Delaunay(src_xy)
t_del = time.perf_counter()
tree0 = cKDTree(src_xy)
t_tree = time.perf_counter()
interp = LinearNDInterpolator(tri, src_vals)
filled0 = interp(dst_xy)
still = np.isnan(filled0)
if still.any():
    _, nn = tree0.query(dst_xy[still])
    filled0[still] = src_vals[nn]
t_end = time.perf_counter()
print(f"  Delaunay build:     {t_del - t0:.3f}s")
print(f"  cKDTree build:      {t_tree - t_del:.3f}s")
print(f"  Interp + NN fill:   {t_end - t_tree:.3f}s")
print(f"  TOTAL (1 call):     {t_end - t0:.3f}s")
print(f"  x6 ADMM calls:     {6 * (t_end - t0):.3f}s")
print()

# ============================================================
# Method 1: kNN IDW (k=8), tree rebuilt each call
# ============================================================
print("=" * 65)
print("Method 1: kNN IDW (k=8), tree rebuilt each call")
print("=" * 65)
t0 = time.perf_counter()
tree1 = cKDTree(src_xy)
t_tree = time.perf_counter()
dist1, idx1 = tree1.query(dst_xy, k=8)
t_query = time.perf_counter()
dist1 = np.maximum(dist1, 1e-10)
w1 = 1.0 / dist1 ** 2
w1 /= w1.sum(axis=1, keepdims=True)
filled1 = np.sum(w1 * src_vals[idx1], axis=1)
t_end = time.perf_counter()
print(f"  cKDTree build:      {t_tree - t0:.3f}s")
print(f"  k-NN query:         {t_query - t_tree:.3f}s")
print(f"  IDW compute:        {t_end - t_query:.3f}s")
print(f"  TOTAL (1 call):     {t_end - t0:.3f}s")
print(f"  x6 ADMM calls:     {6 * (t_end - t0):.3f}s")
err1 = np.abs(filled1 - filled0)
print(f"  vs Delaunay: max={err1.max():.2e}, mean={err1.mean():.2e}")
print()

# ============================================================
# Method 2: kNN IDW (k=8), tree CACHED (built once)
# ============================================================
print("=" * 65)
print("Method 2: kNN IDW (k=8), tree CACHED across 6 calls")
print("=" * 65)
t_build = time.perf_counter()
tree2 = cKDTree(src_xy)
t_built = time.perf_counter()
# Per-call cost
t0 = time.perf_counter()
dist2, idx2 = tree2.query(dst_xy, k=8)
dist2 = np.maximum(dist2, 1e-10)
w2 = 1.0 / dist2 ** 2
w2 /= w2.sum(axis=1, keepdims=True)
filled2 = np.sum(w2 * src_vals[idx2], axis=1)
t_call = time.perf_counter()
build_time = t_built - t_build
call_time = t_call - t0
print(f"  cKDTree build (once):  {build_time:.3f}s")
print(f"  Per-call query+IDW:    {call_time:.3f}s")
print(f"  x6 ADMM amortized:    {build_time + 6 * call_time:.3f}s")
err2 = np.abs(filled2 - filled0)
print(f"  vs Delaunay: max={err2.max():.2e}, mean={err2.mean():.2e}")
print()

# ============================================================
# Method 3: Pure nearest neighbor (k=1)
# ============================================================
print("=" * 65)
print("Method 3: Pure nearest neighbor (k=1)")
print("=" * 65)
t0 = time.perf_counter()
tree3 = cKDTree(src_xy)
_, nn3 = tree3.query(dst_xy, k=1)
filled3 = src_vals[nn3]
t_end = time.perf_counter()
print(f"  TOTAL (1 call):     {t_end - t0:.3f}s")
print(f"  x6 ADMM calls:     {6 * (t_end - t0):.3f}s")
err3 = np.abs(filled3 - filled0)
print(f"  vs Delaunay: max={err3.max():.2e}, mean={err3.mean():.2e}")
print()

# ============================================================
# Method 4: kNN IDW (k=4), cached tree
# ============================================================
print("=" * 65)
print("Method 4: kNN IDW (k=4), tree CACHED")
print("=" * 65)
t_build = time.perf_counter()
tree4 = cKDTree(src_xy)
t_built = time.perf_counter()
t0 = time.perf_counter()
dist4, idx4 = tree4.query(dst_xy, k=4)
dist4 = np.maximum(dist4, 1e-10)
w4 = 1.0 / dist4 ** 2
w4 /= w4.sum(axis=1, keepdims=True)
filled4 = np.sum(w4 * src_vals[idx4], axis=1)
t_call = time.perf_counter()
build_time = t_built - t_build
call_time = t_call - t0
print(f"  cKDTree build (once):  {build_time:.3f}s")
print(f"  Per-call query+IDW:    {call_time:.3f}s")
print(f"  x6 ADMM amortized:    {build_time + 6 * call_time:.3f}s")
err4 = np.abs(filled4 - filled0)
print(f"  vs Delaunay: max={err4.max():.2e}, mean={err4.mean():.2e}")
print()

# ============================================================
# Method 5: kNN IDW with CACHED kNN indices (query once, reuse)
# ============================================================
print("=" * 65)
print("Method 5: CACHED kNN indices + IDW (query once for all 6 calls)")
print("=" * 65)
t_build = time.perf_counter()
tree5 = cKDTree(src_xy)
dist5, idx5 = tree5.query(dst_xy, k=8)
t_built = time.perf_counter()
# Per-call: only IDW compute with different src_vals
t0 = time.perf_counter()
dist5c = np.maximum(dist5, 1e-10)
w5 = 1.0 / dist5c ** 2
w5 /= w5.sum(axis=1, keepdims=True)
# Simulate 4 components
for c in range(4):
    sv = F_out[4 * not_nan_idx + c]
    sv_nan = np.isnan(sv)
    if sv_nan.any():
        sv = sv.copy()
        sv[sv_nan] = 0.0
    vals_at_nn = sv[idx5]
    result = np.sum(w5 * vals_at_nn, axis=1)
t_call = time.perf_counter()
build_time = t_built - t_build
call_time = t_call - t0
print(f"  Build tree + query (once): {build_time:.3f}s")
print(f"  Per-call IDW (4 comps):    {call_time:.3f}s")
print(f"  x6 ADMM amortized:        {build_time + 6 * call_time:.3f}s")
err5 = np.abs(np.sum(w5 * src_vals[idx5], axis=1) - filled0)
print(f"  vs Delaunay: max={err5.max():.2e}, mean={err5.mean():.2e}")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 65)
print("SUMMARY: Total cost for 6 fill_nan calls in ADMM (n_comp=4)")
print("=" * 65)
# Reconstruct proper totals for n_comp=4
# Current: Delaunay + 4x LinearNDInterpolator per call, x6 calls
t0 = time.perf_counter()
F_filled_current = fill_nan_rbf(F_out.copy(), coords, n_components=4)
t_current_1call = time.perf_counter() - t0

print(f"  Current (Delaunay, per call):     {t_current_1call:.3f}s  x6 = {6*t_current_1call:.1f}s")
print(f"  Method 2 (kNN k=8, cached tree):  build {build_time:.3f}s + 6x{call_time:.3f}s = {build_time+6*call_time:.3f}s")
print(f"  Method 5 (cached tree+indices):   build {t_built-t_build:.3f}s + 6x{t_call-t0:.3f}s = {t_built-t_build+6*(t_call-t0):.3f}s")
print()

# Accuracy comparison
print("Accuracy (F11 component, vs Delaunay linear):")
print(f"  kNN k=8 IDW:  max err = {err2.max():.2e}, mean err = {err2.mean():.2e}")
print(f"  kNN k=4 IDW:  max err = {err4.max():.2e}, mean err = {err4.mean():.2e}")
print(f"  Nearest (k=1): max err = {err3.max():.2e}, mean err = {err3.mean():.2e}")

# Context: what are the actual F values?
print(f"\n  F11 value range: [{src_vals.min():.4f}, {src_vals.max():.4f}]")
print(f"  F11 std: {src_vals.std():.4f}")
print(f"  Relative error (kNN k=8): {err2.mean() / max(src_vals.std(), 1e-10):.2e}")
