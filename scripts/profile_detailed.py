"""Detailed per-component profiling of the AL-DIC pipeline at ~247K nodes.

Creates a synthetic 1024x1024 speckle image pair with known 1px translation,
sets up a uniform mesh at step=2, and individually times each sub-operation.
"""

import sys
import time
import warnings

import numpy as np

# Suppress all warnings for clean output
warnings.filterwarnings("ignore")

# Add source to path
sys.path.insert(0, "src")

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import DICMesh, GridxyROIRange
from al_dic.io.image_ops import compute_image_gradient, normalize_images
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.solver.icgn_batch import precompute_subsets_6dof, precompute_subsets_2dof
from al_dic.utils.outlier_detection import detect_bad_points, fill_nan_idw
from al_dic.solver.fem_assembly import compute_all_elements_gp
from al_dic.solver.subpb2_solver import (
    _precompute_geometry, precompute_subpb2, _solve_cached,
)
from al_dic.strain.nodal_strain_fem import (
    global_nodal_strain_fem, _detect_outliers_movmedian,
)
from al_dic.utils.region_analysis import precompute_node_regions

from dataclasses import replace


def make_speckle(h, w, seed=42):
    """Generate a synthetic speckle pattern."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((h, w))
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(base, sigma=2.0)


def timer(fn, label, repeats=1):
    """Time a function call and print result."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = np.mean(times)
    print(f"  {label:55s} {avg:8.3f}s")
    return result, avg


# ============================================================
# Setup
# ============================================================
print("=" * 72)
print("AL-DIC Detailed Profiling - 1024x1024, step=2 (~247K nodes)")
print("=" * 72)

H, W = 1024, 1024
step = 2
winsize = 16

print("\n[Setup] Generating synthetic images...")
f_img = make_speckle(H, W, seed=42)
g_img = np.roll(f_img, 1, axis=1)  # 1px horizontal shift

roi = GridxyROIRange(
    gridx=(winsize, W - winsize - 1),
    gridy=(winsize, H - winsize - 1),
)
para = dicpara_default(
    winsize=winsize, winstepsize=step, winsize_min=step,
    gridxy_roi_range=roi, img_size=(H, W),
    tol=1e-3, icgn_max_iter=50, admm_max_iter=4,
    mu=1e-3, alpha=0.0,
)

# Normalize
images = [f_img, g_img]
masks = [np.ones((H, W)), np.ones((H, W))]
img_norm, clamped_roi = normalize_images(images, roi)
para = replace(para, gridxy_roi_range=clamped_roi)

f_norm = img_norm[0]
g_norm = img_norm[1]
f_mask = masks[0].astype(np.float64)

print("[Setup] Computing image gradients...")
Df = compute_image_gradient(f_norm, f_mask)

print("[Setup] Building mesh...")
x0 = np.arange(roi.gridx[0], roi.gridx[1] + 1, step, dtype=np.float64)
y0 = np.arange(roi.gridy[0], roi.gridy[1] + 1, step, dtype=np.float64)
mesh = mesh_setup(x0, y0, para)
coords = mesh.coordinates_fem
elems = mesh.elements_fem
n_nodes = coords.shape[0]
n_ele = elems.shape[0]
print(f"  Nodes: {n_nodes}, Elements: {n_ele}")

# Initial displacement: close to ground truth (1px in x)
U0 = np.zeros(2 * n_nodes, dtype=np.float64)
U0[0::2] = 1.0  # ~1px u guess

# Warm up Numba JIT (not included in timing)
print("\n[Warmup] Compiling Numba kernels (first call)...")
from al_dic.solver.numba_kernels import HAS_NUMBA, icgn_6dof_parallel, icgn_2dof_parallel
if HAS_NUMBA:
    # Quick warmup with tiny data
    _c = np.array([[50.0, 50.0], [80.0, 80.0]])
    _pre_warmup = precompute_subsets_6dof(_c, f_norm, Df.df_dx, Df.df_dy, f_mask, winsize)
    icgn_6dof_parallel(
        _c, np.array([0.5, 0.5]), np.array([0.0, 0.0]),
        _pre_warmup["ref_all"], _pre_warmup["gx_all"], _pre_warmup["gy_all"],
        _pre_warmup["mask_all"], _pre_warmup["XX_all"], _pre_warmup["YY_all"],
        _pre_warmup["H_all"], _pre_warmup["meanf_all"], _pre_warmup["bottomf_all"],
        _pre_warmup["valid"], g_norm, 1e-3, 50,
    )
    # 2-DOF warmup
    _ws_x = np.full(2, winsize, dtype=int)
    _ws_y = np.full(2, winsize, dtype=int)
    _pre2w = precompute_subsets_2dof(_c, f_norm, Df.df_dx, Df.df_dy, f_mask, _ws_x, _ws_y)
    _U_old = np.zeros((2, 2))
    _F_old = np.zeros((2, 4))
    _ud = np.zeros((2, 2))
    icgn_2dof_parallel(
        _c, _U_old, _F_old, _ud,
        _pre2w["ref_all"], _pre2w["gx_all"], _pre2w["gy_all"], _pre2w["mask_all"],
        _pre2w["XX_all"], _pre2w["YY_all"], _pre2w["H2_img_all"],
        _pre2w["meanf_all"], _pre2w["bottomf_all"],
        _pre2w["valid"], g_norm, 1e-3, 1e-3, 50,
    )
    print("  Numba JIT warmup complete.")
else:
    print("  Numba not available, using Python fallback.")

timings = {}

# ============================================================
# A. LOCAL IC-GN 6-DOF
# ============================================================
print("\n" + "=" * 72)
print("A. LOCAL IC-GN 6-DOF")
print("=" * 72)

# A.1 Precompute subsets
pre6, t = timer(
    lambda: precompute_subsets_6dof(coords, f_norm, Df.df_dx, Df.df_dy, f_mask, winsize),
    "A.1 precompute_subsets_6dof (Numba prange)",
)
timings["A1_precompute_6dof"] = t

# A.2 IC-GN iterations (Numba prange)
U0_2d = U0.reshape(-1, 2)
rounded_coords = np.round(coords).astype(np.float64)


def run_icgn6():
    return icgn_6dof_parallel(
        rounded_coords,
        U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
        pre6["ref_all"], pre6["gx_all"], pre6["gy_all"], pre6["mask_all"],
        pre6["XX_all"], pre6["YY_all"], pre6["H_all"],
        pre6["meanf_all"], pre6["bottomf_all"],
        pre6["valid"], g_norm, para.tol, para.icgn_max_iter,
    )


(P_out, conv6), t = timer(run_icgn6, "A.2 icgn_6dof_parallel (Numba prange)")
timings["A2_icgn6dof"] = t

# A.3 Assemble U, F vectors
def assemble_uf():
    U = np.empty(2 * n_nodes, dtype=np.float64)
    U[0::2] = P_out[:, 4]
    U[1::2] = P_out[:, 5]
    F = np.empty(4 * n_nodes, dtype=np.float64)
    F[0::4] = P_out[:, 0]
    F[1::4] = P_out[:, 1]
    F[2::4] = P_out[:, 2]
    F[3::4] = P_out[:, 3]
    return U, F

(U_local, F_local), t = timer(assemble_uf, "A.3 Assemble interleaved U, F")
timings["A3_assemble"] = t

# A.4 Detect bad points
mark_hole = pre6["mark_hole"]
mark_hole_strain = np.where(mark_hole)[0].astype(np.int64)

def run_detect_bad():
    return detect_bad_points(conv6, para.icgn_max_iter, coords, sigma_factor=1.0, min_threshold=6)

(bad_pts, bad_pt_num), t = timer(run_detect_bad, "A.4 detect_bad_points")
timings["A4_detect_bad"] = t

# A.5 Set NaN + fill_nan_idw (U)
U_with_nan = U_local.copy()
U_with_nan[2 * bad_pts] = np.nan
U_with_nan[2 * bad_pts + 1] = np.nan
F_with_nan = F_local.copy()
F_with_nan[4 * bad_pts] = np.nan
F_with_nan[4 * bad_pts + 1] = np.nan
F_with_nan[4 * bad_pts + 2] = np.nan
F_with_nan[4 * bad_pts + 3] = np.nan

_, t = timer(lambda: fill_nan_idw(U_with_nan, coords, n_components=2),
             "A.5 fill_nan_idw (U, n_comp=2)")
timings["A5_fill_nan_U"] = t
U_filled = fill_nan_idw(U_with_nan, coords, n_components=2)

_, t = timer(lambda: fill_nan_idw(F_with_nan, coords, n_components=4),
             "A.6 fill_nan_idw (F, n_comp=4)")
timings["A6_fill_nan_F"] = t
F_filled = fill_nan_idw(F_with_nan, coords, n_components=4)

n_nan_u = np.sum(np.isnan(U_with_nan[0::2]))
print(f"  (bad points: {bad_pt_num}, NaN nodes for fill: {n_nan_u})")

A_total = sum(timings[k] for k in timings if k.startswith("A"))
print(f"  {'A. TOTAL LOCAL IC-GN 6-DOF':55s} {A_total:8.3f}s")

# ============================================================
# B. FEM STRAIN (global_nodal_strain_fem) - detailed
# ============================================================
print("\n" + "=" * 72)
print("B. FEM STRAIN (global_nodal_strain_fem)")
print("=" * 72)

# Use U_filled as input for strain computation
U_for_strain = U_filled.copy()

# B.1 Gather element node coordinates
def gather_coords():
    ptx = np.zeros((n_ele, 8), dtype=np.float64)
    pty = np.zeros((n_ele, 8), dtype=np.float64)
    for k in range(8):
        valid = elems[:, k] >= 0
        ptx[valid, k] = coords[elems[valid, k], 0]
        pty[valid, k] = coords[elems[valid, k], 1]
    delta = (elems[:, 4:8] >= 0).astype(np.float64)
    return ptx, pty, delta

(ptx, pty, delta), t = timer(gather_coords, "B.1 Gather element coords + delta")
timings["B1_gather_coords"] = t

# B.2 Build DOF index arrays
def build_dof_indices():
    dummy_node = n_nodes
    all_index_u = np.zeros((n_ele, 16), dtype=np.int64)
    for k in range(8):
        node_ids = elems[:, k].copy()
        node_ids[node_ids < 0] = dummy_node
        all_index_u[:, 2 * k] = 2 * node_ids
        all_index_u[:, 2 * k + 1] = 2 * node_ids + 1
    return all_index_u

all_idx_u, t = timer(build_dof_indices, "B.2 Build DOF index arrays")
timings["B2_dof_indices"] = t

# B.3 Gather element displacements
def gather_ele_disp():
    U_ext = np.concatenate([U_for_strain, np.zeros(2, dtype=np.float64)])
    return U_ext[all_idx_u]

U_ele, t = timer(gather_ele_disp, "B.3 Gather element displacements")
timings["B3_gather_disp"] = t

# B.4 compute_all_elements_gp at centroid
_, t = timer(
    lambda: compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, n_ele),
    "B.4 compute_all_elements_gp (ksi=0, eta=0)",
)
timings["B4_shape_funcs"] = t
_, DN_all, Jdet = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, n_ele)

# B.5 Displacement gradients at element center
def compute_gradients():
    dNdx = DN_all[:, 0, 0::2]
    dNdy = DN_all[:, 1, 0::2]
    u_ele_comp = U_ele[:, 0::2]
    v_ele_comp = U_ele[:, 1::2]
    dudx = np.sum(dNdx * u_ele_comp, axis=1)
    dudy = np.sum(dNdy * u_ele_comp, axis=1)
    dvdx = np.sum(dNdx * v_ele_comp, axis=1)
    dvdy = np.sum(dNdy * v_ele_comp, axis=1)
    return dudx, dudy, dvdx, dvdy

(dudx, dudy, dvdx, dvdy), t = timer(compute_gradients, "B.5 Displacement gradients (dNdx*u_ele)")
timings["B5_gradients"] = t

# B.6 Area-weighted averaging (add.at)
ele_area = np.abs(Jdet)

def area_weighted_avg():
    F_nodal = np.zeros((n_nodes, 4), dtype=np.float64)
    node_weight = np.zeros(n_nodes, dtype=np.float64)
    for k in range(8):
        nids = elems[:, k]
        valid = nids >= 0
        ni = nids[valid]
        w = ele_area[valid]
        np.add.at(F_nodal[:, 0], ni, w * dudx[valid])
        np.add.at(F_nodal[:, 1], ni, w * dvdx[valid])
        np.add.at(F_nodal[:, 2], ni, w * dudy[valid])
        np.add.at(F_nodal[:, 3], ni, w * dvdy[valid])
        np.add.at(node_weight, ni, w)
    orphan = node_weight < 1e-15
    node_weight[orphan] = 1.0
    F_nodal /= node_weight[:, None]
    F_nodal[orphan, :] = np.nan
    return F_nodal

F_nodal, t = timer(area_weighted_avg, "B.6 Area-weighted averaging (add.at x8)")
timings["B6_add_at"] = t

# B.7 Interleave
def interleave_F():
    F_out = np.empty(4 * n_nodes, dtype=np.float64)
    F_out[0::4] = F_nodal[:, 0]
    F_out[1::4] = F_nodal[:, 1]
    F_out[2::4] = F_nodal[:, 2]
    F_out[3::4] = F_nodal[:, 3]
    return F_out

F_out, t = timer(interleave_F, "B.7 Interleave F vector")
timings["B7_interleave"] = t

# B.8 Outlier detection (movmedian)
window = 1 + para.winstepsize

def detect_outliers():
    o1 = _detect_outliers_movmedian(F_out[0::4], window)
    o2 = _detect_outliers_movmedian(F_out[1::4], window)
    o3 = _detect_outliers_movmedian(F_out[2::4], window)
    o4 = _detect_outliers_movmedian(F_out[3::4], window)
    return o1 | o2 | o3 | o4

outlier_any, t = timer(detect_outliers, "B.8 Outlier detection (movmedian x4)")
timings["B8_outlier"] = t

# B.9 Set NaN + fill_nan_idw (F, n_comp=4)
outlier_idx = np.where(outlier_any)[0]
F_out_nan = F_out.copy()
for c in range(4):
    F_out_nan[4 * outlier_idx + c] = np.nan

n_nan_strain = len(outlier_idx)

_, t = timer(lambda: fill_nan_idw(F_out_nan, coords, n_components=4),
             "B.9 fill_nan_idw (F strain, n_comp=4)")
timings["B9_fill_nan_strain"] = t
print(f"  (outlier nodes: {n_nan_strain})")

B_total = sum(timings[k] for k in timings if k.startswith("B"))
print(f"  {'B. TOTAL FEM STRAIN':55s} {B_total:8.3f}s")

# ============================================================
# C. PRECOMPUTE SUBPB1 (2-DOF subsets)
# ============================================================
print("\n" + "=" * 72)
print("C. PRECOMPUTE SUBPB1")
print("=" * 72)

winsize_x_arr = np.full(n_nodes, winsize, dtype=int)
winsize_y_arr = np.full(n_nodes, winsize, dtype=int)

pre2, t = timer(
    lambda: precompute_subsets_2dof(coords, f_norm, Df.df_dx, Df.df_dy, f_mask,
                                     winsize_x_arr, winsize_y_arr),
    "C.1 precompute_subsets_2dof (Numba prange)",
)
timings["C1_precompute_2dof"] = t

C_total = t
print(f"  {'C. TOTAL PRECOMPUTE SUBPB1':55s} {C_total:8.3f}s")

# ============================================================
# D. PRECOMPUTE SUBPB2 (FEM stiffness + ILU)
# ============================================================
print("\n" + "=" * 72)
print("D. PRECOMPUTE SUBPB2")
print("=" * 72)

mu_val = para.mu
beta_val = 1e-3 * step ** 2 * mu_val
alpha_val = 0.0
gp_order = para.gauss_pt_order

# D.1 Element coords + delta
def d1_coords():
    c = mesh.coordinates_fem
    e = mesh.elements_fem
    ne = e.shape[0]
    ptx = np.zeros((ne, 8), dtype=np.float64)
    pty = np.zeros((ne, 8), dtype=np.float64)
    for k in range(8):
        v = e[:, k] >= 0
        ptx[v, k] = c[e[v, k], 0]
        pty[v, k] = c[e[v, k], 1]
    delta = (e[:, 4:8] >= 0).astype(np.float64)
    return ptx, pty, delta

_, t = timer(d1_coords, "D.1 Element coords + delta")
timings["D1_elem_coords"] = t

# D.2 DOF index arrays + COO indices
def d2_dof():
    e = mesh.elements_fem
    ne = e.shape[0]
    dummy = n_nodes
    aiu = np.zeros((ne, 16), dtype=np.int64)
    aif = np.zeros((ne, 16, 4), dtype=np.int64)
    for k in range(8):
        nids = e[:, k].copy()
        nids[nids < 0] = dummy
        aiu[:, 2*k] = 2*nids
        aiu[:, 2*k+1] = 2*nids+1
        fb = 4*nids
        for col in (2*k, 2*k+1):
            aif[:, col, 0] = fb
            aif[:, col, 1] = fb+2
            aif[:, col, 2] = fb+1
            aif[:, col, 3] = fb+3
    lr, lc = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    ti = aiu[:, lr.ravel()].ravel()
    tj = aiu[:, lc.ravel()].ravel()
    return aiu, aif, ti, tj

_, t = timer(d2_dof, "D.2 DOF indices + COO arrays")
timings["D2_dof_coo"] = t

# D.3 Gauss-point shape functions + einsum NtN/DtD
(ptx_d, pty_d, delta_d) = d1_coords()

from al_dic.solver.subpb2_solver import _gauss_points_1d

def d3_shape_funcs():
    gqpt, gqwt = _gauss_points_1d(gp_order)
    ksi_g, eta_g = np.meshgrid(gqpt, gqpt, indexing="ij")
    wk_g, we_g = np.meshgrid(gqwt, gqwt, indexing="ij")
    gp_data = []
    for idx in range(len(ksi_g.ravel())):
        ksi = float(ksi_g.ravel()[idx])
        eta = float(eta_g.ravel()[idx])
        wk = float(wk_g.ravel()[idx])
        we = float(we_g.ravel()[idx])
        N_a, DN_a, Jd = compute_all_elements_gp(ksi, eta, ptx_d, pty_d, delta_d, n_ele)
        weight = Jd * (wk * we)
        NtN = np.einsum("eai,eaj->eij", N_a, N_a)
        DtD = np.einsum("eai,eaj->eij", DN_a, DN_a)
        gp_data.append({"N_all": N_a, "DN_all": DN_a, "NtN": NtN, "DtD": DtD, "weight": weight})
    return gp_data

gp_data, t = timer(d3_shape_funcs, f"D.3 Shape funcs + einsum ({gp_order}^2 Gauss pts)")
timings["D3_shape_einsum"] = t

# D.4 Stiffness matrix assembly
from scipy import sparse

def d4_stiffness():
    _, _, ti, tj = d2_dof()
    temp = np.zeros((n_ele, 256), dtype=np.float64)
    for gp in gp_data:
        NtN_f = gp["NtN"].reshape(n_ele, 256)
        DtD_f = gp["DtD"].reshape(n_ele, 256)
        temp += gp["weight"][:, None] * ((beta_val + alpha_val) * DtD_f + mu_val * NtN_f)
    big_n = 2 * n_nodes + 2
    A = sparse.coo_matrix((temp.ravel(), (ti, tj)), shape=(big_n, big_n)).tocsr()
    return A

A_mat, t = timer(d4_stiffness, "D.4 Stiffness assembly (COO -> CSR)")
timings["D4_stiffness"] = t

# D.5 Extract free-DOF submatrix
involved_nodes = np.unique(elems[elems >= 0])
involved_dofs = np.sort(np.concatenate([2*involved_nodes, 2*involved_nodes+1]))
free_dofs = involved_dofs  # no Dirichlet BCs in this test

def d5_extract_free():
    return A_mat[np.ix_(free_dofs, free_dofs)]

A_free, t = timer(d5_extract_free, f"D.5 Extract free-DOF submatrix ({len(free_dofs)} DOFs)")
timings["D5_extract_free"] = t

# D.6 ILU factorization
from scipy.sparse.linalg import spilu

def d6_ilu():
    return spilu(A_free.tocsc(), drop_tol=1e-3)

ilu, t = timer(d6_ilu, "D.6 ILU factorization (spilu)")
timings["D6_ilu"] = t

D_total = sum(timings[k] for k in timings if k.startswith("D"))
print(f"  {'D. TOTAL PRECOMPUTE SUBPB2':55s} {D_total:8.3f}s")

# ============================================================
# E. ADMM ITERATION (one iteration, detailed)
# ============================================================
print("\n" + "=" * 72)
print("E. ADMM ITERATION (single iteration)")
print("=" * 72)

# Setup: use IC-GN results as subpb1, simulate initial subpb2
U_subpb1 = U_filled.copy()
F_subpb1 = F_filled.copy()

# Run one full precompute_subpb2 to get proper cache
subpb2_cache = precompute_subpb2(mesh, gp_order, beta_val, mu_val, alpha_val)

# Run one subpb2 to get initial U_subpb2, F_subpb2
from al_dic.solver.subpb2_solver import subpb2_solver
grad_dual = np.zeros(4 * n_nodes, dtype=np.float64)
disp_dual = np.zeros(2 * n_nodes, dtype=np.float64)

U_subpb2 = subpb2_solver(
    mesh, gp_order, beta_val, mu_val,
    U_subpb1, F_subpb1, grad_dual, disp_dual, alpha_val, step,
    precomputed=subpb2_cache,
)
F_subpb2 = global_nodal_strain_fem(mesh, para, U_subpb2)
grad_dual = F_subpb2 - F_subpb1
disp_dual = U_subpb2 - U_subpb1

# Now time one full ADMM iteration
print("\n  --- E.1 Subproblem 1 (2-DOF IC-GN) ---")

# E.1a IC-GN dispatch (Numba)
U_old_2d = U_subpb2.reshape(-1, 2)
F_old_2d = F_subpb2.reshape(-1, 4)
udual_2d = disp_dual.reshape(-1, 2)

def e1a_icgn2dof():
    return icgn_2dof_parallel(
        rounded_coords, U_old_2d.copy(), F_old_2d.copy(), udual_2d.copy(),
        pre2["ref_all"], pre2["gx_all"], pre2["gy_all"], pre2["mask_all"],
        pre2["XX_all"], pre2["YY_all"], pre2["H2_img_all"],
        pre2["meanf_all"], pre2["bottomf_all"],
        pre2["valid"], g_norm, mu_val, para.tol, para.icgn_max_iter,
    )

(U_2dof_out, conv_2dof), t = timer(e1a_icgn2dof, "E.1a icgn_2dof_parallel (Numba prange)")
timings["E1a_icgn2dof"] = t

# E.1b Assemble + detect_bad_points
def e1b_assemble_detect():
    U_new = U_subpb2.copy()
    U_new[0::2] = U_2dof_out[:, 0]
    U_new[1::2] = U_2dof_out[:, 1]
    bp, bpn = detect_bad_points(conv_2dof, para.icgn_max_iter, coords,
                                 sigma_factor=para.outlier_sigma_factor,
                                 min_threshold=para.outlier_min_threshold)
    U_new[2*bp] = np.nan
    U_new[2*bp+1] = np.nan
    return U_new, bp, bpn

(U_subpb1_new, bp1, bpn1), t = timer(e1b_assemble_detect, "E.1b Assemble + detect_bad_points")
timings["E1b_detect"] = t

# E.1c fill_nan_idw (U only for subpb1)
_, t = timer(lambda: fill_nan_idw(U_subpb1_new, coords, n_components=2),
             f"E.1c fill_nan_idw (U, {np.sum(np.isnan(U_subpb1_new[0::2]))} NaN nodes)")
timings["E1c_fill_nan"] = t
U_subpb1_new = fill_nan_idw(U_subpb1_new, coords, n_components=2)

E1_total = timings["E1a_icgn2dof"] + timings["E1b_detect"] + timings["E1c_fill_nan"]
print(f"  {'E.1 TOTAL SUBPB1':55s} {E1_total:8.3f}s")

print("\n  --- E.2 FEM Strain (after subpb2) ---")
_, t = timer(lambda: global_nodal_strain_fem(mesh, para, U_subpb2),
             "E.2 global_nodal_strain_fem (full)")
timings["E2_fem_strain"] = t

print("\n  --- E.3 Subproblem 2 (_solve_cached) ---")

# E.3a Gather element vectors
def e3a_gather():
    big_n = subpb2_cache["big_n"]
    aiu = subpb2_cache["all_index_u"]
    aif = subpb2_cache["all_index_f"]
    ne = subpb2_cache["n_ele"]
    U_ext = np.concatenate([U_subpb1_new, np.zeros(2)])
    F_ext = np.concatenate([F_subpb1, np.zeros(4)])
    u_ele = U_ext[aiu]
    fmw = np.zeros((ne, 16, 4), dtype=np.float64)
    for c in range(4):
        fmw[:, :, c] = F_ext[aif[:, :, c]]
    return u_ele, fmw

(u_ele_s2, fmw_s2), t = timer(e3a_gather, "E.3a Gather element vectors")
timings["E3a_gather"] = t

# E.3b RHS assembly (einsum + accumulate)
def e3b_rhs():
    ne = subpb2_cache["n_ele"]
    b_v = np.zeros((ne, 16), dtype=np.float64)
    for gp in subpb2_cache["gp_data"]:
        w = gp["weight"]
        t1 = np.einsum("eci,eic->ei", gp["DN_all"], fmw_s2)
        t2 = np.einsum("eij,ej->ei", gp["NtN"], u_ele_s2)
        t3 = np.einsum("eij,ej->ei", gp["DtD"], u_ele_s2)
        b_v += w[:, None] * (beta_val * t1 + mu_val * t2 + alpha_val * t3)
    return b_v

b_v, t = timer(e3b_rhs, f"E.3b RHS einsum ({gp_order}^2 Gauss pts)")
timings["E3b_rhs"] = t

# E.3c Scatter-add to global
def e3c_scatter():
    big_n = subpb2_cache["big_n"]
    b = np.zeros(big_n, dtype=np.float64)
    np.add.at(b, subpb2_cache["all_index_u"].ravel(), b_v.ravel())
    return b

b_global, t = timer(e3c_scatter, "E.3c Scatter-add to global vector")
timings["E3c_scatter"] = t

# E.3d PCG solve
from scipy.sparse.linalg import LinearOperator, cg

def e3d_pcg():
    fd = subpb2_cache["free_dofs"]
    b_free = b_global[fd]
    ilu_cache = subpb2_cache.get("ilu")
    A_free_cache = subpb2_cache.get("A_free")
    if ilu_cache is not None:
        M_op = LinearOperator(A_free_cache.shape, ilu_cache.solve)
        x, info = cg(A_free_cache, b_free, rtol=1e-6, maxiter=1000, M=M_op)
        return x, info
    return None, -1

(x_sol, info), t = timer(e3d_pcg, f"E.3d PCG solve (ILU precond, {len(free_dofs)} DOFs)")
timings["E3d_pcg"] = t

E3_total = timings["E3a_gather"] + timings["E3b_rhs"] + timings["E3c_scatter"] + timings["E3d_pcg"]
print(f"  {'E.3 TOTAL SUBPB2 SOLVE':55s} {E3_total:8.3f}s")

print("\n  --- E.4 FEM Strain (after subpb2) ---")
_, t = timer(lambda: global_nodal_strain_fem(mesh, para, U_subpb2),
             "E.4 global_nodal_strain_fem (full)")
timings["E4_fem_strain"] = t

print("\n  --- E.5 Dual update + convergence check ---")
def e5_dual():
    gd = F_subpb2 - F_subpb1
    dd = U_subpb2 - U_subpb1
    nrm = np.linalg.norm(dd) / np.sqrt(len(dd))
    return gd, dd, nrm

_, t = timer(e5_dual, "E.5 Dual update + norm")
timings["E5_dual"] = t

E_total = E1_total + timings["E2_fem_strain"] + E3_total + timings["E4_fem_strain"] + timings["E5_dual"]
print(f"\n  {'E. TOTAL ONE ADMM ITERATION':55s} {E_total:8.3f}s")

# ============================================================
# F. IC-GN INTERNAL BREAKDOWN (FLOP estimates for Numba)
# ============================================================
print("\n" + "=" * 72)
print("F. IC-GN INTERNAL BREAKDOWN (FLOP estimates, Numba compiled)")
print("=" * 72)

Sx = winsize + 1  # 17
Sy = winsize + 1  # 17
n_pix = Sx * Sy   # 289
avg_iter_6dof = np.mean(conv6[conv6 > 0]) if np.any(conv6 > 0) else 5
avg_iter_2dof = np.mean(conv_2dof[conv_2dof > 0]) if np.any(conv_2dof > 0) else 3

print(f"\n  Subset size: {Sx}x{Sy} = {n_pix} pixels")
print(f"  Avg 6-DOF iterations: {avg_iter_6dof:.1f}")
print(f"  Avg 2-DOF iterations: {avg_iter_2dof:.1f}")
print(f"  Total nodes: {n_nodes}")

# Per-pixel costs
# Bicubic interp: 4x4 grid, each: 2 cubic_weight calls (2 branches, ~6 FLOP each)
#   + multiply + add = ~100 FLOP per pixel
# Warp coordinate: 6 FLOP per pixel (affine transform)
# ZNSSD residual: ~5 FLOP per pixel
# Gradient assembly: ~20 FLOP per pixel (6 components)
flops_warp_per_iter = n_pix * 106  # coord transform + bicubic
flops_znssd_per_iter = n_pix * 25  # mean, norm, residual
flops_gradient_per_iter = n_pix * 20  # H, gradient assembly
flops_solve_6x6 = 6 * 6 * 6  # ~216 FLOP Gaussian elimination
flops_compose = 36  # matrix operations

total_per_node_6dof = avg_iter_6dof * (flops_warp_per_iter + flops_znssd_per_iter + flops_gradient_per_iter + flops_solve_6x6 + flops_compose)
total_per_node_2dof = avg_iter_2dof * (flops_warp_per_iter + flops_znssd_per_iter + n_pix * 10 + 8 + 12)

print(f"\n  Per-node 6-DOF total FLOPs: {total_per_node_6dof:.0f}")
print(f"    Warp (bicubic interp):     {avg_iter_6dof * flops_warp_per_iter:.0f} ({avg_iter_6dof * flops_warp_per_iter / total_per_node_6dof * 100:.0f}%)")
print(f"    ZNSSD stats:               {avg_iter_6dof * flops_znssd_per_iter:.0f} ({avg_iter_6dof * flops_znssd_per_iter / total_per_node_6dof * 100:.0f}%)")
print(f"    Gradient assembly:         {avg_iter_6dof * flops_gradient_per_iter:.0f} ({avg_iter_6dof * flops_gradient_per_iter / total_per_node_6dof * 100:.0f}%)")
print(f"    6x6 solve:                 {avg_iter_6dof * flops_solve_6x6:.0f} (<1%)")
print(f"    Compose warp:              {avg_iter_6dof * flops_compose:.0f} (<1%)")

# ============================================================
# GRAND SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("GRAND SUMMARY: Full Pipeline (1 frame, 3 ADMM iterations)")
print("=" * 72)

# Estimate full pipeline time
local_icgn_total = A_total
precompute_subpb1_total = C_total
precompute_subpb2_total = D_total
# Section 5 = subpb2_solve(1x) + fem_strain(1x) + corrections
s5_subpb2_solve = E3_total
s5_fem_strain = timings["E2_fem_strain"]
# ADMM: 3 iterations of (subpb1 + subpb2 + 2*fem_strain + dual)
admm_per_iter = E_total
n_admm = 3

summary = {
    "A. Local IC-GN 6-DOF": A_total,
    "   A.1 Precompute subsets": timings["A1_precompute_6dof"],
    "   A.2 IC-GN iterations": timings["A2_icgn6dof"],
    "   A.3 Assemble U,F": timings["A3_assemble"],
    "   A.4 Detect bad points": timings["A4_detect_bad"],
    "   A.5 fill_nan_idw (U)": timings["A5_fill_nan_U"],
    "   A.6 fill_nan_idw (F)": timings["A6_fill_nan_F"],
    "B. FEM Strain (1 call)": B_total,
    "   B.1-3 Gather coords/DOF/disp": timings["B1_gather_coords"] + timings["B2_dof_indices"] + timings["B3_gather_disp"],
    "   B.4 Shape functions": timings["B4_shape_funcs"],
    "   B.5 Gradients": timings["B5_gradients"],
    "   B.6 Area-weighted avg": timings["B6_add_at"],
    "   B.7-8 Interleave + outlier": timings["B7_interleave"] + timings["B8_outlier"],
    "   B.9 fill_nan_idw (F)": timings["B9_fill_nan_strain"],
    "C. Precompute subpb1": C_total,
    "D. Precompute subpb2": D_total,
    "   D.1-2 Coords + DOF + COO": timings["D1_elem_coords"] + timings["D2_dof_coo"],
    "   D.3 Shape funcs + einsum": timings["D3_shape_einsum"],
    "   D.4 Stiffness assembly": timings["D4_stiffness"],
    "   D.5 Extract free-DOF": timings["D5_extract_free"],
    "   D.6 ILU factorization": timings["D6_ilu"],
    "E. ADMM iteration (x1)": E_total,
    "   E.1 Subpb1 (2-DOF IC-GN)": E1_total,
    "      E.1a IC-GN dispatch": timings["E1a_icgn2dof"],
    "      E.1b Detect bad pts": timings["E1b_detect"],
    "      E.1c fill_nan_idw": timings["E1c_fill_nan"],
    "   E.2 FEM strain": timings["E2_fem_strain"],
    "   E.3 Subpb2 solve": E3_total,
    "      E.3a Gather": timings["E3a_gather"],
    "      E.3b RHS einsum": timings["E3b_rhs"],
    "      E.3c Scatter": timings["E3c_scatter"],
    "      E.3d PCG solve": timings["E3d_pcg"],
    "   E.4 FEM strain (2nd)": timings["E4_fem_strain"],
    "   E.5 Dual update": timings["E5_dual"],
}

estimated_total = local_icgn_total + s5_subpb2_solve + s5_fem_strain + precompute_subpb1_total + precompute_subpb2_total + n_admm * admm_per_iter

print(f"\n{'Component':<55s} {'Time':>8s} {'%':>6s}")
print("-" * 72)
for label, val in summary.items():
    pct = val / estimated_total * 100
    indent = "  " if label.startswith("   ") else ""
    if label.startswith("      "):
        print(f"  {label:<53s} {val:8.3f}s {pct:5.1f}%")
    elif label.startswith("   "):
        print(f"  {label:<53s} {val:8.3f}s {pct:5.1f}%")
    else:
        print(f"{label:<55s} {val:8.3f}s {pct:5.1f}%")

print("-" * 72)
print(f"{'ESTIMATED TOTAL (S4+S5+precompute+3xADMM)':<55s} {estimated_total:8.3f}s  100%")
print(f"\nNOTE: Numba IC-GN internals (bicubic interp, ZNSSD, 6x6 solve)")
print(f"  are compiled into a single prange loop and cannot be individually")
print(f"  timed from Python. See section F for FLOP-based estimates.")
