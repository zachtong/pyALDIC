"""Benchmark optimization strategies for precompute_subpb2 sub-operations."""

import sys
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.solver.fem_assembly import compute_all_elements_gp
from staq_dic.solver.subpb2_solver import _gauss_points_1d

# ============================================================
# Setup
# ============================================================
H, W, step, ws = 1024, 1024, 2, 16
roi = GridxyROIRange(gridx=(ws, W - ws - 1), gridy=(ws, H - ws - 1))
para = dicpara_default(
    winsize=ws, winstepsize=step, winsize_min=step,
    gridxy_roi_range=roi, img_size=(H, W),
)
x0 = np.arange(roi.gridx[0], roi.gridx[1] + 1, step, dtype=np.float64)
y0 = np.arange(roi.gridy[0], roi.gridy[1] + 1, step, dtype=np.float64)
mesh = mesh_setup(x0, y0, para)
coords = mesh.coordinates_fem
elems = mesh.elements_fem
n_nodes = coords.shape[0]
n_ele = elems.shape[0]
gp_order = 2
beta, mu, alpha = 4e-9, 1e-3, 0.0
big_n = 2 * n_nodes + 2
dummy_node = n_nodes

print(f"Nodes: {n_nodes}, Elements: {n_ele}, DOFs: {2 * n_nodes}")
print()

# Pre-build common data
ptx = np.zeros((n_ele, 8), dtype=np.float64)
pty = np.zeros((n_ele, 8), dtype=np.float64)
for k in range(8):
    v = elems[:, k] >= 0
    ptx[v, k] = coords[elems[v, k], 0]
    pty[v, k] = coords[elems[v, k], 1]
delta = (elems[:, 4:8] >= 0).astype(np.float64)

all_index_u = np.zeros((n_ele, 16), dtype=np.int64)
for k in range(8):
    nids = elems[:, k].copy()
    nids[nids < 0] = dummy_node
    all_index_u[:, 2 * k] = 2 * nids
    all_index_u[:, 2 * k + 1] = 2 * nids + 1

local_r, local_c = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
trip_I = all_index_u[:, local_r.ravel()].ravel()
trip_J = all_index_u[:, local_c.ravel()].ravel()

# ============================================================
# D.3 Optimization: einsum vs matmul
# ============================================================
print("=" * 65)
print("D.3: einsum vs matmul for NtN / DtD")
print("=" * 65)

gqpt, gqwt = _gauss_points_1d(gp_order)
ksi_g, eta_g = np.meshgrid(gqpt, gqpt, indexing="ij")

# Compute one GP for testing
N_all, DN_all, Jdet = compute_all_elements_gp(
    float(ksi_g.ravel()[0]), float(eta_g.ravel()[0]),
    ptx, pty, delta, n_ele,
)

# einsum (current)
t0 = time.perf_counter()
for _ in range(4):  # simulate 4 GPs
    NtN_ein = np.einsum("eai,eaj->eij", N_all, N_all)
    DtD_ein = np.einsum("eai,eaj->eij", DN_all, DN_all)
t_einsum = time.perf_counter() - t0

# matmul: N^T @ N
t0 = time.perf_counter()
for _ in range(4):
    NtN_mat = np.matmul(N_all.transpose(0, 2, 1), N_all)
    DtD_mat = np.matmul(DN_all.transpose(0, 2, 1), DN_all)
t_matmul = time.perf_counter() - t0

print(f"  einsum x4 GPs:   {t_einsum:.3f}s")
print(f"  matmul x4 GPs:   {t_matmul:.3f}s  ({t_einsum/t_matmul:.1f}x)")
print(f"  Match: {np.allclose(NtN_ein, NtN_mat) and np.allclose(DtD_ein, DtD_mat)}")

# Exploit sparsity: N_all is (nEle, 2, 16) with known zero pattern
# N_all[:, 0, odd] = 0, N_all[:, 1, even] = 0
# So NtN[i,j] = N[0,i]*N[0,j] + N[1,i]*N[1,j]
# For even i, even j: NtN = N[0,i]*N[0,j] (since N[1,even]=0)
# This means NtN has block-diagonal structure!
t0 = time.perf_counter()
for _ in range(4):
    # N_all: (nEle, 2, 16). Non-zero: row0 at cols 0,2,4,...,14; row1 at 1,3,...,15
    N_even = N_all[:, 0, 0::2]  # (nEle, 8) - the u shape functions
    N_odd = N_all[:, 1, 1::2]   # (nEle, 8) - the v shape functions (same values!)
    # NtN[2i, 2j] = N_even[i]*N_even[j], NtN[2i+1, 2j+1] = N_odd[i]*N_odd[j]
    # NtN[2i, 2j+1] = 0, NtN[2i+1, 2j] = 0
    NN_block = np.einsum("ei,ej->eij", N_even, N_even)  # (nEle, 8, 8)
    # Expand to (nEle, 16, 16)
    NtN_sparse = np.zeros((n_ele, 16, 16), dtype=np.float64)
    NtN_sparse[:, 0::2, 0::2] = NN_block
    NtN_sparse[:, 1::2, 1::2] = NN_block  # Same because N_even == N_odd

    # DN_all: (nEle, 4, 16). Non-zero pattern:
    # row0 (du/dx): cols 0,2,...,14
    # row1 (du/dy): cols 0,2,...,14
    # row2 (dv/dx): cols 1,3,...,15
    # row3 (dv/dy): cols 1,3,...,15
    dNdx = DN_all[:, 0, 0::2]  # (nEle, 8)
    dNdy = DN_all[:, 1, 0::2]  # (nEle, 8)
    # DtD = DN^T @ DN. Since DN has block structure:
    # DtD[2i, 2j] = dNdx[i]*dNdx[j] + dNdy[i]*dNdy[j]
    # DtD[2i+1, 2j+1] = same
    # DtD[2i, 2j+1] = 0
    DD_block = (np.einsum("ei,ej->eij", dNdx, dNdx)
                + np.einsum("ei,ej->eij", dNdy, dNdy))
    DtD_sparse = np.zeros((n_ele, 16, 16), dtype=np.float64)
    DtD_sparse[:, 0::2, 0::2] = DD_block
    DtD_sparse[:, 1::2, 1::2] = DD_block
t_sparse = time.perf_counter() - t0
print(f"  sparse exploit:  {t_sparse:.3f}s  ({t_einsum/t_sparse:.1f}x)")
print(f"  Match NtN: {np.allclose(NtN_ein, NtN_sparse)}")
print(f"  Match DtD: {np.allclose(DtD_ein, DtD_sparse)}")

# Even better: only compute the 8x8 block, flatten to 64 instead of 256
t0 = time.perf_counter()
for _ in range(4):
    N_vals = N_all[:, 0, 0::2]  # (nEle, 8)
    NN8 = np.einsum("ei,ej->eij", N_vals, N_vals)  # (nEle, 8, 8)
    dNdx_ = DN_all[:, 0, 0::2]
    dNdy_ = DN_all[:, 1, 0::2]
    DD8 = np.einsum("ei,ej->eij", dNdx_, dNdx_) + np.einsum("ei,ej->eij", dNdy_, dNdy_)
t_block8 = time.perf_counter() - t0
print(f"  8x8 block only:  {t_block8:.3f}s  ({t_einsum/t_block8:.1f}x)")
print()

# ============================================================
# D.4 Optimization: COO->CSR alternatives
# ============================================================
print("=" * 65)
print("D.4: COO->CSR assembly alternatives")
print("=" * 65)

# Generate element stiffness values
gp_data = []
for idx in range(len(ksi_g.ravel())):
    ksi = float(ksi_g.ravel()[idx])
    eta = float(eta_g.ravel()[idx])
    N_a, DN_a, Jd = compute_all_elements_gp(ksi, eta, ptx, pty, delta, n_ele)
    gp_data.append((N_a, DN_a, Jd))

# Method A: Current COO -> CSR
t0 = time.perf_counter()
temp_a = np.zeros((n_ele, 256), dtype=np.float64)
for N_a, DN_a, Jd in gp_data:
    NtN = np.einsum("eai,eaj->eij", N_a, N_a)
    DtD = np.einsum("eai,eaj->eij", DN_a, DN_a)
    temp_a += Jd[:, None] * ((beta + alpha) * DtD.reshape(n_ele, 256) + mu * NtN.reshape(n_ele, 256))
A_coo = sparse.coo_matrix((temp_a.ravel(), (trip_I, trip_J)), shape=(big_n, big_n)).tocsr()
t_coo = time.perf_counter() - t0
print(f"  A. Current (COO->CSR):  {t_coo:.3f}s")

# Method B: Pre-compute CSR structure, then scatter-add
t0 = time.perf_counter()
# Step 1: Build CSR structure from topology (one-time)
ones_trip = np.ones(len(trip_I), dtype=np.float64)
A_struct = sparse.coo_matrix((ones_trip, (trip_I, trip_J)), shape=(big_n, big_n)).tocsr()
t_struct = time.perf_counter() - t0
print(f"  B1. Build CSR structure: {t_struct:.3f}s (one-time)")

# Step 2: Build scatter map: COO index -> CSR data index
t0 = time.perf_counter()
# For each COO entry (I,J), find position in CSR data
# CSR format: data[indptr[i]:indptr[i+1]] corresponds to row i
# indices[indptr[i]:indptr[i+1]] are the column indices
indptr = A_struct.indptr
indices = A_struct.indices

# Build reverse lookup: for row i, col j -> data position
# Use searchsorted on each row's column indices
scatter_map = np.empty(len(trip_I), dtype=np.int64)
for i in range(big_n):
    row_start = indptr[i]
    row_end = indptr[i + 1]
    if row_start == row_end:
        continue
    row_cols = indices[row_start:row_end]
    # Find all COO entries in this row
    mask = trip_I == i
    if not mask.any():
        continue
    coo_cols = trip_J[mask]
    positions = np.searchsorted(row_cols, coo_cols) + row_start
    scatter_map[mask] = positions
t_map = time.perf_counter() - t0
print(f"  B2. Build scatter map:  {t_map:.3f}s (one-time, SLOW - needs optimization)")

# Step 3: Actual assembly using scatter map
t0 = time.perf_counter()
csr_data = np.zeros(A_struct.nnz, dtype=np.float64)
np.add.at(csr_data, scatter_map, temp_a.ravel())
A_scatter = sparse.csr_matrix((csr_data, indices.copy(), indptr.copy()), shape=(big_n, big_n))
t_scatter = time.perf_counter() - t0
print(f"  B3. Scatter assembly:   {t_scatter:.3f}s (per-call)")
print(f"  B. Match: {np.allclose(A_coo.toarray()[:10,:10], A_scatter.toarray()[:10,:10])}")

# Method C: Use scipy.sparse.csr_array with duplicate handling
# csr_matrix constructor also handles duplicates
t0 = time.perf_counter()
A_direct = sparse.csr_matrix((temp_a.ravel(), (trip_I, trip_J)), shape=(big_n, big_n))
t_direct = time.perf_counter() - t0
print(f"  C. Direct CSR (skip COO): {t_direct:.3f}s")
print(f"  C. Match: {np.allclose(A_coo.toarray()[:10,:10], A_direct.toarray()[:10,:10])}")
print()

# ============================================================
# D.3+D.4 Combined: 8x8 block assembly
# ============================================================
print("=" * 65)
print("D.3+D.4 Combined: 8x8 block + reduced COO assembly")
print("=" * 65)

# Build 8-node index array (not 16-DOF)
node_index = np.zeros((n_ele, 8), dtype=np.int64)
for k in range(8):
    nids = elems[:, k].copy()
    nids[nids < 0] = dummy_node
    node_index[:, k] = nids

# 8x8 COO indices for node-based assembly
lr8, lc8 = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
trip_I8 = node_index[:, lr8.ravel()].ravel()
trip_J8 = node_index[:, lc8.ravel()].ravel()
# Each 8x8 block entry maps to a 2x2 block in the full matrix
# K_full[2*i, 2*j] = K_full[2*i+1, 2*j+1] = K8[i,j]
# K_full[2*i, 2*j+1] = K_full[2*i+1, 2*j] = 0

t0 = time.perf_counter()
temp8 = np.zeros((n_ele, 64), dtype=np.float64)
for N_a, DN_a, Jd in gp_data:
    N_v = N_a[:, 0, 0::2]  # (nEle, 8)
    dNdx_v = DN_a[:, 0, 0::2]
    dNdy_v = DN_a[:, 1, 0::2]
    NN8 = np.einsum("ei,ej->eij", N_v, N_v)
    DD8 = np.einsum("ei,ej->eij", dNdx_v, dNdx_v) + np.einsum("ei,ej->eij", dNdy_v, dNdy_v)
    temp8 += Jd[:, None] * ((beta + alpha) * DD8.reshape(n_ele, 64) + mu * NN8.reshape(n_ele, 64))

# Assemble into node-based matrix, then expand to DOF-based
n_node_total = n_nodes + 1
K8 = sparse.coo_matrix(
    (temp8.ravel(), (trip_I8, trip_J8)),
    shape=(n_node_total, n_node_total),
).tocsr()

# Expand to 2x2 block: kronecker with I_2
A_block = sparse.kron(K8, sparse.eye(2, format="csr"), format="csr")
t_block = time.perf_counter() - t0
print(f"  8x8 block assembly:  {t_block:.3f}s  (vs {t_coo:.3f}s current)")
print(f"  Speedup: {t_coo / t_block:.1f}x")

# Verify
print(f"  Shape: {A_block.shape} (expected {(big_n, big_n)})")
err = abs(A_coo[:big_n-2, :big_n-2] - A_block[:big_n-2, :big_n-2]).max()
print(f"  Max error vs current: {err:.2e}")
print()

# ============================================================
# D.6 Optimization: ILU alternatives
# ============================================================
print("=" * 65)
print("D.6: Preconditioner alternatives")
print("=" * 65)

from scipy.sparse.linalg import spilu, LinearOperator, cg

involved_nodes = np.unique(elems[elems >= 0])
involved_dofs = np.sort(np.concatenate([2 * involved_nodes, 2 * involved_nodes + 1]))
free_dofs = involved_dofs
A_free = A_coo[np.ix_(free_dofs, free_dofs)]
n_free = len(free_dofs)

# Current: ILU
t0 = time.perf_counter()
ilu = spilu(A_free.tocsc(), drop_tol=1e-3)
t_ilu = time.perf_counter() - t0
print(f"  ILU (drop_tol=1e-3):    {t_ilu:.3f}s")

# Diagonal (Jacobi) preconditioner
t0 = time.perf_counter()
diag = A_free.diagonal()
diag[diag == 0] = 1.0
M_diag = sparse.diags(1.0 / diag)
t_diag = time.perf_counter() - t0
print(f"  Diagonal (Jacobi):      {t_diag:.3f}s")

# Test PCG convergence with each preconditioner
b_test = np.random.default_rng(42).standard_normal(n_free)

t0 = time.perf_counter()
M_ilu = LinearOperator(A_free.shape, ilu.solve)
x_ilu, info_ilu = cg(A_free, b_test, rtol=1e-6, maxiter=1000, M=M_ilu)
t_pcg_ilu = time.perf_counter() - t0

t0 = time.perf_counter()
x_diag, info_diag = cg(A_free, b_test, rtol=1e-6, maxiter=1000, M=M_diag)
t_pcg_diag = time.perf_counter() - t0

t0 = time.perf_counter()
x_none, info_none = cg(A_free, b_test, rtol=1e-6, maxiter=1000)
t_pcg_none = time.perf_counter() - t0

print(f"\n  PCG solve comparison (rtol=1e-6):")
print(f"  ILU precond:  {t_pcg_ilu:.3f}s  (info={info_ilu})")
print(f"  Diag precond: {t_pcg_diag:.3f}s  (info={info_diag})")
print(f"  No precond:   {t_pcg_none:.3f}s  (info={info_none})")
print(f"  (info=0: converged, >0: not converged after N iters)")

# ILU with higher drop_tol (faster build, weaker preconditioner)
t0 = time.perf_counter()
ilu2 = spilu(A_free.tocsc(), drop_tol=1e-2)
t_ilu2 = time.perf_counter() - t0
print(f"\n  ILU (drop_tol=1e-2):    {t_ilu2:.3f}s")
M_ilu2 = LinearOperator(A_free.shape, ilu2.solve)
t0 = time.perf_counter()
x_ilu2, info_ilu2 = cg(A_free, b_test, rtol=1e-6, maxiter=1000, M=M_ilu2)
t_pcg_ilu2 = time.perf_counter() - t0
print(f"  PCG with ILU2: {t_pcg_ilu2:.3f}s  (info={info_ilu2})")

print()
print("=" * 65)
print("SUMMARY: Estimated total precompute_subpb2 with optimizations")
print("=" * 65)
print(f"  Current:          ~19.9s")
print(f"  D.3 8x8 block:   saves ~{6.2 - t_block + t_coo - t_coo:.1f}s -> ~{t_block:.1f}s for D.3+D.4 combined")
print(f"  Best estimate:    D.1-2({1.6:.1f}) + D.3+D.4({t_block:.1f}) + D.5(0.1) + D.6({t_ilu:.1f}) = {1.6 + t_block + 0.1 + t_ilu:.1f}s")
