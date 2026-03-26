"""ADMM subproblem 2 solver: FEM global kinematic compatibility.

Port of MATLAB solver/subpb2_solver.m (Jin Yang, Caltech).

Solves the ADMM subproblem 2 over a quadtree FEM mesh to find a globally
kinematically compatible deformation field.  Assembles the stiffness matrix
and load vector via Gauss quadrature over all elements, then solves the
resulting sparse linear system.

MATLAB/Python differences:
    - MATLAB ``pagemtimes`` → ``np.einsum`` batch matrix operations.
    - MATLAB COO sparse assembly → ``scipy.sparse.coo_matrix``.
    - MATLAB ``ichol`` + ``pcg`` → ``scipy.sparse.linalg.spilu`` + ``cg``.
    - MATLAB 1-based indices → Python 0-based throughout.
    - Missing midpoint nodes: MATLAB uses 0, Python uses -1.  Both map to
      a dummy node index for safe array indexing.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, spilu, spsolve

from ..core.data_structures import DICMesh
from .fem_assembly import compute_all_elements_gp


def _gauss_points_1d(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return 1D Gauss-Legendre quadrature points and weights.

    Args:
        order: Number of points per dimension (2-5).

    Returns:
        (points, weights) each of shape (order,).
    """
    if order == 2:
        g = 1.0 / np.sqrt(3.0)
        return np.array([-g, g]), np.array([1.0, 1.0])
    elif order == 3:
        g = np.sqrt(3.0 / 5.0)
        return (
            np.array([0.0, g, -g]),
            np.array([8.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0]),
        )
    elif order == 4:
        return (
            np.array([0.339981, -0.339981, 0.861136, -0.861136]),
            np.array([0.652145, 0.652145, 0.347855, 0.347855]),
        )
    elif order == 5:
        return (
            np.array([0.0, 0.538469, -0.538469, 0.90618, -0.90618]),
            np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927]),
        )
    else:
        raise ValueError(f"Gauss order {order} not supported (use 2-5)")


def subpb2_solver(
    dic_mesh: DICMesh,
    gauss_pt_order: int,
    beta: float,
    mu: float,
    U: NDArray[np.float64],
    F: NDArray[np.float64],
    udual: NDArray[np.float64],
    vdual: NDArray[np.float64],
    alpha: float,
    winstepsize: int,
) -> NDArray[np.float64]:
    """Solve ADMM subproblem 2 for globally compatible displacement.

    Assembles and solves the FEM system::

        A * Uhat = b

    where ``A = (beta+alpha)*DN'DN + mu*N'N``  (stiffness + mass)
    and   ``b = beta*DN'*F + mu*N'N*U + alpha*DN'DN*U``  (load).

    The dual variables ``udual`` and ``vdual`` are not directly used in this
    subproblem — they enter the ADMM loop only through the updated ``U`` and
    ``F`` from subproblem 1.

    Args:
        dic_mesh: DIC FE mesh with coordinates, elements, and BCs.
        gauss_pt_order: Gauss quadrature order (2, 3, 4, or 5).
        beta: ADMM penalty for deformation gradient compatibility.
        mu: ADMM penalty for displacement compatibility.
        U: Displacement vector (2*n_nodes,), interleaved [u0,v0,...].
        F: Deformation gradient vector (4*n_nodes,), interleaved.
        udual: Displacement dual variable (2*n_nodes,).
        vdual: Deformation gradient dual variable (4*n_nodes,).
        alpha: Smoothness regularization coefficient (typically 0.0).
        winstepsize: Mesh spacing in pixels (used for Neumann BCs).

    Returns:
        Uhat: Solved displacement vector (2*n_nodes,), interleaved.
    """
    coords = dic_mesh.coordinates_fem
    elems = dic_mesh.elements_fem
    n_nodes = coords.shape[0]
    n_ele = elems.shape[0]
    fem_size = 2 * n_nodes

    if n_ele == 0:
        return U.copy()

    dirichlet = dic_mesh.dirichlet
    neumann = dic_mesh.neumann

    # --- Dummy node for missing midpoints ---
    # elements_fem uses -1 for absent midside nodes.  Map them to a dummy
    # node (index n_nodes) whose DOFs index into zero-padded vectors.
    dummy_node = n_nodes
    big_n = fem_size + 2  # 2 extra DOFs for the dummy node

    # Pad U and F with zeros for dummy node DOFs
    U_ext = np.concatenate([U, np.zeros(2, dtype=np.float64)])
    F_ext = np.concatenate([F, np.zeros(4, dtype=np.float64)])

    # --- Element node coordinates and hanging-node delta flags ---
    ptx = np.zeros((n_ele, 8), dtype=np.float64)
    pty = np.zeros((n_ele, 8), dtype=np.float64)
    for k in range(8):
        valid = elems[:, k] >= 0
        ptx[valid, k] = coords[elems[valid, k], 0]
        pty[valid, k] = coords[elems[valid, k], 1]

    delta = (elems[:, 4:8] >= 0).astype(np.float64)  # (n_ele, 4)

    # --- Build DOF index arrays ---
    # all_index_u: (n_ele, 16) maps element-local DOFs to global DOFs
    # all_index_f: (n_ele, 16, 4) maps element-local DOFs to F components
    all_index_u = np.zeros((n_ele, 16), dtype=np.int64)
    all_index_f = np.zeros((n_ele, 16, 4), dtype=np.int64)

    for k in range(8):
        node_ids = elems[:, k].copy()
        node_ids[node_ids < 0] = dummy_node

        # Displacement DOFs (interleaved u, v)
        all_index_u[:, 2 * k] = 2 * node_ids
        all_index_u[:, 2 * k + 1] = 2 * node_ids + 1

        # Gradient DOFs matching DN_all row order:
        #   row 0: du/dx -> F11 at 4n,     row 1: du/dy -> F12 at 4n+2
        #   row 2: dv/dx -> F21 at 4n+1,   row 3: dv/dy -> F22 at 4n+3
        f_base = 4 * node_ids
        for col in (2 * k, 2 * k + 1):
            all_index_f[:, col, 0] = f_base          # F11
            all_index_f[:, col, 1] = f_base + 2      # F12
            all_index_f[:, col, 2] = f_base + 1      # F21
            all_index_f[:, col, 3] = f_base + 3      # F22

    # --- Gauss quadrature tensor-product grid ---
    gqpt, gqwt = _gauss_points_1d(gauss_pt_order)
    ksi_grid, eta_grid = np.meshgrid(gqpt, gqpt, indexing="ij")
    wk_grid, we_grid = np.meshgrid(gqwt, gqwt, indexing="ij")
    ksi_list = ksi_grid.ravel()
    eta_list = eta_grid.ravel()
    wk_list = wk_grid.ravel()
    we_list = we_grid.ravel()
    n_gp = len(ksi_list)

    # --- COO sparse assembly indices (same for all Gauss points) ---
    local_r, local_c = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    local_r = local_r.ravel()  # (256,)
    local_c = local_c.ravel()  # (256,)

    trip_I = all_index_u[:, local_r].ravel()  # (n_ele * 256,)
    trip_J = all_index_u[:, local_c].ravel()  # (n_ele * 256,)

    # --- Gather element-level vectors from global vectors ---
    # In subpb2, dual variables are zeroed: UMinusv = U, FMinusW = F
    u_minus_v_ele = U_ext[all_index_u]  # (n_ele, 16)
    u_ele = U_ext[all_index_u]          # (n_ele, 16)

    f_minus_w_ele = np.zeros((n_ele, 16, 4), dtype=np.float64)
    for c in range(4):
        f_minus_w_ele[:, :, c] = F_ext[all_index_f[:, :, c]]

    # --- Accumulate stiffness and load over Gauss points ---
    temp_a_all = np.zeros((n_ele, 256), dtype=np.float64)
    b_v_all = np.zeros((n_ele, 16), dtype=np.float64)

    for gp_idx in range(n_gp):
        ksi = float(ksi_list[gp_idx])
        eta = float(eta_list[gp_idx])
        wk = float(wk_list[gp_idx])
        we = float(we_list[gp_idx])

        N_all, DN_all, Jdet = compute_all_elements_gp(
            ksi, eta, ptx, pty, delta, n_ele,
        )
        weight = Jdet * (wk * we)  # (n_ele,)

        # Batch N^T*N and DN^T*DN via einsum: (n_ele, 16, 16)
        NtN = np.einsum("eai,eaj->eij", N_all, N_all)
        DtD = np.einsum("eai,eaj->eij", DN_all, DN_all)

        # Flatten to (n_ele, 256) and accumulate stiffness
        NtN_flat = NtN.reshape(n_ele, 256)
        DtD_flat = DtD.reshape(n_ele, 256)
        temp_a_all += weight[:, None] * (
            (beta + alpha) * DtD_flat + mu * NtN_flat
        )

        # --- Load vector terms ---
        # Term 1: beta * DN^T * (F - W) -> (n_ele, 16)
        term1 = np.einsum("eci,eic->ei", DN_all, f_minus_w_ele)

        # Term 2: mu * N^T*N * (U - v) -> (n_ele, 16)
        term2 = np.einsum("eij,ej->ei", NtN, u_minus_v_ele)

        # Term 3: alpha * DN^T*DN * U -> (n_ele, 16)
        term3 = np.einsum("eij,ej->ei", DtD, u_ele)

        be = weight[:, None] * (beta * term1 + mu * term2 + alpha * term3)
        b_v_all += be

    # --- Sparse stiffness matrix assembly ---
    trip_V = temp_a_all.ravel()
    A = sparse.coo_matrix(
        (trip_V, (trip_I, trip_J)), shape=(big_n, big_n),
    ).tocsr()

    # --- Load vector assembly (scatter-add) ---
    b = np.zeros(big_n, dtype=np.float64)
    np.add.at(b, all_index_u.ravel(), b_v_all.ravel())

    # --- Identify involved DOFs (exclude dummy node) ---
    involved_nodes = np.unique(elems[elems >= 0])
    involved_dofs = np.sort(np.concatenate([
        2 * involved_nodes, 2 * involved_nodes + 1,
    ]))

    # --- Neumann boundary conditions ---
    if neumann.shape[0] > 0:
        bc_force = -1.0 / winstepsize * F
        for j in range(neumann.shape[0]):
            n1 = int(neumann[j, 0])
            n2 = int(neumann[j, 1])
            nx_val = neumann[j, 2]
            ny_val = neumann[j, 3]
            edge_len = np.linalg.norm(coords[n1] - coords[n2])

            for nid in (n1, n2):
                # u DOF: F11*nx + F12*ny
                b[2 * nid] += 0.5 * edge_len * (
                    bc_force[4 * nid] * nx_val
                    + bc_force[4 * nid + 2] * ny_val
                )
                # v DOF: F21*nx + F22*ny
                b[2 * nid + 1] += 0.5 * edge_len * (
                    bc_force[4 * nid + 1] * nx_val
                    + bc_force[4 * nid + 3] * ny_val
                )

    # --- Dirichlet boundary conditions ---
    dirichlet_unique = (
        np.unique(dirichlet) if len(dirichlet) > 0
        else np.empty(0, dtype=np.int64)
    )

    if len(dirichlet_unique) > 0:
        dirichlet_dofs = np.sort(np.concatenate([
            2 * dirichlet_unique, 2 * dirichlet_unique + 1,
        ]))
    else:
        dirichlet_dofs = np.empty(0, dtype=np.int64)

    free_dofs = np.setdiff1d(involved_dofs, dirichlet_dofs)

    # Set known Dirichlet values and modify RHS
    Uhat = np.zeros(big_n, dtype=np.float64)
    if len(dirichlet_unique) > 0:
        for nid in dirichlet_unique:
            Uhat[2 * nid] = U[2 * nid]
            Uhat[2 * nid + 1] = U[2 * nid + 1]
        b -= A.dot(Uhat)

    # --- Solve the linear system ---
    n_free = len(free_dofs)
    if n_free == 0:
        return Uhat[:fem_size]

    A_free = A[np.ix_(free_dofs, free_dofs)]
    b_free = b[free_dofs]

    if n_free > 50000:
        # Large system: PCG with incomplete LU preconditioner
        try:
            ilu = spilu(A_free.tocsc(), drop_tol=1e-3)
            M_op = LinearOperator(A_free.shape, ilu.solve)
            x_sol, info = cg(A_free, b_free, tol=1e-6, maxiter=1000, M=M_op)
            if info != 0:
                warnings.warn(
                    f"PCG did not converge (info={info}), "
                    "falling back to direct solver.",
                    stacklevel=2,
                )
                x_sol = spsolve(A_free.tocsc(), b_free)
        except Exception:
            x_sol = spsolve(A_free.tocsc(), b_free)
    else:
        x_sol = spsolve(A_free.tocsc(), b_free)

    Uhat[free_dofs] = x_sol
    return Uhat[:fem_size]
