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
from scipy.sparse.linalg import LinearOperator, cg, spilu, splu, spsolve

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


# ---------------------------------------------------------------------------
# Mesh-geometry pre-computation (constant across ADMM iterations)
# ---------------------------------------------------------------------------

def _precompute_geometry(dic_mesh, gauss_pt_order):
    """Pre-compute mesh geometry, DOF indices, and Gauss-point shape functions.

    Exploits the 8×8 block-diagonal structure of the FEM stiffness matrix.
    Because u and v DOFs use identical shape functions, the 16×16 element
    stiffness decomposes as ``kron(K8, I_2)`` where K8 is the 8×8 node-level
    stiffness.  This reduces memory and computation by ~4×.

    Everything returned here depends only on the mesh and quadrature order,
    not on U, F, or ADMM parameters.
    """
    coords = dic_mesh.coordinates_fem
    elems = dic_mesh.elements_fem
    n_nodes = coords.shape[0]
    n_ele = elems.shape[0]
    fem_size = 2 * n_nodes
    dummy_node = n_nodes
    big_n = fem_size + 2

    # Element node coordinates and hanging-node delta flags
    ptx = np.zeros((n_ele, 8), dtype=np.float64)
    pty = np.zeros((n_ele, 8), dtype=np.float64)
    for k in range(8):
        valid = elems[:, k] >= 0
        ptx[valid, k] = coords[elems[valid, k], 0]
        pty[valid, k] = coords[elems[valid, k], 1]
    delta = (elems[:, 4:8] >= 0).astype(np.float64)

    # Node index per element (with dummy for missing midpoints)
    node_index = np.zeros((n_ele, 8), dtype=np.int64)
    for k in range(8):
        nids = elems[:, k].copy()
        nids[nids < 0] = dummy_node
        node_index[:, k] = nids

    # COO sparse assembly indices for 8×8 node-level stiffness
    local_r8, local_c8 = np.meshgrid(
        np.arange(8), np.arange(8), indexing="ij",
    )
    trip_I8 = node_index[:, local_r8.ravel()].ravel()
    trip_J8 = node_index[:, local_c8.ravel()].ravel()

    # Gauss quadrature points
    gqpt, gqwt = _gauss_points_1d(gauss_pt_order)
    ksi_grid, eta_grid = np.meshgrid(gqpt, gqpt, indexing="ij")
    wk_grid, we_grid = np.meshgrid(gqwt, gqwt, indexing="ij")

    # Pre-compute 8-component shape functions at each Gauss point
    gp_data = []
    for gp_idx in range(len(ksi_grid.ravel())):
        ksi = float(ksi_grid.ravel()[gp_idx])
        eta = float(eta_grid.ravel()[gp_idx])
        wk = float(wk_grid.ravel()[gp_idx])
        we = float(we_grid.ravel()[gp_idx])

        N_all, DN_all, Jdet = compute_all_elements_gp(
            ksi, eta, ptx, pty, delta, n_ele,
        )
        weight = Jdet * (wk * we)

        # Extract 8-component values from interleaved 16-column arrays
        Nvals = N_all[:, 0, 0::2]     # (n_ele, 8) shape function values
        dNdx = DN_all[:, 0, 0::2]     # (n_ele, 8) dN/dx
        dNdy = DN_all[:, 1, 0::2]     # (n_ele, 8) dN/dy

        # 8×8 node-level matrices (outer products)
        # NN8[i,j] = Ni * Nj  (mass-like term)
        # DD8[i,j] = dNi/dx * dNj/dx + dNi/dy * dNj/dy  (stiffness-like)
        NN8 = Nvals[:, :, None] * Nvals[:, None, :]    # (n_ele, 8, 8)
        DD8 = (dNdx[:, :, None] * dNdx[:, None, :]
               + dNdy[:, :, None] * dNdy[:, None, :])  # (n_ele, 8, 8)

        gp_data.append({
            "dNdx": dNdx, "dNdy": dNdy,
            "NN8": NN8, "DD8": DD8, "weight": weight,
        })

    # Boundary DOFs
    dirichlet = dic_mesh.dirichlet
    involved_nodes = np.unique(elems[elems >= 0])
    involved_dofs = np.sort(np.concatenate([
        2 * involved_nodes, 2 * involved_nodes + 1,
    ]))
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

    return {
        "n_ele": n_ele, "n_nodes": n_nodes, "fem_size": fem_size,
        "big_n": big_n, "dummy_node": dummy_node,
        "node_index": node_index,
        "trip_I8": trip_I8, "trip_J8": trip_J8,
        "gp_data": gp_data,
        "free_dofs": free_dofs, "dirichlet_unique": dirichlet_unique,
        "coords": coords, "neumann": dic_mesh.neumann,
    }


def precompute_subpb2(
    dic_mesh: DICMesh,
    gauss_pt_order: int,
    beta: float,
    mu: float,
    alpha: float,
) -> dict:
    """Pre-compute and factorize the subpb2 stiffness matrix.

    Call once before the ADMM loop when beta, mu, alpha are fixed.
    Pass the result to ``subpb2_solver(..., precomputed=cache)`` to
    skip redundant assembly and factorization.

    Args:
        dic_mesh: FEM mesh.
        gauss_pt_order: Gauss quadrature order.
        beta, mu, alpha: ADMM parameters (must not change between calls).

    Returns:
        Cache dict for ``subpb2_solver``.
    """
    geo = _precompute_geometry(dic_mesh, gauss_pt_order)
    n_ele = geo["n_ele"]
    n_nodes = geo["n_nodes"]
    big_n = geo["big_n"]
    free_dofs = geo["free_dofs"]

    if n_ele == 0:
        geo["lu"] = None
        geo["A"] = None
        return geo

    # Accumulate 8×8 node-level element stiffness over Gauss points
    temp8_all = np.zeros((n_ele, 64), dtype=np.float64)
    for gp in geo["gp_data"]:
        NN8_flat = gp["NN8"].reshape(n_ele, 64)
        DD8_flat = gp["DD8"].reshape(n_ele, 64)
        temp8_all += gp["weight"][:, None] * (
            (beta + alpha) * DD8_flat + mu * NN8_flat
        )

    # Assemble node-level stiffness K8: (n_nodes+1, n_nodes+1)
    K8 = sparse.coo_matrix(
        (temp8_all.ravel(), (geo["trip_I8"], geo["trip_J8"])),
        shape=(n_nodes + 1, n_nodes + 1),
    ).tocsr()

    # Expand to interleaved DOF stiffness: A = kron(K8, I_2)
    # A[2i, 2j] = A[2i+1, 2j+1] = K8[i, j]  (block-diagonal)
    A = sparse.kron(K8, sparse.eye(2, format="csr"), format="csr")
    geo["A"] = A

    # Extract free-DOF submatrix and factorize
    n_free = len(free_dofs)
    if n_free == 0:
        geo["lu"] = None
        return geo

    A_free = A[np.ix_(free_dofs, free_dofs)]

    if n_free > 50000:
        # Large: cache ILU preconditioner for PCG
        try:
            ilu = spilu(A_free.tocsc(), drop_tol=1e-3)
            geo["ilu"] = ilu
            geo["A_free"] = A_free
            geo["lu"] = None
        except Exception:
            geo["ilu"] = None
            geo["A_free"] = A_free
            geo["lu"] = None
    else:
        # Direct: cache full LU factorization
        try:
            geo["lu"] = splu(A_free.tocsc())
        except Exception:
            geo["lu"] = None
            geo["A_free"] = A_free

    return geo


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
    precomputed: dict | None = None,
) -> NDArray[np.float64]:
    """Solve ADMM subproblem 2 for globally compatible displacement.

    Assembles and solves the FEM system::

        A * Uhat = b

    where ``A = (beta+alpha)*DN'DN + mu*N'N``  (stiffness + mass)
    and   ``b = beta*DN'*F + mu*N'N*U + alpha*DN'DN*U``  (load).

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
        precomputed: Cache from ``precompute_subpb2``.  If provided,
            skips stiffness assembly and uses cached factorization.

    Returns:
        Uhat: Solved displacement vector (2*n_nodes,), interleaved.
    """
    if precomputed is not None:
        return _solve_cached(precomputed, beta, mu, alpha, U, F, winstepsize)

    return _solve_full(dic_mesh, gauss_pt_order, beta, mu, U, F, alpha, winstepsize)


def _solve_cached(cache, beta, mu, alpha, U, F, winstepsize):
    """Solve subpb2 using pre-computed stiffness and factorization.

    Uses the 8×8 block-diagonal decomposition: u and v DOFs are assembled
    independently with the same 8×8 node-level matrices NN8 and DD8,
    reducing einsum and scatter costs by ~4×.
    """
    n_ele = cache["n_ele"]
    fem_size = cache["fem_size"]
    big_n = cache["big_n"]
    free_dofs = cache["free_dofs"]
    dirichlet_unique = cache["dirichlet_unique"]

    if n_ele == 0:
        return U.copy()

    # Pad U and F for dummy node
    U_ext = np.concatenate([U, np.zeros(2, dtype=np.float64)])
    F_ext = np.concatenate([F, np.zeros(4, dtype=np.float64)])

    node_index = cache["node_index"]  # (n_ele, 8)

    # Gather element-level vectors using node indices
    u_even = U_ext[2 * node_index]          # (n_ele, 8) — u components
    u_odd = U_ext[2 * node_index + 1]       # (n_ele, 8) — v components
    F11_ele = F_ext[4 * node_index]          # (n_ele, 8)
    F21_ele = F_ext[4 * node_index + 1]      # (n_ele, 8)
    F12_ele = F_ext[4 * node_index + 2]      # (n_ele, 8)
    F22_ele = F_ext[4 * node_index + 3]      # (n_ele, 8)

    # Assemble load vector using 8×8 block form
    b_even = np.zeros((n_ele, 8), dtype=np.float64)
    b_odd = np.zeros((n_ele, 8), dtype=np.float64)

    for gp in cache["gp_data"]:
        weight = gp["weight"]      # (n_ele,)
        dNdx = gp["dNdx"]          # (n_ele, 8)
        dNdy = gp["dNdy"]          # (n_ele, 8)
        NN8 = gp["NN8"]            # (n_ele, 8, 8)
        DD8 = gp["DD8"]            # (n_ele, 8, 8)

        # term1: beta * DN^T @ F  (decomposed by u/v DOFs)
        t1_u = dNdx * F11_ele + dNdy * F12_ele  # (n_ele, 8)
        t1_v = dNdx * F21_ele + dNdy * F22_ele  # (n_ele, 8)

        # term2: mu * N^T N @ u  (decomposed by u/v DOFs)
        t2_u = np.einsum("eij,ej->ei", NN8, u_even)  # (n_ele, 8)
        t2_v = np.einsum("eij,ej->ei", NN8, u_odd)   # (n_ele, 8)

        w = weight[:, None]
        if alpha != 0.0:
            # term3: alpha * DN^T DN @ u
            t3_u = np.einsum("eij,ej->ei", DD8, u_even)
            t3_v = np.einsum("eij,ej->ei", DD8, u_odd)
            b_even += w * (beta * t1_u + mu * t2_u + alpha * t3_u)
            b_odd += w * (beta * t1_v + mu * t2_v + alpha * t3_v)
        else:
            b_even += w * (beta * t1_u + mu * t2_u)
            b_odd += w * (beta * t1_v + mu * t2_v)

    # Scatter-add to global load vector (interleaved DOFs)
    b = np.zeros(big_n, dtype=np.float64)
    np.add.at(b, (2 * node_index).ravel(), b_even.ravel())
    np.add.at(b, (2 * node_index + 1).ravel(), b_odd.ravel())

    # Neumann BCs
    neumann = cache["neumann"]
    coords = cache["coords"]
    if neumann.shape[0] > 0:
        bc_force = -1.0 / winstepsize * F
        for j in range(neumann.shape[0]):
            n1 = int(neumann[j, 0])
            n2 = int(neumann[j, 1])
            nx_val = neumann[j, 2]
            ny_val = neumann[j, 3]
            edge_len = np.linalg.norm(coords[n1] - coords[n2])
            for nid in (n1, n2):
                b[2 * nid] += 0.5 * edge_len * (
                    bc_force[4 * nid] * nx_val
                    + bc_force[4 * nid + 2] * ny_val
                )
                b[2 * nid + 1] += 0.5 * edge_len * (
                    bc_force[4 * nid + 1] * nx_val
                    + bc_force[4 * nid + 3] * ny_val
                )

    # Dirichlet BCs
    # Initialize Uhat with input U so orphan nodes (not referenced by any
    # element) retain their displacement instead of being zeroed out.
    # This happens with quadtree meshes where hole-interior elements are
    # removed but their nodes remain in the coordinate array.
    Uhat = np.zeros(big_n, dtype=np.float64)
    Uhat[:fem_size] = U
    A = cache["A"]
    if len(dirichlet_unique) > 0:
        # Zero out free DOFs before Dirichlet correction so only
        # Dirichlet values contribute to the RHS adjustment.
        Uhat_bc = np.zeros(big_n, dtype=np.float64)
        for nid in dirichlet_unique:
            Uhat_bc[2 * nid] = U[2 * nid]
            Uhat_bc[2 * nid + 1] = U[2 * nid + 1]
        b -= A.dot(Uhat_bc)

    n_free = len(free_dofs)
    if n_free == 0:
        return Uhat[:fem_size]

    b_free = b[free_dofs]

    # Solve using cached factorization
    lu = cache.get("lu")
    if lu is not None:
        x_sol = lu.solve(b_free)
    else:
        # Fallback: PCG with cached ILU or direct solve
        ilu = cache.get("ilu")
        A_free = cache.get("A_free")
        if A_free is None:
            A_free = A[np.ix_(free_dofs, free_dofs)]

        if ilu is not None:
            M_op = LinearOperator(A_free.shape, ilu.solve)
            x_sol, info = cg(A_free, b_free, rtol=1e-6, maxiter=1000, M=M_op)
            if info != 0:
                x_sol = spsolve(A_free.tocsc(), b_free)
        else:
            x_sol = spsolve(A_free.tocsc(), b_free)

    Uhat[free_dofs] = x_sol
    return Uhat[:fem_size]


def _solve_full(dic_mesh, gauss_pt_order, beta, mu, U, F, alpha, winstepsize):
    """Original full assembly + solve (no caching)."""
    coords = dic_mesh.coordinates_fem
    elems = dic_mesh.elements_fem
    n_nodes = coords.shape[0]
    n_ele = elems.shape[0]
    fem_size = 2 * n_nodes

    if n_ele == 0:
        return U.copy()

    dirichlet = dic_mesh.dirichlet
    neumann = dic_mesh.neumann

    dummy_node = n_nodes
    big_n = fem_size + 2

    U_ext = np.concatenate([U, np.zeros(2, dtype=np.float64)])
    F_ext = np.concatenate([F, np.zeros(4, dtype=np.float64)])

    # Element node coordinates and hanging-node delta flags
    ptx = np.zeros((n_ele, 8), dtype=np.float64)
    pty = np.zeros((n_ele, 8), dtype=np.float64)
    for k in range(8):
        valid = elems[:, k] >= 0
        ptx[valid, k] = coords[elems[valid, k], 0]
        pty[valid, k] = coords[elems[valid, k], 1]
    delta = (elems[:, 4:8] >= 0).astype(np.float64)

    # DOF index arrays
    all_index_u = np.zeros((n_ele, 16), dtype=np.int64)
    all_index_f = np.zeros((n_ele, 16, 4), dtype=np.int64)
    for k in range(8):
        node_ids = elems[:, k].copy()
        node_ids[node_ids < 0] = dummy_node
        all_index_u[:, 2 * k] = 2 * node_ids
        all_index_u[:, 2 * k + 1] = 2 * node_ids + 1
        f_base = 4 * node_ids
        for col in (2 * k, 2 * k + 1):
            all_index_f[:, col, 0] = f_base
            all_index_f[:, col, 1] = f_base + 2
            all_index_f[:, col, 2] = f_base + 1
            all_index_f[:, col, 3] = f_base + 3

    # Gauss quadrature
    gqpt, gqwt = _gauss_points_1d(gauss_pt_order)
    ksi_grid, eta_grid = np.meshgrid(gqpt, gqpt, indexing="ij")
    wk_grid, we_grid = np.meshgrid(gqwt, gqwt, indexing="ij")
    ksi_list = ksi_grid.ravel()
    eta_list = eta_grid.ravel()
    wk_list = wk_grid.ravel()
    we_list = we_grid.ravel()
    n_gp = len(ksi_list)

    local_r, local_c = np.meshgrid(
        np.arange(16), np.arange(16), indexing="ij",
    )
    trip_I = all_index_u[:, local_r.ravel()].ravel()
    trip_J = all_index_u[:, local_c.ravel()].ravel()

    u_minus_v_ele = U_ext[all_index_u]
    u_ele = U_ext[all_index_u]
    f_minus_w_ele = np.zeros((n_ele, 16, 4), dtype=np.float64)
    for c in range(4):
        f_minus_w_ele[:, :, c] = F_ext[all_index_f[:, :, c]]

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
        weight = Jdet * (wk * we)
        NtN = np.einsum("eai,eaj->eij", N_all, N_all)
        DtD = np.einsum("eai,eaj->eij", DN_all, DN_all)

        NtN_flat = NtN.reshape(n_ele, 256)
        DtD_flat = DtD.reshape(n_ele, 256)
        temp_a_all += weight[:, None] * (
            (beta + alpha) * DtD_flat + mu * NtN_flat
        )

        term1 = np.einsum("eci,eic->ei", DN_all, f_minus_w_ele)
        term2 = np.einsum("eij,ej->ei", NtN, u_minus_v_ele)
        term3 = np.einsum("eij,ej->ei", DtD, u_ele)
        b_v_all += weight[:, None] * (beta * term1 + mu * term2 + alpha * term3)

    trip_V = temp_a_all.ravel()
    A = sparse.coo_matrix(
        (trip_V, (trip_I, trip_J)), shape=(big_n, big_n),
    ).tocsr()

    b = np.zeros(big_n, dtype=np.float64)
    np.add.at(b, all_index_u.ravel(), b_v_all.ravel())

    involved_nodes = np.unique(elems[elems >= 0])
    involved_dofs = np.sort(np.concatenate([
        2 * involved_nodes, 2 * involved_nodes + 1,
    ]))

    if neumann.shape[0] > 0:
        bc_force = -1.0 / winstepsize * F
        for j in range(neumann.shape[0]):
            n1 = int(neumann[j, 0])
            n2 = int(neumann[j, 1])
            nx_val = neumann[j, 2]
            ny_val = neumann[j, 3]
            edge_len = np.linalg.norm(coords[n1] - coords[n2])
            for nid in (n1, n2):
                b[2 * nid] += 0.5 * edge_len * (
                    bc_force[4 * nid] * nx_val
                    + bc_force[4 * nid + 2] * ny_val
                )
                b[2 * nid + 1] += 0.5 * edge_len * (
                    bc_force[4 * nid + 1] * nx_val
                    + bc_force[4 * nid + 3] * ny_val
                )

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

    # Initialize with input U so orphan nodes (not in any element) keep
    # their displacement.  See _solve_cached for full explanation.
    Uhat = np.zeros(big_n, dtype=np.float64)
    Uhat[:fem_size] = U
    if len(dirichlet_unique) > 0:
        Uhat_bc = np.zeros(big_n, dtype=np.float64)
        for nid in dirichlet_unique:
            Uhat_bc[2 * nid] = U[2 * nid]
            Uhat_bc[2 * nid + 1] = U[2 * nid + 1]
        b -= A.dot(Uhat_bc)

    n_free = len(free_dofs)
    if n_free == 0:
        return Uhat[:fem_size]

    A_free = A[np.ix_(free_dofs, free_dofs)]
    b_free = b[free_dofs]

    if n_free > 50000:
        try:
            ilu = spilu(A_free.tocsc(), drop_tol=1e-3)
            M_op = LinearOperator(A_free.shape, ilu.solve)
            x_sol, info = cg(A_free, b_free, rtol=1e-6, maxiter=1000, M=M_op)
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
