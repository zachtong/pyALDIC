"""Full quadtree mesh generation pipeline.

Port of MATLAB mesh/generate_mesh.m (Jin Yang, Caltech).

Orchestrates the adaptive quadtree refinement workflow:
    1. mark_edge   — identify elements straddling mask boundaries
    2. qrefine_r   — red-refine marked elements (with 1-irregular rule)
    3. Reorder nodes within each element to CCW order
    4. Inject hanging (irregular) nodes into Q8 midside columns
    5. mark_inside  — remove elements inside mask holes
    6. Identify boundary-adjacent nodes (markCoordHoleEdge)
    7. Interpolate displacement U0 from uniform to quadtree mesh

MATLAB/Python differences:
    - MATLAB uses 1-based indices everywhere; Python uses 0-based.
    - MATLAB ``bwconncomp`` → ``scipy.ndimage.label``.
    - MATLAB ``scatteredInterpolant('natural','nearest')`` →
      ``scipy.interpolate.LinearNDInterpolator`` with nearest-neighbor fallback.
    - MATLAB ``sub2ind(imgSize, x, y)`` → Python ravel_multi_index with (y, x).
    - Hanging node midside mapping: Q8 columns 4-7 (0-based) correspond to
      edges (n1,n2), (n2,n3), (n3,n0), (n0,n1) respectively.

References:
    [1] J Yang, K Bhattacharya. Fast adaptive mesh augmented Lagrangian
        Digital Image Correlation. Exp. Mech., 2021.
    [2] S Funken, A Schmidt. Adaptive mesh refinement in 2D: an efficient
        implementation in MATLAB. Comp. Meth. Appl. Math. 20:459-479, 2020.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter, label

from ..core.data_structures import DICMesh, DICPara, ImageGradients
from .mark_edge import mark_edge
from .mark_inside import mark_inside
from .qrefine_r import qrefine_r


def generate_mesh(
    mesh: DICMesh,
    para: DICPara,
    Df: ImageGradients,
    U0: NDArray[np.float64],
) -> tuple[DICMesh, NDArray[np.float64]]:
    """Generate an adaptive quadtree mesh from a uniform starting mesh.

    Iteratively refines elements that straddle mask edges until no further
    refinement is needed (or elements reach ``element_min_size``).  After
    refinement, nodes inside mask holes are identified and the displacement
    field ``U0`` is interpolated from the old uniform mesh onto the new
    quadtree mesh.

    Args:
        mesh: Initial uniform DICMesh (from ``mesh_setup``).
        para: DIC parameters.  Uses ``img_ref_mask``, ``winstepsize``,
            ``winsize``.
        Df: Reference image gradients (provides ``img_ref_mask``,
            ``img_size``).
        U0: Initial displacement vector on the uniform mesh, shape
            (2*n_nodes_uniform,), interleaved [u0, v0, u1, v1, ...].

    Returns:
        A tuple ``(mesh_qt, U0_qt)`` where:
            - ``mesh_qt``: Refined DICMesh with updated ``coordinates_fem``,
              ``elements_fem``, ``irregular``, ``mark_coord_hole_edge``,
              and ``coordinates_fem_world``.
            - ``U0_qt``: Displacement vector interpolated onto the quadtree
              mesh, shape (2*n_nodes_qt,).
    """
    mask = para.img_ref_mask
    if mask is None:
        raise ValueError("DICPara.img_ref_mask must be set for quadtree mesh generation")

    coords_qt = mesh.coordinates_fem.copy()
    elems_qt = mesh.elements_fem[:, :4].copy()  # Work with Q4 during refinement
    irregular = np.empty((0, 3), dtype=np.int64)

    # --- Step 1: Iterative quadtree refinement ---
    while True:
        marks = mark_edge(coords_qt, elems_qt, mask, mesh.element_min_size)
        marked_idx = np.where(marks)[0]
        coords_qt, elems_qt, irregular = qrefine_r(
            coords_qt, elems_qt, irregular, marked_idx
        )
        if len(marked_idx) == 0:
            break

    # --- Step 2: Reorder element nodes to CCW (BL, BR, TR, TL) ---
    _reorder_element_nodes_ccw(coords_qt, elems_qt)

    # --- Step 3: Build Q8 elements with hanging nodes ---
    elems_q8 = _inject_hanging_nodes(elems_qt, irregular)

    # --- Step 4: Remove elements inside holes ---
    _, outside_idx = mark_inside(coords_qt, elems_q8, mask)
    elems_q8 = elems_q8[outside_idx]

    # --- Step 5: Identify boundary-adjacent nodes ---
    mark_coord_hole_edge = _find_boundary_nodes(
        coords_qt, elems_q8, mesh.coordinates_fem, para.winstepsize
    )

    # --- Step 6: Interpolate U0 from uniform to quadtree mesh ---
    U0_qt = _interpolate_u0(
        mesh.coordinates_fem, coords_qt, U0, Df.img_ref_mask, Df.img_size
    )

    # --- Step 7: Assemble output mesh ---
    h, _w = Df.img_size
    coords_world = np.column_stack([
        coords_qt[:, 0],
        h + 1 - coords_qt[:, 1],
    ])

    mesh_qt = DICMesh(
        coordinates_fem=coords_qt,
        elements_fem=elems_q8,
        irregular=irregular,
        mark_coord_hole_edge=mark_coord_hole_edge,
        coordinates_fem_world=coords_world,
        x0=mesh.x0,
        y0=mesh.y0,
        element_min_size=mesh.element_min_size,
    )

    return mesh_qt, U0_qt


def _reorder_element_nodes_ccw(
    coords: NDArray[np.float64],
    elements: NDArray[np.int64],
) -> None:
    """Reorder Q4 element nodes to counter-clockwise: BL, BR, TR, TL.

    Uses weighted distance trick: d = (x - xmin)^2 + 2*(y - ymin)^2.
    Sorted order is [BL, BR, TL, TR] → pick [0, 1, 3, 2] for CCW.

    Modifies ``elements`` in-place.
    """
    n_elem = elements.shape[0]
    for i in range(n_elem):
        cx = coords[elements[i, :4], 0]
        cy = coords[elements[i, :4], 1]

        x_sorted = np.sort(cx)
        y_sorted = np.sort(cy)

        # Weighted distance: BL→0, BR→small, TL→medium, TR→large
        dist = (cx - x_sorted[0]) ** 2 + 2.0 * (cy - y_sorted[0]) ** 2
        order = np.argsort(dist)

        # argsort gives [BL, BR, TL, TR] → we want [BL, BR, TR, TL]
        elements[i, :4] = elements[i, order[[0, 1, 3, 2]]]


def _inject_hanging_nodes(
    elements_q4: NDArray[np.int64],
    irregular: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Inject hanging (irregular) nodes into Q8 midside columns.

    For each irregular node [a, b, c] (c = midpoint of edge a-b),
    find which element edge matches (a,b) or (b,a) and assign c to
    the corresponding midside column.

    Q8 layout (0-based columns):
        - Columns 0-3: corner nodes [n0, n1, n2, n3]
        - Column 4: midside of edge (n1, n2)
        - Column 5: midside of edge (n2, n3)
        - Column 6: midside of edge (n3, n0)
        - Column 7: midside of edge (n0, n1)

    Returns:
        (n_elements, 8) Q8 element connectivity array.
    """
    n_elem = elements_q4.shape[0]
    # Initialize midside columns to -1 (no midside node).
    # The solver checks `elems[:, 4:8] >= 0` to detect Q8 midside nodes,
    # so 0 would be misinterpreted as a valid midside node at node index 0.
    elements_q8 = np.full((n_elem, 8), -1, dtype=np.int64)
    elements_q8[:, :4] = elements_q4

    if irregular.shape[0] == 0:
        return elements_q8

    # Edge pairs for each midside column and their corresponding column index
    # edge_def[k] = (col_a, col_b, midside_col)
    # Edge (n0,n1) → col 7, Edge (n1,n2) → col 4, Edge (n2,n3) → col 5, Edge (n3,n0) → col 6
    edge_defs = [
        (0, 1, 7),  # edge n0→n1 → midside col 7
        (1, 2, 4),  # edge n1→n2 → midside col 4
        (2, 3, 5),  # edge n2→n3 → midside col 5
        (3, 0, 6),  # edge n3→n0 → midside col 6
    ]

    for irr_row in irregular:
        a, b, c = irr_row[0], irr_row[1], irr_row[2]

        for col_a, col_b, mid_col in edge_defs:
            # Check forward direction: (a, b) matches (elements[:, col_a], elements[:, col_b])
            match_fwd = (elements_q4[:, col_a] == a) & (elements_q4[:, col_b] == b)
            elements_q8[match_fwd, mid_col] = c

            # Check reverse direction: (a, b) matches (elements[:, col_b], elements[:, col_a])
            match_rev = (elements_q4[:, col_b] == a) & (elements_q4[:, col_a] == b)
            elements_q8[match_rev, mid_col] = c

    return elements_q8


def _find_boundary_nodes(
    coords_qt: NDArray[np.float64],
    elements_q8: NDArray[np.int64],
    coords_uniform: NDArray[np.float64],
    winstepsize: int,
) -> NDArray[np.int64]:
    """Find node indices near hole edges and mesh boundaries.

    Combines two criteria:
    1. Nodes in refined (smaller) elements
    2. Nodes near the mesh boundary (within 1.01 * winstepsize)

    Then expands the set by 2 neighbor-ring iterations.

    Returns:
        Sorted array of unique 0-based node indices.
    """
    n_elem = elements_q8.shape[0]
    if n_elem == 0:
        return np.empty(0, dtype=np.int64)

    corners = elements_q8[:, :4]

    # Element diagonal size = distance from node0 to node2
    dx = coords_qt[corners[:, 0], 0] - coords_qt[corners[:, 2], 0]
    dy = coords_qt[corners[:, 0], 1] - coords_qt[corners[:, 2], 1]
    elem_size = np.sqrt(dx ** 2 + dy ** 2)

    # Threshold: elements smaller than ~sqrt(2) * winstepsize are "refined"
    threshold = 0.99 * np.sqrt(2.0) * winstepsize
    refined_idx = np.where(elem_size < threshold)[0]

    # Boundary elements: nodes within 1.01 * winstepsize of mesh edge
    x_min_mesh = coords_uniform[:, 0].min()
    x_max_mesh = coords_uniform[:, 0].max()
    y_min_mesh = coords_uniform[:, 1].min()
    y_max_mesh = coords_uniform[:, 1].max()
    tol = 1.01 * winstepsize

    left = np.where(coords_qt[corners[:, 0], 0] < x_min_mesh + tol)[0]
    right = np.where(coords_qt[corners[:, 2], 0] > x_max_mesh - tol)[0]
    top = np.where(coords_qt[corners[:, 0], 1] < y_min_mesh + tol)[0]
    bottom = np.where(coords_qt[corners[:, 2], 1] > y_max_mesh - tol)[0]

    # Union all marked element indices
    marked_elems = np.unique(np.concatenate([
        refined_idx, left, right, top, bottom
    ]))

    # Collect unique node indices from marked elements
    mark_nodes = np.unique(elements_q8[marked_elems].ravel())
    # Remove -1 midside placeholders (they represent "no midside node")
    mark_nodes = mark_nodes[mark_nodes >= 0]

    # Expand by 2 neighbor rings
    for _ in range(2):
        # Find elements sharing any node with current marked nodes
        has_marked = np.zeros(n_elem, dtype=np.bool_)
        for ei in range(n_elem):
            if np.any(np.isin(elements_q8[ei], mark_nodes)):
                has_marked[ei] = True

        neighbor_elems = np.where(has_marked)[0]
        mark_nodes = np.unique(elements_q8[neighbor_elems].ravel())
        mark_nodes = mark_nodes[mark_nodes > 0]

    return np.sort(mark_nodes)


def _interpolate_u0(
    coords_old: NDArray[np.float64],
    coords_new: NDArray[np.float64],
    U0: NDArray[np.float64],
    mask: NDArray[np.float64],
    img_size: tuple[int, int],
) -> NDArray[np.float64]:
    """Interpolate displacement from uniform mesh to quadtree mesh.

    Uses connected-component region analysis to interpolate per-region,
    avoiding cross-hole artifacts. For each connected region in the
    (slightly dilated) mask, nodes are matched and displacement is
    interpolated using LinearNDInterpolator with nearest-neighbor fallback.

    Args:
        coords_old: (n_old, 2) uniform mesh node coordinates.
        coords_new: (n_new, 2) quadtree mesh node coordinates.
        U0: (2*n_old,) interleaved displacement on uniform mesh.
        mask: (H, W) binary mask.
        img_size: (height, width).

    Returns:
        (2*n_new,) interleaved displacement on quadtree mesh.
    """
    n_new = coords_new.shape[0]
    U0_qt = np.zeros(2 * n_new, dtype=np.float64)

    # Find non-NaN nodes in old displacement
    u_old = U0[0::2]
    not_nan = ~np.isnan(u_old)
    not_nan_idx = np.where(not_nan)[0]

    if len(not_nan_idx) == 0:
        return U0_qt

    # Dilate mask slightly and find connected components
    dilated = gaussian_filter(mask.astype(np.float64), sigma=1.0)
    dilated_binary = (dilated > 0.01).astype(np.int32)
    labeled, n_regions = label(dilated_binary)

    h, w = img_size

    # Convert coordinates to pixel indices for region lookup
    # Coordinates: col 0 = x, col 1 = y → pixel index = (y, x)
    def _coord_to_label(coord_array: NDArray[np.float64]) -> NDArray[np.int64]:
        """Map coordinates to region labels via the labeled image."""
        x_idx = np.clip(np.round(coord_array[:, 0]).astype(np.int64), 0, w - 1)
        y_idx = np.clip(np.round(coord_array[:, 1]).astype(np.int64), 0, h - 1)
        return labeled[y_idx, x_idx]

    new_labels = _coord_to_label(coords_new)
    old_labels = _coord_to_label(coords_old)
    old_not_nan_labels = old_labels[not_nan_idx]

    for region_id in range(1, n_regions + 1):
        # Quadtree nodes in this region
        dst_mask = new_labels == region_id
        dst_idx = np.where(dst_mask)[0]
        if len(dst_idx) == 0:
            continue

        # Uniform mesh source nodes in this region
        src_in_region = old_not_nan_labels == region_id
        src_idx = not_nan_idx[src_in_region]
        if len(src_idx) < 3:
            continue

        src_xy = coords_old[src_idx]
        dst_xy = coords_new[dst_idx]

        try:
            # Interpolate u-displacement
            u_vals = U0[2 * src_idx]
            interp_u = LinearNDInterpolator(src_xy, u_vals)
            u_new = interp_u(dst_xy)

            # Fill NaN from linear with nearest-neighbor
            nan_mask = np.isnan(u_new)
            if nan_mask.any():
                nn_u = NearestNDInterpolator(src_xy, u_vals)
                u_new[nan_mask] = nn_u(dst_xy[nan_mask])

            U0_qt[2 * dst_idx] = u_new

            # Interpolate v-displacement
            v_vals = U0[2 * src_idx + 1]
            interp_v = LinearNDInterpolator(src_xy, v_vals)
            v_new = interp_v(dst_xy)

            nan_mask = np.isnan(v_new)
            if nan_mask.any():
                nn_v = NearestNDInterpolator(src_xy, v_vals)
                v_new[nan_mask] = nn_v(dst_xy[nan_mask])

            U0_qt[2 * dst_idx + 1] = v_new

        except Exception as exc:
            warnings.warn(
                f"U0 interpolation failed for region {region_id}: {exc}",
                stacklevel=2,
            )

    return U0_qt
