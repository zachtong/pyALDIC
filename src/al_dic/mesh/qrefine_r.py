"""Red refinement of quadrilateral elements with 1-irregular rule.

Port of MATLAB mesh/qrefine_r.m (Funken & Schmidt, 2020).

Marked Q4 elements are refined by bisecting all four edges (red refinement),
producing 4 child elements per parent. The 1-irregular rule is enforced
to prevent more than one hanging node per edge.

All indices are 0-based.

References:
    S Funken, A Schmidt. Adaptive mesh refinement in 2D: an efficient
    implementation in MATLAB. Comp. Meth. Appl. Math. 20:459-479, 2020.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .geometric_data import provide_geometric_data


def qrefine_r(
    coordinates: NDArray[np.float64],
    elements: NDArray[np.int64],
    irregular: NDArray[np.int64],
    marked_elements: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Perform red refinement on marked quadrilateral elements.

    Each marked element is split into 4 children by inserting midpoints on
    all edges and at the element center. The 1-irregular rule is enforced.

    Args:
        coordinates: (n_nodes, 2) node coordinates.
        elements: (n_elements, 4) Q4 element connectivity, 0-based.
        irregular: (n_irregular, 3) hanging-node constraints, 0-based.
            Each row [a, b, c]: node c is the midpoint of edge (a, b).
        marked_elements: 0-based indices of elements to refine.

    Returns:
        (new_coordinates, new_elements, new_irregular), all 0-based.
    """
    if len(marked_elements) == 0:
        return coordinates.copy(), elements.copy(), irregular.copy()

    n_elem = elements.shape[0]
    n_irr = irregular.shape[0] if irregular.ndim == 2 and irregular.shape[0] > 0 else 0

    # --- Step 1: Get edge topology ---
    result = provide_geometric_data(irregular, elements)
    edge2nodes = result[0]     # (n_edges, 2)
    irr2edges = result[1]      # (n_irr, 3) edge indices for each irregular row
    elem2edges = result[2]     # (n_elem, 4) edge indices for each element

    n_edges = edge2nodes.shape[0]

    # --- Step 2: Mark edges for refinement ---
    # Convention: 0 = unmarked, 1 = marked for new node, -1 = existing irregular
    edge_mark = np.zeros(n_edges, dtype=np.int64)

    # Mark all edges of marked elements
    edge_mark[elem2edges[marked_elements].ravel()] = 1

    # Mark parent edges of existing irregular nodes
    if n_irr > 0:
        edge_mark[irr2edges[:, 0]] = 1

    # --- Step 3: 1-irregular rule closure loop ---
    kdx = np.array([0])  # non-empty to enter loop
    swap = np.array([0])
    while len(kdx) > 0 or len(swap) > 0:
        # Per-element edge marks
        marked_per_elem = edge_mark[elem2edges]  # (n_elem, 4)
        abs_marks = np.abs(marked_per_elem)
        sum_abs = abs_marks.sum(axis=1)
        min_marks = marked_per_elem.min(axis=1)

        # Elements needing additional marking:
        # - Has 2-3 marked edges (incomplete red) OR has any -1 edge
        # - But NOT already fully marked (sum = 4)
        kdx = np.where(
            (sum_abs < 4) & ((sum_abs > 2) | (min_marks < 0))
        )[0]

        if len(kdx) > 0:
            # Find unmarked edges of these elements
            sub = marked_per_elem[kdx]  # (len(kdx), 4)
            rows, cols = np.where(sub == 0)
            edge_mark[elem2edges[kdx[rows], cols]] = 1

        # Handle irregular edges: if sub-edges are marked, flag parent
        if n_irr > 0:
            irr_marks = edge_mark[irr2edges]  # (n_irr, 3)
            has_sub_marked = np.any(irr_marks[:, 1:] != 0, axis=1)
            parent_edges = irr2edges[has_sub_marked, 0]
            swap = np.where(edge_mark[parent_edges] != -1)[0]
            edge_mark[parent_edges[swap]] = -1
        else:
            swap = np.array([], dtype=np.int64)

    # --- Step 4: Generate new nodes on marked edges ---
    if n_irr > 0:
        edge_mark[irr2edges[:, 0]] = -1  # Ensure irregular parent edges are -1

    new_edge_mask = edge_mark > 0
    n_new_nodes = new_edge_mask.sum()

    # edge_node_map: -1 = no node, 0+ = 0-based new node index
    edge_node_map = np.full(n_edges, -1, dtype=np.int64)
    n_old = len(coordinates)
    edge_node_map[new_edge_mask] = n_old + np.arange(n_new_nodes)

    # Compute midpoint coordinates
    new_coords = (
        coordinates[edge2nodes[new_edge_mask, 0]]
        + coordinates[edge2nodes[new_edge_mask, 1]]
    ) / 2.0
    coords = np.vstack([coordinates, new_coords])

    # --- Step 5: Restore existing irregular nodes ---
    if n_irr > 0:
        edge_node_map[irr2edges[:, 0]] = irregular[:, 2]

    # --- Step 6: Build new elements ---
    # new_nodes[i, j] = node index of midpoint on edge j of element i (-1 = none)
    new_nodes = edge_node_map[elem2edges]  # (n_elem, 4)

    # Classify elements by number of refined edges
    has_node = (new_nodes >= 0).astype(np.int64)
    reftyp = has_node @ (2 ** np.arange(4))  # binary: edge0=1, edge1=2, edge2=4, edge3=8
    is_red = reftyp == 15  # all 4 edges have midpoints
    is_none = reftyp < 15  # not fully refined → keep as-is

    # Compute output element count
    n_none = is_none.sum()
    n_red = is_red.sum()
    n_new_elem = n_none + 4 * n_red

    # Generate center nodes for red elements
    red_idx = np.where(is_red)[0]
    mid_nodes = np.full(n_elem, -1, dtype=np.int64)
    n_center = len(red_idx)
    mid_nodes[red_idx] = len(coords) + np.arange(n_center)
    center_coords = (
        coords[elements[red_idx, 0]]
        + coords[elements[red_idx, 1]]
        + coords[elements[red_idx, 2]]
        + coords[elements[red_idx, 3]]
    ) / 4.0
    coords = np.vstack([coords, center_coords]) if n_center > 0 else coords

    # Build output element array
    new_elements = np.zeros((n_new_elem, 4), dtype=np.int64)

    # Elements that are not refined: keep as-is
    elem_ptr = np.zeros(n_elem, dtype=np.int64)
    elem_ptr[is_none] = 1
    elem_ptr[is_red] = 4
    elem_start = np.concatenate([[0], np.cumsum(elem_ptr)])

    # Copy unchanged elements
    none_idx = np.where(is_none)[0]
    for i in none_idx:
        new_elements[elem_start[i]] = elements[i]

    # Red refinement: each parent → 4 children
    # Parent: nodes [0, 1, 2, 3], edge midpoints [e0, e1, e2, e3], center [c]
    #   Child 0: [node0, e0, c, e3]
    #   Child 1: [node1, e1, c, e0]
    #   Child 2: [node2, e2, c, e1]
    #   Child 3: [node3, e3, c, e2]
    for i in red_idx:
        s = elem_start[i]
        c = mid_nodes[i]
        e = new_nodes[i]  # [e0, e1, e2, e3]
        n = elements[i]   # [n0, n1, n2, n3]
        new_elements[s + 0] = [n[0], e[0], c, e[3]]
        new_elements[s + 1] = [n[1], e[1], c, e[0]]
        new_elements[s + 2] = [n[2], e[2], c, e[1]]
        new_elements[s + 3] = [n[3], e[3], c, e[2]]

    # --- Step 7: Generate new irregular data ---
    # Partially refined elements (reftyp > 0 and < 15) generate irregular entries
    partial_idx = np.where((reftyp > 0) & (reftyp < 15))[0]
    new_irregular_list = []

    for i in partial_idx:
        for j in range(4):
            if new_nodes[i, j] >= 0:
                edge_idx = elem2edges[i, j]
                a, b = edge2nodes[edge_idx]
                new_irregular_list.append([a, b, new_nodes[i, j]])

    # Also handle sub-edges of existing irregular nodes that got refined
    if n_irr > 0:
        sub_nodes = edge_node_map[irr2edges[:, 1:]]  # (n_irr, 2)
        for i in range(n_irr):
            for j in range(2):
                if sub_nodes[i, j] >= 0:
                    edge_idx = irr2edges[i, j + 1]
                    a, b = edge2nodes[edge_idx]
                    new_irregular_list.append([a, b, sub_nodes[i, j]])

    if new_irregular_list:
        new_irregular = np.array(new_irregular_list, dtype=np.int64)
    else:
        new_irregular = np.empty((0, 3), dtype=np.int64)

    return coords, new_elements, new_irregular
