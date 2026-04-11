"""Edge-node topology extraction from Q4/Q8 elements.

Port of MATLAB mesh/provide_geometric_data.m (Funken & Schmidt, 2020).

Builds the edge-to-node mapping and element-to-edge mapping required by
qrefine_r and the FEM assembly.

All indices are 0-based.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def provide_geometric_data(
    irregular: NDArray[np.int64],
    elements4: NDArray[np.int64],
    *boundaries: NDArray[np.int64],
) -> tuple:
    """Extract edge-node topology and edge mappings.

    Port of MATLAB provide_geometric_data.m.

    This function:
    1. Collects all edges from Q4 elements and irregular (hanging) node data.
    2. Assigns a unique index to each edge (sorted node pair).
    3. Returns mappings from elements/irregular/boundaries to edges.

    Args:
        irregular: (n_irr, 3) hanging-node constraints. Cols 0-1 are edge
            endpoints, col 2 is the hanging node. Can be empty (0, 3).
        elements4: (n_elem, 4) Q4 element connectivity (corner nodes), 0-based.
        *boundaries: Optional boundary arrays, each (n_bc, 2), each row is
            an edge defined by two node indices.

    Returns:
        Tuple of:
            - edge2nodes: (n_edges, 2) unique edges, each row sorted.
            - irregular2edges: (n_irr, 3) mapping irregular to edge indices.
              Col 0 = edge index for the irregular edge (cols 0-1 of irregular).
              Cols 1-2 = edge indices for the two sub-edges (not always present).
            - element2edges: (n_elem, 4) edge indices per element.
            - boundary2edges[0], boundary2edges[1], ... : edge indices for
              each boundary array passed.
    """
    n_irr = irregular.shape[0] if irregular.ndim == 2 else 0

    # Collect all edges
    # From Q4 elements: edges are (0-1, 1-2, 2-3, 3-0) for each element
    n_elem = elements4.shape[0]
    elem_edges = np.column_stack([
        elements4[:, [0, 1, 2, 3]].ravel(),
        elements4[:, [1, 2, 3, 0]].ravel(),
    ]).reshape(-1, 2)  # (4*n_elem, 2)

    # From irregular: columns 0-1 define edges
    all_edges_list = [elem_edges]
    ptr = [4 * n_elem]

    if n_irr > 0:
        irr_edges = irregular[:, :2]
        all_edges_list.append(irr_edges)
        ptr.append(n_irr)
    else:
        ptr.append(0)

    for bnd in boundaries:
        if bnd is not None and len(bnd) > 0:
            all_edges_list.append(bnd[:, :2])
            ptr.append(len(bnd))
        else:
            ptr.append(0)

    all_edges = np.vstack(all_edges_list)

    # Sort each edge so that (min, max)
    sorted_edges = np.sort(all_edges, axis=1)

    # Find unique edges and their indices
    edge2nodes, inverse = np.unique(sorted_edges, axis=0, return_inverse=True)

    # Split inverse back into element edges, irregular edges, boundary edges
    cumptr = np.cumsum([0] + ptr)

    element2edges = inverse[cumptr[0]:cumptr[1]].reshape(n_elem, 4)

    if n_irr > 0:
        irr_edge_indices = inverse[cumptr[1]:cumptr[2]]
        # For irregular2edges, we need the main edge + sub-edges
        # The MATLAB version builds a more complex structure; for simplicity
        # we store just the main edge index for now, with sub-edge handling
        # done during refinement.
        # irregular2edges: (n_irr, 3) but we only fill col 0 reliably
        irregular2edges = np.zeros((n_irr, 3), dtype=np.int64)
        irregular2edges[:, 0] = irr_edge_indices

        # Sub-edges: for each irregular node (a, b, c), the sub-edges are
        # (a, c) and (c, b). Find their edge indices.
        for i in range(n_irr):
            a, b, c = irregular[i, 0], irregular[i, 1], irregular[i, 2]
            sub1 = tuple(sorted([a, c]))
            sub2 = tuple(sorted([c, b]))
            # Search in edge2nodes
            idx1 = _find_edge(edge2nodes, sub1)
            idx2 = _find_edge(edge2nodes, sub2)
            if idx1 >= 0:
                irregular2edges[i, 1] = idx1
            if idx2 >= 0:
                irregular2edges[i, 2] = idx2
    else:
        irregular2edges = np.empty((0, 3), dtype=np.int64)

    # Boundary edge mappings
    boundary2edges = []
    for j in range(len(boundaries)):
        start = cumptr[2 + j]
        end = cumptr[3 + j] if j + 3 < len(cumptr) else len(inverse)
        boundary2edges.append(inverse[start:end])

    return (edge2nodes, irregular2edges, element2edges, *boundary2edges)


def _find_edge(edge2nodes: NDArray[np.int64], edge: tuple[int, int]) -> int:
    """Find the index of an edge in edge2nodes using binary search.

    Args:
        edge2nodes: (n_edges, 2) sorted unique edges.
        edge: Tuple (min_node, max_node) to find.

    Returns:
        Edge index, or -1 if not found.
    """
    target = np.array(edge, dtype=np.int64)
    # Binary search on first column, then check second
    idx = np.searchsorted(edge2nodes[:, 0], target[0])
    while idx < len(edge2nodes) and edge2nodes[idx, 0] == target[0]:
        if edge2nodes[idx, 1] == target[1]:
            return int(idx)
        idx += 1
    return -1
