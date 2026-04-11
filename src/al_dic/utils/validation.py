"""Assertions and validation helpers for DIC data.

Used throughout the pipeline to catch data integrity issues early.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def assert_displacement_vector(U: NDArray[np.float64], n_nodes: int, name: str = "U") -> None:
    """Assert that U is a valid interleaved displacement vector.

    Args:
        U: Displacement vector, expected shape (2*n_nodes,).
        n_nodes: Number of FEM nodes.
        name: Variable name for error messages.

    Raises:
        ValueError: If shape or content is invalid.
    """
    expected_len = 2 * n_nodes
    if U.shape != (expected_len,):
        raise ValueError(
            f"{name} shape {U.shape} != expected ({expected_len},) "
            f"for {n_nodes} nodes."
        )
    if np.all(np.isnan(U)):
        raise ValueError(f"{name} is entirely NaN.")


def assert_def_grad_vector(F: NDArray[np.float64], n_nodes: int, name: str = "F") -> None:
    """Assert that F is a valid interleaved deformation gradient vector.

    Args:
        F: Deformation gradient, expected shape (4*n_nodes,).
        n_nodes: Number of FEM nodes.
        name: Variable name for error messages.

    Raises:
        ValueError: If shape or content is invalid.
    """
    expected_len = 4 * n_nodes
    if F.shape != (expected_len,):
        raise ValueError(
            f"{name} shape {F.shape} != expected ({expected_len},) "
            f"for {n_nodes} nodes."
        )


def assert_mesh_consistent(
    coordinates_fem: NDArray[np.float64],
    elements_fem: NDArray[np.int64],
) -> None:
    """Assert basic mesh consistency.

    Checks that element indices are within bounds and that coordinates
    have the expected shape.

    Raises:
        ValueError: If mesh is inconsistent.
    """
    if coordinates_fem.ndim != 2 or coordinates_fem.shape[1] != 2:
        raise ValueError(
            f"coordinates_fem shape {coordinates_fem.shape} must be (N, 2)."
        )
    if elements_fem.ndim != 2 or elements_fem.shape[1] != 8:
        raise ValueError(
            f"elements_fem shape {elements_fem.shape} must be (M, 8)."
        )
    n_nodes = coordinates_fem.shape[0]
    if elements_fem.max() >= n_nodes:
        raise ValueError(
            f"Element index {elements_fem.max()} >= n_nodes {n_nodes} "
            f"(indices must be 0-based)."
        )
    if elements_fem.min() < 0:
        raise ValueError("Element indices contain negative values.")
