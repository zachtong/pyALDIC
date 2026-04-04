"""Shared utilities: interpolation, region analysis, validation."""

from .interpolation import FieldInterpolator, scatter_to_grid, scattered_interpolant
from .outlier_detection import detect_bad_points, fill_nan_idw
from .region_analysis import NodeRegionMap, precompute_node_regions
from .validation import (
    assert_def_grad_vector,
    assert_displacement_vector,
    assert_mesh_consistent,
)

__all__ = [
    "FieldInterpolator",
    "scatter_to_grid",
    "scattered_interpolant",
    "detect_bad_points",
    "fill_nan_idw",
    "NodeRegionMap",
    "precompute_node_regions",
    "assert_displacement_vector",
    "assert_def_grad_vector",
    "assert_mesh_consistent",
]
