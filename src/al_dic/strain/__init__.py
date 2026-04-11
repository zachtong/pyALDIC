"""Strain computation and smoothing."""

from .compute_strain import compute_strain
from .nodal_strain_fem import global_nodal_strain_fem
from .smooth_field import smooth_field_sparse

__all__ = [
    "compute_strain",
    "global_nodal_strain_fem",
    "smooth_field_sparse",
]
