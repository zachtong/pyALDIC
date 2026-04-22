"""AL-DIC: Augmented Lagrangian DIC with adaptive quadtree mesh."""

__version__ = "0.4.1"

# Public API — stable imports for external consumers
from .core.config import dicpara_default, validate_dicpara
from .core.data_structures import (
    DICMesh,
    DICPara,
    FrameResult,
    FrameSchedule,
    PipelineResult,
    StrainResult,
)
from .core.pipeline import run_aldic

__all__ = [
    # Pipeline entry point
    "run_aldic",
    # Configuration
    "dicpara_default",
    "validate_dicpara",
    # Data structures
    "DICPara",
    "DICMesh",
    "FrameSchedule",
    "FrameResult",
    "StrainResult",
    "PipelineResult",
]
