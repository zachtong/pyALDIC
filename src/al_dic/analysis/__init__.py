"""Probe-based post-processing analysis.

Core layer: no Qt, no ``tr()``. The GUI in ``al_dic.gui.panels.analysis_tab``
is one consumer; a batch script using the ``run_aldic`` API is another.
"""

from al_dic.analysis.engine import (
    AnalysisEngine,
    Kymograph,
    Profile,
    SampleStatus,
)
from al_dic.analysis.extract import extract_series, field_values, frame_count
from al_dic.analysis.probes import (
    AreaGeom,
    LineGeom,
    PointGeom,
    Probe,
    ProbeSet,
    allowed_reductions,
)
from al_dic.analysis.series import FrameStatus, TimeSeries

# The changelog promised ``from al_dic.analysis import extract_series``; it
# was never exported, so the documented import raised ImportError.
__all__ = [
    "AnalysisEngine",
    "AreaGeom",
    "FrameStatus",
    "Kymograph",
    "LineGeom",
    "PointGeom",
    "Probe",
    "ProbeSet",
    "Profile",
    "SampleStatus",
    "TimeSeries",
    "allowed_reductions",
    "extract_series",
    "field_values",
    "frame_count",
]
