"""Turn a probe and a finished run into a curve -- the scripting entry point.

    from al_dic.analysis import LineGeom, Probe, extract_series

    ts = extract_series(result, probe, field="strain_eyy", reduction="mean")

A thin front for ``AnalysisEngine``: each call builds one engine, so one
triangulation, however many frames it reads. Code reading several probes or
fields from the same run should hold an engine and call it directly.

Frame indices follow the rest of the application: 0 is the reference image,
and frame *n* reads ``result_disp[n - 1]``.

Physical units are parameters, never read from application state. Lengths --
displacement fields, elongation, crack opening -- are multiplied by
``pixel_size``; strains are ratios and never are.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from al_dic.analysis.engine import (
    DEFAULT_MIN_VALID_FRACTION,
    AnalysisEngine,
)
from al_dic.analysis.probes import Probe
from al_dic.analysis.series import TimeSeries
from al_dic.core.fields import is_strain_field, validate_field


def field_values(
    result,
    field: str,
    frame: int,
    *,
    pixel_size: float = 1.0,
) -> NDArray[np.float64] | None:
    """Per-node values of *field* on *frame*, NaN where unreliable.

    None when the field does not exist for this run (strain not computed, or
    a frame past the end).
    """
    validate_field(field)
    engine = AnalysisEngine(result)
    got = engine.node_field(field, frame)
    if got is None:
        return None
    vals, valid = got
    out = np.where(valid, vals, np.nan)
    return out if is_strain_field(field) else out * float(pixel_size)


def frame_count(result) -> int:
    """Number of frames a probe can report on, reference frame included."""
    return len(result.result_disp) + 1


def extract_series(
    result,
    probe: Probe,
    field: str | None,
    reduction: str,
    *,
    masks: Sequence[NDArray | None] | None = None,
    ref_mask: NDArray | None = None,
    pixel_size: float = 1.0,
    length_unit: str = "px",
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    frames: Sequence[int] | None = None,
    axes: str = "image",
) -> TimeSeries:
    """Read *probe* across the sequence and reduce each frame to one number.

    Parameters
    ----------
    reduction:
        A statistic (``value``, ``mean``, ``median``, ``max``, ``min``,
        ``std``, ``valid_fraction``) or, for a line probe, a gauge quantity
        (``strain``, ``true_strain``, ``elongation``, ``cod``,
        ``cod_sliding``, ``cod_magnitude``), for which *field* is ignored.
    ref_mask:
        Frame-0 region-of-interest mask; holes and notches present from the
        start come from here.
    masks:
        Optional extra barrier per entry of *frames*, ``< 0.5`` marking void.
        A real run needs none: material a crack consumes already has NaN
        ``U_accum``. This is for runs whose displacements were built by hand.
    axes:
        ``"image"`` (v positive down, as on screen) or ``"world"`` (v
        positive up, as in the node export). Only ``disp_v`` differs.
    """
    engine = AnalysisEngine(result, ref_mask=ref_mask)
    return engine.series(
        probe, field, reduction,
        pixel_size=pixel_size,
        length_unit=length_unit,
        min_valid_fraction=min_valid_fraction,
        frames=frames,
        masks=masks,
        axes=axes,
    )


__all__ = [
    "DEFAULT_MIN_VALID_FRACTION", "extract_series", "field_values",
    "frame_count",
]
