"""A probe's curve, and the reason behind every frame that is not plain.

A ``TimeSeries`` carries its own gaps and the reason for each one. Charts and
CSV both read ``status``, so what a user sees on screen and what they get in a
file cannot disagree about which frames mean anything.

``status`` says *why* a frame is special; whether a value survived is a
separate fact (``values[i]`` finite or NaN). A line that a crack runs through
keeps a value from the material either side and is marked ``CRACK``; a region
whose reliable points collapse is ``BELOW_THRESHOLD`` and blank. The first
version tied the two together and blanked every probe a crack or a hole
touched -- which is most of the probes anyone places on a fracture specimen.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class FrameStatus(enum.Enum):
    """Why a frame is not plain OK. The value's presence is separate."""

    #: Everything the probe covers was measured.
    OK = "ok"
    #: Part of the probe was consumed by a crack by this frame. For a line or
    #: region the value comes from the material that remains; for a gauge it
    #: is the reading across the crack (elongation including the opening).
    CRACK = "crack"
    #: Fewer valid points than the threshold asks for; left blank.
    BELOW_THRESHOLD = "below_threshold"
    #: Only unreliable points: strain edge trim or NaN values.
    UNRELIABLE = "unreliable"
    #: A gauge endpoint is not on measurable material on this frame.
    ENDPOINT_LOST = "endpoint_lost"
    #: The field does not exist for this run -- e.g. strain not computed.
    NOT_COMPUTED = "not_computed"
    #: The probe covers no material at all.
    NO_DATA = "no_data"


#: Axis conventions a displacement can be read in. Only ``disp_v`` differs.
IMAGE_AXES = "image"   # y down: what the screen shows
WORLD_AXES = "world"   # y up: what the node export writes


@dataclass(frozen=True)
class TimeSeries:
    """One probe, one quantity, across the sequence.

    ``axes`` records the convention the values were computed in. It matters
    only for ``disp_v`` -- strain is stored in world axes whatever this says,
    and the other displacements and every gauge quantity are the same either
    way -- but a file that says which it holds cannot be misread.
    """

    frames: NDArray[np.int64]
    values: NDArray[np.float64]
    valid_fraction: NDArray[np.float64]
    status: list[FrameStatus]
    unit: str
    axes: str = IMAGE_AXES

    def first_frame(self, status: FrameStatus) -> int | None:
        """The first frame with *status*, if any -- e.g. crack arrival."""
        for i, st in enumerate(self.status):
            if st is status:
                return int(self.frames[i])
        return None

    def contiguous_runs(self) -> Iterator[tuple[int, int]]:
        """Half-open index ranges of consecutive finite values.

        Charts draw one polyline per run, so a gap stays a gap. The reference
        implementation set ``connectNulls`` and drew straight through missing
        data.
        """
        finite = np.isfinite(self.values)
        start: int | None = None
        for i, ok in enumerate(finite):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                yield (start, i)
                start = None
        if start is not None:
            yield (start, len(finite))


__all__ = ["IMAGE_AXES", "WORLD_AXES", "FrameStatus", "TimeSeries"]
