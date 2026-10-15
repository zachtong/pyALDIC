"""Tracking diagnostics for the cumulative displacement transform (layer 2).

The cumulative transform tracks every frame-0 node through the pair meshes
along its reference path. Two things can go wrong silently there:

- a tracked point lands outside the convex hull of the pair's measured
  nodes, and its displacement is copied from the nearest measured node
  (harmless when that node is close, wrong when it is far);
- a tracked point lands on masked-out pixels, and the crack-aware rules
  stop tracking it (NaN from that frame on).

:class:`TrackingCollector` counts both while the composition runs, per
composed frame, without changing a single value. It is fail-safe: its first
error is logged once and turns it off, and the run then reports ``None``
diagnostics instead of losing results.

:func:`assess_tracking` turns the counts into findings with English keys
(the thresholds below), :func:`log_tracking_summary` logs them, and
:func:`finish_tracking_diagnostics` does both at the end of ``run_aldic``.

Pure core module: English log text only (the GUI translates the findings),
inputs are never mutated.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._grid_extent import ReferenceGrids, grid_node_count
from .data_structures import (
    FrameTrackingStats,
    RefGridExtent,
    TrackingDiagnostics,
)

logger = logging.getLogger(__name__)

# An extrapolation is far when the nearest measured node is more than this
# many grid steps (winstepsize) away.
FAR_EXTRAPOLATION_STEPS = 1

# Assessment thresholds (plan D16; loss fractions are the user's decision).
# grid_miss: a frame has at least max(MIN_POINTS, ceil(FRACTION * queried))
# far extrapolations inside the measurable area.
GRID_MISS_MIN_POINTS = 3
GRID_MISS_FRACTION = 0.005
# material_loss: WARNING above LOSS_WARN_FRACTION of the material nodes lost
# by the last frame, or above LOSS_BURST_FRACTION lost between consecutive
# frames; INFO from LOSS_NOTICE_FRACTION.
LOSS_WARN_FRACTION = 0.50
LOSS_BURST_FRACTION = 0.20
LOSS_NOTICE_FRACTION = 0.01
# The material_loss rule that set the severity (TrackingFinding.rule):
# the burst rule only when the cumulative one did not fire.
LOSS_RULE_CUMULATIVE = "cumulative"
LOSS_RULE_BURST = "burst"

# Finding keys and severities (English; the GUI translates them).
FINDING_GRID_MISS = "grid_miss"
FINDING_MATERIAL_LOSS = "material_loss"
FINDING_LEFT_MEASURABLE_AREA = "left_measurable_area"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
# Advice keys: incremental runs need an ROI per reference frame,
# accumulative runs an ROI per frame that covers the specimen.
ADVICE_REFERENCE_ROI = "reference_frame_roi"
ADVICE_FRAME_ROI = "frame_roi"

# Taint bits kept per node along its path (uint8).
_TAINT_FAR = np.uint8(1)  # far extrapolation inside the measurable box
_TAINT_BORDER = np.uint8(2)  # far extrapolation outside it (image edge)

_Box = tuple[int, int, int, int]


def measurable_box_for(img_shape: tuple[int, int], winsize: int) -> _Box:
    """``(x_lo, x_hi, y_lo, y_hi)``, inclusive, where a whole subset fits.

    Matches the half-window clamp of the grid builders: node centers lie in
    ``[winsize // 2, dim - 1 - winsize // 2]`` on each axis.
    """
    h, w = img_shape
    hw = int(winsize) // 2
    return (hw, int(w) - 1 - hw, hw, int(h) - 1 - hw)


def far_extrapolation_px(winstepsize: float) -> float | None:
    """Far-extrapolation threshold in px (``None`` without a grid step)."""
    if winstepsize <= 0:
        return None
    return float(FAR_EXTRAPOLATION_STEPS * winstepsize)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Step:
    """Counts of the composition step in progress (one deformed frame)."""

    frame: int
    ref_frame: int
    queried: NDArray[np.bool_]  # per node: finite when the step began
    n_extrapolated: int
    n_far: int
    n_border: int
    max_px: float
    taint: NDArray[np.uint8]  # parent's taint | this step's bits
    parent_killed: NDArray[np.bool_]
    alive_after_add: NDArray[np.bool_] | None = None
    n_lost_no_data: int = 0
    n_lost_child: int = 0


class TrackingCollector:
    """Fail-safe counter fed by the cumulative composition.

    The composition calls, per step from ``ref_frame`` to ``frame``:
    :meth:`begin_step` with the interpolation's outside mask and nearest
    distances, :meth:`parent_kills`, :meth:`after_increment`,
    :meth:`child_kills`, then :meth:`end_step`; :meth:`result` at the end.

    Every public method catches its own exceptions: the first failure is
    logged once with ``logger.exception`` and turns the collector off, after
    which every call does nothing and :meth:`result` returns ``None``. The
    composition's values never depend on it, and no argument is modified.

    Args:
        n_nodes: Nodes of the frame-0 mesh (length of every per-node array).
        far_px: Extrapolations farther than this count as far; ``None``
            disables the far (and border) classification.
        measurable_box: Inclusive ``(x_lo, x_hi, y_lo, y_hi)``; a far query
            outside it counts as border. ``None`` disables that split.
    """

    def __init__(
        self,
        n_nodes: int,
        far_px: float | None = None,
        measurable_box: _Box | None = None,
    ) -> None:
        self._ok = True
        self._n_nodes = 0
        self._far_px: float | None = None
        self._box: _Box | None = None
        self._n_never_material = 0
        self._taint: dict[int, NDArray[np.uint8]] = {}
        self._frames: dict[int, FrameTrackingStats] = {}
        self._step: _Step | None = None
        self._guard(self._setup, n_nodes, far_px, measurable_box)

    # --- public, fail-safe -------------------------------------------------

    def born_dead(self, born_dead: NDArray[np.bool_]) -> None:
        """Frame-0 nodes on masked-out pixels (never material)."""
        self._guard(self._record_born_dead, born_dead)

    def begin_step(
        self,
        frame: int,
        ref_frame: int,
        coords: NDArray[np.float64],
        finite_q: NDArray[np.bool_],
        outside: NDArray[np.bool_] | None,
        nn_dist: NDArray[np.float64] | None,
    ) -> None:
        """Start the step; ``outside``/``nn_dist`` cover ``coords[finite_q]``
        (``None`` when no point was queried)."""
        self._guard(
            self._begin_step, frame, ref_frame, coords, finite_q, outside,
            nn_dist,
        )

    def parent_kills(self, kill_idx: NDArray[np.intp]) -> None:
        """Nodes stopped by the reference frame's mask this step."""
        self._guard(self._parent_kills, kill_idx)

    def after_increment(self, coords: NDArray[np.float64]) -> None:
        """Positions right after adding this step's increment."""
        self._guard(self._after_increment, coords)

    def child_kills(self, kill_idx: NDArray[np.intp]) -> None:
        """Nodes stopped by the deformed frame's mask this step."""
        self._guard(self._child_kills, kill_idx)

    def end_step(self, coords: NDArray[np.float64]) -> None:
        """Positions at the end of the step (the frame's tracked points)."""
        self._guard(self._end_step, coords)

    def result(self) -> TrackingDiagnostics | None:
        """The diagnostics, or ``None`` if counting failed."""
        return self._guard(self._build)

    # --- stages ------------------------------------------------------------

    def _guard(self, stage: Callable[..., Any], *args: Any) -> Any:
        if not self._ok:
            return None
        try:
            return stage(*args)
        except Exception:
            self._ok = False
            logger.exception(
                "Tracking diagnostics failed and were turned off; the "
                "displacements are not affected."
            )
            return None

    def _setup(self, n_nodes, far_px, measurable_box) -> None:
        self._n_nodes = int(n_nodes)
        self._far_px = None if far_px is None else float(far_px)
        if measurable_box is not None:
            x_lo, x_hi, y_lo, y_hi = (int(b) for b in measurable_box)
            self._box = (x_lo, x_hi, y_lo, y_hi)
        self._taint = {0: np.zeros(self._n_nodes, dtype=np.uint8)}

    def _record_born_dead(self, born_dead: NDArray[np.bool_]) -> None:
        self._n_never_material = int(np.count_nonzero(born_dead))

    def _begin_step(
        self, frame, ref_frame, coords, finite_q, outside, nn_dist,
    ) -> None:
        queried = np.array(finite_q, dtype=bool)
        bits = np.zeros(self._n_nodes, dtype=np.uint8)
        n_out = n_far = n_border = 0
        max_px = 0.0
        if outside is not None and queried.any():
            out = np.asarray(outside, dtype=bool)
            dist = np.asarray(nn_dist, dtype=np.float64)
            far = self._far(out, dist)
            border = far & ~self._in_box(np.asarray(coords)[queried])
            idx = np.flatnonzero(queried)
            bits[idx[far & ~border]] = _TAINT_FAR
            bits[idx[border]] = _TAINT_BORDER
            n_out = int(np.count_nonzero(out))
            n_far = int(np.count_nonzero(far))
            n_border = int(np.count_nonzero(border))
            max_px = float(dist[out].max(initial=0.0))
        self._step = _Step(
            frame=int(frame), ref_frame=int(ref_frame), queried=queried,
            n_extrapolated=n_out, n_far=n_far, n_border=n_border,
            max_px=max_px, taint=self._taint[int(ref_frame)] | bits,
            parent_killed=np.zeros(self._n_nodes, dtype=bool),
        )

    def _far(self, out, dist) -> NDArray[np.bool_]:
        if self._far_px is None:
            return np.zeros_like(out)
        return out & (dist > self._far_px)

    def _in_box(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        if self._box is None:
            return np.ones(len(points), dtype=bool)
        x_lo, x_hi, y_lo, y_hi = self._box
        x, y = points[:, 0], points[:, 1]
        return (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)

    def _parent_kills(self, kill_idx) -> None:
        killed = self._step.parent_killed.copy()
        killed[np.asarray(kill_idx, dtype=np.intp)] = True
        self._step = replace(self._step, parent_killed=killed)

    def _after_increment(self, coords) -> None:
        step = self._step
        alive = np.isfinite(np.asarray(coords)).all(axis=1)
        no_data = step.queried & ~step.parent_killed & ~alive
        self._step = replace(
            step, alive_after_add=alive,
            n_lost_no_data=int(np.count_nonzero(no_data)),
        )

    def _child_kills(self, kill_idx) -> None:
        step = self._step
        killed = np.unique(np.asarray(kill_idx, dtype=np.intp))
        n = int(np.count_nonzero(step.alive_after_add[killed]))
        self._step = replace(step, n_lost_child=step.n_lost_child + n)

    def _end_step(self, coords) -> None:
        step = self._step
        alive = np.isfinite(np.asarray(coords)).all(axis=1)
        self._frames[step.frame] = FrameTrackingStats(
            frame=step.frame,
            ref_frame=step.ref_frame,
            n_queried=int(np.count_nonzero(step.queried)),
            n_extrapolated=step.n_extrapolated,
            n_extrapolated_far=step.n_far,
            n_extrapolated_border=step.n_border,
            max_extrapolation_px=step.max_px,
            n_lost_parent_mask=int(np.count_nonzero(step.parent_killed)),
            n_lost_child_mask=step.n_lost_child,
            n_lost_no_data=step.n_lost_no_data,
            n_tracked=int(np.count_nonzero(alive)),
            n_tracked_extrapolated=int(
                np.count_nonzero(alive & (step.taint != 0))
            ),
        )
        self._taint[step.frame] = step.taint
        self._step = None

    def _build(self) -> TrackingDiagnostics:
        frames = tuple(self._frames[k] for k in sorted(self._frames))
        return TrackingDiagnostics(
            n_nodes=self._n_nodes,
            n_never_material=self._n_never_material,
            far_extrapolation_px=self._far_px,
            measurable_box=self._box,
            has_later_references=any(s.ref_frame != 0 for s in frames),
            frames=frames,
        )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingFinding:
    """One assessed problem, keyed in English for the GUI to translate.

    Attributes:
        key: ``FINDING_*``.
        severity: ``SEVERITY_*``.
        frame: Image index of the frame reported (the worst one; for a
            loss over the run, the last frame).
        ref_frame: Image index of that frame's reference frame.
        count: Nodes concerned on that frame. grid_miss: far
            extrapolations inside the measurable area; material_loss:
            material nodes no longer tracked by that frame;
            left_measurable_area: far extrapolations outside it.
        fraction: ``count`` over that frame's queried points (grid_miss,
            left_measurable_area) or over the material nodes
            (material_loss).
        max_px: That frame's ``max_extrapolation_px``: the largest distance
            over all its out-of-hull points, not only the ``count`` ones
            (0.0 for material_loss).
        n_frames: Frames that show it (material_loss: frames that lost
            material since the previous frame).
        rule: material_loss: ``LOSS_RULE_BURST`` when only the burst rule
            raised it to WARNING (``frame`` is then the burst frame), else
            ``LOSS_RULE_CUMULATIVE``. Empty for the other keys.
        burst_count: material_loss: material nodes lost on ``frame`` alone,
            against the previous composed frame in time (all the material
            for the first one). 0 for the other keys.
    """

    key: str
    severity: str
    frame: int
    ref_frame: int
    count: int
    fraction: float
    max_px: float
    n_frames: int
    rule: str = ""
    burst_count: int = 0


def assess_tracking(diag: TrackingDiagnostics) -> tuple[TrackingFinding, ...]:
    """Findings for ``diag``, WARNING ones first (rule order otherwise).

    The raw out-of-hull count never raises a finding: healthy runs copy
    hundreds of values per step from nodes a pixel away.
    """
    frames = sorted(diag.frames, key=lambda s: s.frame)
    found = [
        f for f in (
            _grid_miss(frames),
            _material_loss(frames, diag.n_material),
            _left_measurable_area(frames),
        )
        if f is not None
    ]
    return tuple(sorted(found, key=lambda f: f.severity != SEVERITY_WARNING))


def tracking_advice(diag: TrackingDiagnostics) -> str:
    """``ADVICE_REFERENCE_ROI`` for incremental tracking, else
    ``ADVICE_FRAME_ROI``."""
    if diag.has_later_references:
        return ADVICE_REFERENCE_ROI
    return ADVICE_FRAME_ROI


def _finding(
    key: str, severity: str, s: FrameTrackingStats, count: int, denom: int,
    max_px: float, n_frames: int,
) -> TrackingFinding:
    return TrackingFinding(
        key=key, severity=severity, frame=s.frame, ref_frame=s.ref_frame,
        count=int(count), fraction=count / denom if denom > 0 else 0.0,
        max_px=float(max_px), n_frames=int(n_frames),
    )


def _in_region_far(s: FrameTrackingStats) -> int:
    return s.n_extrapolated_far - s.n_extrapolated_border


def _grid_miss_threshold(n_queried: int) -> int:
    return max(GRID_MISS_MIN_POINTS, math.ceil(GRID_MISS_FRACTION * n_queried))


def _grid_miss(frames: Sequence[FrameTrackingStats]) -> TrackingFinding | None:
    hits = [
        s for s in frames
        if _in_region_far(s) >= _grid_miss_threshold(s.n_queried)
    ]
    if not hits:
        return None
    worst = max(hits, key=_in_region_far)
    return _finding(
        FINDING_GRID_MISS, SEVERITY_WARNING, worst, _in_region_far(worst),
        worst.n_queried, worst.max_extrapolation_px, len(hits),
    )


def _material_loss(
    frames: Sequence[FrameTrackingStats], n_material: int,
) -> TrackingFinding | None:
    if n_material <= 0 or not frames:
        return None
    # New losses against the previous frame in time (frame 0 tracks all
    # material), so an accumulative run's per-frame totals are not bursts.
    previous = [n_material] + [s.n_tracked for s in frames[:-1]]
    bursts = [p - s.n_tracked for p, s in zip(previous, frames)]
    last = len(frames) - 1
    worst = max(range(len(frames)), key=bursts.__getitem__)  # first max
    lost = (n_material - frames[last].n_tracked) / n_material
    if lost > LOSS_WARN_FRACTION:
        severity, k, rule = SEVERITY_WARNING, last, LOSS_RULE_CUMULATIVE
    elif bursts[worst] / n_material > LOSS_BURST_FRACTION:
        severity, k, rule = SEVERITY_WARNING, worst, LOSS_RULE_BURST
    elif lost >= LOSS_NOTICE_FRACTION:
        severity, k, rule = SEVERITY_INFO, last, LOSS_RULE_CUMULATIVE
    else:
        return None
    s = frames[k]
    finding = _finding(
        FINDING_MATERIAL_LOSS, severity, s, n_material - s.n_tracked,
        n_material, 0.0, sum(1 for b in bursts if b > 0),
    )
    return replace(finding, rule=rule, burst_count=int(bursts[k]))


def _left_measurable_area(
    frames: Sequence[FrameTrackingStats],
) -> TrackingFinding | None:
    hits = [s for s in frames if s.n_extrapolated_border > 0]
    if not hits:
        return None
    worst = max(hits, key=lambda s: s.n_extrapolated_border)
    return _finding(
        FINDING_LEFT_MEASURABLE_AREA, SEVERITY_INFO, worst,
        worst.n_extrapolated_border, worst.n_queried,
        worst.max_extrapolation_px, len(hits),
    )


# ---------------------------------------------------------------------------
# Logging (English: developer stream; the GUI shows translated messages)
# ---------------------------------------------------------------------------

_ADVICE_TEXT = {
    ADVICE_REFERENCE_ROI: (
        "Give every reference frame its own Region of Interest that covers "
        "the specimen in that frame."
    ),
    ADVICE_FRAME_ROI: (
        "Check that each frame's Region of Interest covers the specimen in "
        "that frame."
    ),
}


_LOSS_CAUSE = (
    "because they landed on pixels outside the Region of Interest or their "
    "pair had no measured displacement there; their displacement is NaN "
    "from then on."
)


def log_tracking_summary(
    diag: TrackingDiagnostics, findings: Sequence[TrackingFinding],
) -> None:
    """One INFO summary, plus one WARNING per WARNING finding."""
    _log_summary(diag, findings)
    advice = _ADVICE_TEXT[tracking_advice(diag)]
    for f in findings:
        if f.severity == SEVERITY_WARNING:
            logger.warning("Tracking: %s %s", _warning_text(diag, f), advice)


def _warning_text(diag: TrackingDiagnostics, f: TrackingFinding) -> str:
    """English text of a WARNING finding, without the advice."""
    if f.key == FINDING_GRID_MISS:
        return (
            f"on image index {f.frame}, {f.count} node(s) inside the "
            f"measurable area were more than "
            f"{diag.far_extrapolation_px or 0.0:.1f} px from the nearest "
            f"node measured on reference image index {f.ref_frame}; their "
            f"displacement was copied from that node and may be wrong "
            f"({f.n_frames} frame(s) affected; the farthest out-of-hull "
            f"point on that frame was {f.max_px:.1f} px away)."
        )
    n_material = diag.n_material
    if f.rule == LOSS_RULE_BURST:
        burst_pct = 100.0 * f.burst_count / n_material if n_material else 0.0
        return (
            f"{f.burst_count} material node(s) ({burst_pct:.1f}%) stopped "
            f"being tracked between the previous frame and image index "
            f"{f.frame} ({f.count} of {n_material} "
            f"({100.0 * f.fraction:.1f}%) lost by then) {_LOSS_CAUSE}"
        )
    return (
        f"{f.count} of {n_material} material node(s) "
        f"({100.0 * f.fraction:.1f}%) stopped being tracked by image index "
        f"{f.frame} {_LOSS_CAUSE}"
    )


def _log_summary(
    diag: TrackingDiagnostics, findings: Sequence[TrackingFinding],
) -> None:
    frames = diag.frames
    if not frames:
        logger.info("Tracking summary: no frame was composed.")
        return
    last = frames[-1]
    n_material = diag.n_material
    lost = n_material - last.n_tracked
    notices = [f.key for f in findings if f.severity == SEVERITY_INFO]
    logger.info(
        "Tracking summary: %d frame(s) composed; %d of %d material node(s) "
        "tracked to image index %d (%.1f%% lost); %d out-of-hull "
        "extrapolation(s), max %.1f px, %d farther than %s (%d beyond the "
        "measurable area); notices: %s.",
        len(frames), last.n_tracked, n_material, last.frame,
        100.0 * lost / n_material if n_material > 0 else 0.0,
        sum(s.n_extrapolated for s in frames),
        max((s.max_extrapolation_px for s in frames), default=0.0),
        sum(s.n_extrapolated_far for s in frames),
        "an unset distance" if diag.far_extrapolation_px is None
        else f"{diag.far_extrapolation_px:g} px",
        sum(s.n_extrapolated_border for s in frames),
        ", ".join(notices) or "none",
    )


# ---------------------------------------------------------------------------
# End of run_aldic
# ---------------------------------------------------------------------------


def finish_tracking_diagnostics(
    diag: TrackingDiagnostics | None,
    grids: ReferenceGrids,
    img_shape: tuple[int, int],
    winsize: int,
    winstepsize: int,
) -> TrackingDiagnostics | None:
    """Attach the reference grid extents, then assess and log ``diag``.

    Only the grids of references that some composed frame used are kept:
    a partial run plans grids for references it never reached.

    Fail-safe: any error is logged once (``logger.exception``) and yields
    ``None``, never an exception, so a finished run is never lost to the
    diagnostics. ``None`` in gives ``None`` out.
    """
    if diag is None:
        return None
    try:
        used = {s.ref_frame for s in diag.frames}
        extents = tuple(
            RefGridExtent(
                ref_frame=ref, roi=roi,
                n_nodes=grid_node_count(roi, img_shape, winsize, winstepsize),
            )
            for ref, roi in grids.by_ref
            if ref in used
        )
        out = replace(diag, grid_extents=extents)
        log_tracking_summary(out, assess_tracking(out))
        return out
    except Exception:
        logger.exception(
            "Tracking diagnostics could not be assessed and were dropped; "
            "the displacements are not affected."
        )
        return None
