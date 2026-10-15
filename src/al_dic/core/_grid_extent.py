"""Plan a measurement grid for each reference frame in incremental mode.

The frame-0 grid range (``gridxy_roi_range``) is built from the frame-0
Region of Interest. In incremental mode a later reference frame K can hold
material outside that range (a translating or rotating specimen), and its
pairs would then be solved on a grid that misses part of the specimen.
``plan_reference_grids`` gives each such reference its own range that
tightly covers the material of mask K, plus ``pad_steps`` rings of nodes.

Every planned range keeps the frame-0 lattice phase: both effective edges
move by whole steps, so ``_centered_arange`` keeps its offset and every K
node lies on the frame-0 lattice. The three grid builders
(``integer_search``, the pyramid's ``_remap_to_standard_grid`` and
``build_grid_for_roi``) therefore agree on every planned range.

A reference that needs no other lattice maps to ``roi0`` itself (the same
object), so a run where nothing triggers stays byte-identical.

Pure module: no GUI text, no I/O beyond logging, inputs are never mutated.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..solver.integer_search import _centered_arange
from .data_structures import GridxyROIRange

logger = logging.getLogger(__name__)

# Rings of nodes added beyond the material bbox of a reference mask.
PAD_STEPS = 1
# Warn when a reference grid has more than this many times the frame-0 nodes
# AND at least GRID_GROWTH_WARN_MIN_NODES more of them: on small grids the
# ratio alone fires for a specimen that merely moved (49 -> 132 nodes).
GRID_GROWTH_WARN_RATIO = 2.0
GRID_GROWTH_WARN_MIN_NODES = 5000
# A pixel is material when mask > threshold (matches majority_masked).
MATERIAL_THRESHOLD = 0.5

# Reasons a reference mask is not used to plan a grid (English keys).
SKIP_EMPTY = "empty"
SKIP_SAME_AS_FRAME0 = "same_as_frame0"
SKIP_UNMASKED = "unmasked"  # no masked-out pixel: np.ones means "no mask"
SKIP_ALL_BORDERS = "touches_all_borders"
SKIP_NO_NODES = "no_measurable_nodes"
SKIP_REASONS = (
    SKIP_EMPTY, SKIP_SAME_AS_FRAME0, SKIP_UNMASKED, SKIP_ALL_BORDERS,
    SKIP_NO_NODES,
)

# Inclusive material bounding box: (x_lo, x_hi, y_lo, y_hi).
_BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ReferenceGrids:
    """Grid range for each reference frame of a run.

    Attributes:
        roi0: The frame-0 grid range (the clamped ``gridxy_roi_range``).
        by_ref: ``(ref, range)`` for every reference frame whose material
            needs a lattice other than frame 0's, sorted by ``ref``.
        skipped: ``(ref, reason)`` for every reference mask that was not
            used for planning, sorted by ``ref``; reasons are ``SKIP_*``.
    """

    roi0: GridxyROIRange
    by_ref: tuple[tuple[int, GridxyROIRange], ...] = ()
    skipped: tuple[tuple[int, str], ...] = ()

    def roi_for(self, ref: int) -> GridxyROIRange:
        """Range for pairs whose reference frame is ``ref``.

        Returns ``roi0`` itself (same object) unless ``ref`` has its own.
        """
        for k, roi in self.by_ref:
            if k == ref:
                return roi
        return self.roi0

    @property
    def is_identity(self) -> bool:
        """True when every reference frame uses the frame-0 grid."""
        return not self.by_ref


@dataclass(frozen=True)
class _Axis:
    """Frame-0 lattice along one image axis."""

    e_lo: int  # effective edges, after the half-window clamp
    e_hi: int
    n_lo: int  # outermost frame-0 nodes
    n_hi: int
    lim_lo: int  # admissible effective edges: [half_w, dim - 1 - half_w]
    lim_hi: int
    step: int


@dataclass(frozen=True)
class _Frame0Mask:
    """Material of mask 0, computed once per plan."""

    source: object
    material: NDArray[np.bool_]
    bbox: _BBox | None
    all_borders: bool


def plan_reference_grids(
    roi0: GridxyROIRange,
    masks: Sequence[NDArray[np.generic]],
    ref_frames: Iterable[int],
    img_shape: tuple[int, int],
    winsize: int,
    winstepsize: int,
    pad_steps: int = PAD_STEPS,
) -> ReferenceGrids:
    """Plan a phase-aligned grid range for every reference frame K != 0.

    K triggers when the material bbox of ``masks[K]`` (``mask > 0.5``)
    leaves the ``roi0`` range box (``roi0`` is already clamped to the
    image). K then gets the tightest frame-0-phase lattice covering the
    bbox with ``pad_steps`` rings, capped so every subset fits in the
    image. Masks that are empty, equal to mask 0, have no masked-out
    pixel, touch all four borders while mask 0 does too, or leave fewer
    than two nodes on an axis are skipped (``SKIP_*``). An all-ones mask
    is the API's "no mask", not material everywhere, so it never grids
    the whole image. ``ref_frames`` are the schedule's reference
    indices (0 is ignored); only their masks are read.

    Raises:
        ValueError: On invalid sizes, a reference without a mask, or a
            reference mask that is not 2-D.
    """
    _validate_scalars(img_shape, winsize, winstepsize, pad_steps)
    refs = sorted({int(r) for r in ref_frames} - {0})
    _validate_refs(refs, len(masks))
    h, w = img_shape
    half_w = winsize // 2
    ax_x = _frame0_axis(roi0.gridx, w, half_w, winstepsize)
    ax_y = _frame0_axis(roi0.gridy, h, half_w, winstepsize)
    if not refs or ax_x is None or ax_y is None:
        return ReferenceGrids(roi0=roi0)
    frame0 = _frame0_mask(masks[0], img_shape)
    by_ref: list[tuple[int, GridxyROIRange]] = []
    skipped: list[tuple[int, str]] = []
    for ref in refs:
        reason, roi = _plan_one(
            masks[ref], frame0, roi0, (ax_x, ax_y), img_shape, pad_steps,
        )
        if reason is not None:
            skipped.append((ref, reason))
        elif roi is not None:
            by_ref.append((ref, roi))
    return ReferenceGrids(roi0=roi0, by_ref=tuple(by_ref), skipped=tuple(skipped))


def grid_node_count(
    roi: GridxyROIRange,
    img_shape: tuple[int, int],
    winsize: int,
    winstepsize: int,
) -> int:
    """Number of nodes of the rectangular lattice that ``roi`` produces."""
    h, w = img_shape
    half_w = winsize // 2
    nx = _axis_nodes(roi.gridx, w, half_w, winstepsize).size
    ny = _axis_nodes(roi.gridy, h, half_w, winstepsize).size
    return int(nx * ny)


def log_grid_plan(
    grids: ReferenceGrids,
    img_shape: tuple[int, int],
    winsize: int,
    winstepsize: int,
) -> None:
    """Log the plan once (INFO) and warn about outsized reference grids.

    A reference grid is outsized when it has more than
    ``GRID_GROWTH_WARN_RATIO`` times the frame-0 nodes and at least
    ``GRID_GROWTH_WARN_MIN_NODES`` more of them.
    """
    n0 = grid_node_count(grids.roi0, img_shape, winsize, winstepsize)
    counts = [
        (ref, grid_node_count(roi, img_shape, winsize, winstepsize))
        for ref, roi in grids.by_ref
    ]
    skips = _format_skips(grids.skipped)
    if counts:
        ref_max, n_max = max(counts, key=lambda c: c[1])
        logger.info(
            "Reference grids: %d reference frame(s) get their own grid; "
            "frame-0 grid %d nodes; largest %d nodes on image index %d; "
            "skipped masks: %s",
            len(counts), n0, n_max, ref_max, skips,
        )
    else:
        logger.info(
            "Reference grids: every reference frame uses the frame-0 grid "
            "(%d nodes); skipped masks: %s",
            n0, skips,
        )
    large = [
        ref for ref, n in counts
        if n > GRID_GROWTH_WARN_RATIO * n0
        and n - n0 >= GRID_GROWTH_WARN_MIN_NODES
    ]
    if large:
        logger.warning(
            "Reference grids larger than %.1fx the frame-0 grid (%d nodes) "
            "and by at least %d nodes on image index(es) %s; stray Region "
            "of Interest components are the usual cause.",
            GRID_GROWTH_WARN_RATIO, n0, GRID_GROWTH_WARN_MIN_NODES, large,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_scalars(
    img_shape: tuple[int, int], winsize: int, winstepsize: int, pad_steps: int,
) -> None:
    if len(img_shape) != 2 or min(img_shape) < 1:
        raise ValueError(
            f"img_shape must be (height, width) with positive sizes "
            f"(got {img_shape})"
        )
    if winsize < 1:
        raise ValueError(f"winsize must be >= 1 (got {winsize})")
    if winstepsize < 1:
        raise ValueError(f"winstepsize must be >= 1 (got {winstepsize})")
    if pad_steps < 0:
        raise ValueError(f"pad_steps must be >= 0 (got {pad_steps})")


def _validate_refs(refs: list[int], n_masks: int) -> None:
    bad = [r for r in refs if r < 0 or r >= n_masks]
    if bad:
        raise ValueError(
            f"reference frame(s) {bad} have no mask ({n_masks} masks given)"
        )


def _axis_nodes(
    edges: tuple[int, int], dim: int, half_w: int, step: int,
) -> NDArray[np.float64]:
    """Node coordinates on one axis, clamped exactly like integer_search."""
    return _centered_arange(
        max(int(edges[0]), half_w), min(int(edges[1]), dim - 1 - half_w), step,
    )


def _frame0_axis(
    edges: tuple[int, int], dim: int, half_w: int, step: int,
) -> _Axis | None:
    """Frame-0 lattice on one axis, or None when it has no node."""
    nodes = _axis_nodes(edges, dim, half_w, step)
    if nodes.size == 0:
        return None
    return _Axis(
        e_lo=max(int(edges[0]), half_w),
        e_hi=min(int(edges[1]), dim - 1 - half_w),
        n_lo=int(nodes[0]),
        n_hi=int(nodes[-1]),
        lim_lo=half_w,
        lim_hi=dim - 1 - half_w,
        step=step,
    )


def _material(mask: NDArray[np.generic]) -> NDArray[np.bool_]:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"masks must be 2-D (got shape {arr.shape})")
    return arr > MATERIAL_THRESHOLD


def _bbox(material: NDArray[np.bool_]) -> _BBox | None:
    """Inclusive bbox of the material pixels via row/column projections."""
    cols = np.flatnonzero(np.any(material, axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(np.any(material, axis=1))
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def _touches_all_borders(bbox: _BBox | None, img_shape: tuple[int, int]) -> bool:
    h, w = img_shape
    return bbox == (0, w - 1, 0, h - 1)


def _frame0_mask(mask0: NDArray[np.generic], img_shape: tuple[int, int]) -> _Frame0Mask:
    material = _material(mask0)
    bbox = _bbox(material)
    return _Frame0Mask(
        source=mask0,
        material=material,
        bbox=bbox,
        all_borders=_touches_all_borders(bbox, img_shape),
    )


def _has_measurable_part(
    bbox: _BBox, img_shape: tuple[int, int], half_w: int,
) -> bool:
    """True when the bbox reaches the band where a node can be centered."""
    h, w = img_shape
    x_ok = max(bbox[0], half_w) <= min(bbox[1], w - 1 - half_w)
    y_ok = max(bbox[2], half_w) <= min(bbox[3], h - 1 - half_w)
    return x_ok and y_ok


def _screen_mask(
    mask: NDArray[np.generic],
    frame0: _Frame0Mask,
    img_shape: tuple[int, int],
    half_w: int,
) -> tuple[str | None, _BBox | None]:
    """Skip reason (D3 a, b, unmasked, c, d, in that order) or the bbox."""
    if mask is frame0.source:
        material, bbox = frame0.material, frame0.bbox
    else:
        material = _material(mask)
        bbox = _bbox(material)
    if bbox is None:
        return SKIP_EMPTY, None
    if material is frame0.material or np.array_equal(material, frame0.material):
        return SKIP_SAME_AS_FRAME0, None
    if material.all():
        return SKIP_UNMASKED, None
    if frame0.all_borders and _touches_all_borders(bbox, img_shape):
        return SKIP_ALL_BORDERS, None
    if not _has_measurable_part(bbox, img_shape, half_w):
        return SKIP_NO_NODES, None
    return None, bbox


def _inside_range(bbox: _BBox, roi: GridxyROIRange) -> bool:
    """Range-box trigger test (range edges, not the node hull)."""
    return (
        bbox[0] >= roi.gridx[0] and bbox[1] <= roi.gridx[1]
        and bbox[2] >= roi.gridy[0] and bbox[3] <= roi.gridy[1]
    )


def _tight_edges(axis: _Axis, b_lo: int, b_hi: int, pad: int) -> tuple[int, int]:
    """Effective edges of the tight phase-aligned cover of [b_lo, b_hi].

    Both edges move by whole steps (D5): ``a`` grows the low side when
    negative and shrinks it when positive; ``b`` mirrors it on the high
    side. The caps keep every subset inside the image.
    """
    s = axis.step
    a = (b_lo - axis.n_lo) // s - pad  # floor((b_lo - n_lo) / s) - pad
    b = -((axis.n_hi - b_hi) // s) + pad  # ceil((b_hi - n_hi) / s) + pad
    a = max(a, -((axis.e_lo - axis.lim_lo) // s))
    b = min(b, (axis.lim_hi - axis.e_hi) // s)
    return axis.e_lo + a * s, axis.e_hi + b * s


def _plan_one(
    mask: NDArray[np.generic],
    frame0: _Frame0Mask,
    roi0: GridxyROIRange,
    axes: tuple[_Axis, _Axis],
    img_shape: tuple[int, int],
    pad: int,
) -> tuple[str | None, GridxyROIRange | None]:
    """``(skip_reason, None)``, ``(None, range)`` or ``(None, None)``."""
    ax_x, ax_y = axes
    reason, bbox = _screen_mask(mask, frame0, img_shape, ax_x.lim_lo)
    if reason is not None or bbox is None:
        return reason, None
    if _inside_range(bbox, roi0):
        return None, None
    gx = _tight_edges(ax_x, bbox[0], bbox[1], pad)
    gy = _tight_edges(ax_y, bbox[2], bbox[3], pad)
    if gx[1] - gx[0] < ax_x.step or gy[1] - gy[0] < ax_y.step:
        return SKIP_NO_NODES, None  # fewer than two nodes on an axis
    if gx == (ax_x.e_lo, ax_x.e_hi) and gy == (ax_y.e_lo, ax_y.e_hi):
        return None, None  # capped back onto the frame-0 lattice
    return None, GridxyROIRange(gridx=gx, gridy=gy)


def _format_skips(skipped: tuple[tuple[int, str], ...]) -> str:
    if not skipped:
        return "none"
    counts = Counter(reason for _, reason in skipped)
    return ", ".join(f"{r}={counts[r]}" for r in SKIP_REASONS if counts[r])
