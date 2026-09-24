"""Read probes across a finished run: one triangulation, sampling as a gather.

The reference nodes never move -- ``U_accum`` is defined on them -- so the
Delaunay triangulation, each probe's sample locations and their barycentric
weights are computed once. Reading a frame is then an index gather over three
corner values per sample. The first implementation rebuilt the triangulation
for every frame of every curve (1.3 s per frame at 20 k nodes).

What makes a sample invalid, and why
------------------------------------
``OUTSIDE``
    Never material: off the mesh, on a pixel outside the frame-0 region of
    interest, or in a triangle that bridges a frame-0 void (a hole, a notch).
    Excluded from the valid fraction -- nothing was lost there.
``CONSUMED``
    Material at frame 0, taken by a crack since: a corner node has NaN
    ``U_accum``. The crack-aware cumulative transform writes exactly that for
    material an opening crack swallows, so the crack is found in frame-0
    coordinates at node resolution without a per-frame mask. (Recovering it
    from ``per_frame_rois`` needs a mesh re-cut and rasterisation per frame,
    which is why the first implementation never did it in the GUI.)
``UNRELIABLE``
    Present but not trustworthy: a NaN value, or a strain node that failed
    ``strain_valid``. The strain edge trim applies to strain fields only;
    displacement is not trimmed on screen and is not trimmed here.

A frame's reduced value uses the valid samples. Its ``FrameStatus`` says why
the frame is not plain OK, independently of whether a value survived: a line a
crack runs through keeps a value from the material either side and is marked
``CRACK``; it is not blanked. The previous whole-probe verdict blanked every
section drawn from a hole edge and every kymograph after the crack arrived.

Units
-----
Everything is computed in pixels. Only quantities that are lengths --
displacement fields, elongation, crack opening -- are multiplied by
``pixel_size`` on the way out. Gauge strain is a ratio of two pixel lengths and
never sees the pixel size; adding scaled displacements to pixel coordinates is
the bug that made a 1 % strain read 3.0 at 300 um/px.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from al_dic.analysis.probes import (
    GAUGE_QUANTITIES,
    SPATIAL_STATISTICS,
    AreaGeom,
    Geometry,
    LineGeom,
    PointGeom,
    Probe,
    allowed_reductions,
)
from al_dic.analysis.series import (
    IMAGE_AXES,
    WORLD_AXES,
    FrameStatus,
    TimeSeries,
)
from al_dic.core.fields import (
    field_unit,
    is_strain_field,
    validate_field,
)
from al_dic.utils.crack_barrier import cross_crack_simplices

#: Samples per mesh step along a line and across a region. Finer than the
#: node spacing invents resolution the data does not have.
SAMPLES_PER_STEP = 2.0
_MIN_LINE_SAMPLES = 2
_MAX_LINE_SAMPLES = 512
_MAX_AREA_SAMPLES = 40_000

_LENGTH_GAUGES = frozenset({"elongation", "cod", "cod_sliding", "cod_magnitude"})

DEFAULT_MIN_VALID_FRACTION = 0.5


class SampleStatus(enum.IntEnum):
    """Verdict on one sample on one frame. Stored as ``uint8`` arrays."""

    VALID = 0
    OUTSIDE = 1
    CONSUMED = 2
    UNRELIABLE = 3


# ---------------------------------------------------------------------------
# Plans and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplePlan:
    """Where a probe reads, fixed for the whole run.

    ``corners``/``weights`` are meaningful where ``inside`` is True.
    ``material`` is False for samples that never were measurable (see
    ``SampleStatus.OUTSIDE``). ``node_ids`` are extra exact-node samples a
    region adds for its maximum and minimum.
    """

    xy: NDArray[np.float64]
    corners: NDArray[np.int64]
    weights: NDArray[np.float64]
    inside: NDArray[np.bool_]
    material: NDArray[np.bool_]
    distance: NDArray[np.float64] | None = None
    node_ids: NDArray[np.int64] | None = None

    @property
    def size(self) -> int:
        return int(self.xy.shape[0])


@dataclass(frozen=True)
class Samples:
    """One frame of one plan: values in pixel units, NaN unless VALID."""

    values: NDArray[np.float64]
    status: NDArray[np.uint8]

    @property
    def valid(self) -> NDArray[np.bool_]:
        return self.status == SampleStatus.VALID

    @property
    def material(self) -> NDArray[np.bool_]:
        return self.status != SampleStatus.OUTSIDE

    @property
    def valid_fraction(self) -> float:
        n = int(self.material.sum())
        return float(self.valid.sum()) / n if n else 0.0

    @property
    def consumed(self) -> bool:
        return bool((self.status == SampleStatus.CONSUMED).any())


@dataclass(frozen=True)
class Profile:
    """A field along a line on one frame, against reference arc length."""

    distance: NDArray[np.float64]
    values: NDArray[np.float64]
    status: NDArray[np.uint8]
    unit: str
    distance_unit: str


@dataclass(frozen=True)
class Kymograph:
    """A field along a line on every frame: ``values[frame, sample]``."""

    frames: NDArray[np.int64]
    distance: NDArray[np.float64]
    values: NDArray[np.float64]
    status: NDArray[np.uint8]
    unit: str
    distance_unit: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """Samples one run. Build once per result; every read reuses it.

    Parameters
    ----------
    result:
        A ``PipelineResult``.
    ref_mask:
        The frame-0 region-of-interest mask, ``< 0.5`` marking void. Holes and
        notches present at frame 0 come from here. Optional.
    """

    def __init__(self, result, *, ref_mask: NDArray | None = None) -> None:
        self._result = result
        nodes = np.asarray(result.dic_mesh.coordinates_fem, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"Mesh nodes must be (N, 2), got {nodes.shape}.")
        self._nodes = nodes
        self._n = nodes.shape[0]
        finite = np.isfinite(nodes).all(axis=1)
        if int(finite.sum()) < 3:
            raise ValueError(
                "Analysis needs at least three mesh nodes with finite "
                f"coordinates (got {int(finite.sum())})."
            )
        # Triangulate the finite nodes; keep corners as global node ids.
        self._tri_ids = np.flatnonzero(finite)
        self._tri = Delaunay(nodes[finite])
        self._simplices = self._tri_ids[self._tri.simplices]

        step = float(getattr(result.dic_para, "winstepsize", 0) or 0)
        self._step = step if step > 0 else _median_spacing(nodes[finite])

        self._mask = None if ref_mask is None else np.asarray(ref_mask)
        if self._mask is None:
            self._void = np.zeros(len(self._simplices), dtype=bool)
        else:
            self._void = cross_crack_simplices(
                self._tri.simplices, nodes[finite], self._mask
            )

        self._alive: dict[int, NDArray[np.bool_]] = {}
        self._plans: dict[Geometry, SamplePlan] = {}

    # -- run facts ----------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Frames a probe can report on, the reference frame included."""
        return len(self._result.result_disp) + 1

    @property
    def step(self) -> float:
        return self._step

    def has_strain(self) -> bool:
        return bool(self._result.result_strain)

    # -- plans ----------------------------------------------------------------

    def plan(self, geometry: Geometry) -> SamplePlan:
        cached = self._plans.get(geometry)
        if cached is None:
            cached = self._build_plan(geometry)
            self._plans[geometry] = cached
        return cached

    def _build_plan(self, geometry: Geometry) -> SamplePlan:
        if isinstance(geometry, PointGeom):
            xy = np.array([[geometry.x, geometry.y]], dtype=np.float64)
            return self._plan_at(xy)
        if isinstance(geometry, LineGeom):
            n = self.line_sample_count(geometry)
            t = np.linspace(0.0, 1.0, n)
            xy = np.column_stack([
                geometry.x0 + t * (geometry.x1 - geometry.x0),
                geometry.y0 + t * (geometry.y1 - geometry.y0),
            ])
            plan = self._plan_at(xy)
            return _with(plan, distance=t * geometry.length())
        if isinstance(geometry, AreaGeom):
            xy = _area_grid(geometry, self._step / SAMPLES_PER_STEP)
            plan = self._plan_at(xy)
            inside_nodes = np.flatnonzero(_in_area(geometry, self._nodes))
            return _with(plan, node_ids=inside_nodes)
        raise TypeError(f"Unsupported geometry {type(geometry).__name__}.")

    def line_sample_count(self, geometry: LineGeom) -> int:
        n = int(round(geometry.length() / self._step * SAMPLES_PER_STEP)) + 1
        return int(np.clip(n, _MIN_LINE_SAMPLES, _MAX_LINE_SAMPLES))

    def _plan_at(self, xy: NDArray[np.float64]) -> SamplePlan:
        simplex = self._tri.find_simplex(xy)
        inside = simplex >= 0
        corners = np.zeros((len(xy), 3), dtype=np.int64)
        weights = np.zeros((len(xy), 3), dtype=np.float64)
        if inside.any():
            s = simplex[inside]
            transform = self._tri.transform[s]
            delta = xy[inside] - transform[:, 2]
            b = np.einsum("ijk,ik->ij", transform[:, :2], delta)
            weights[inside] = np.column_stack([b, 1.0 - b.sum(axis=1)])
            corners[inside] = self._simplices[s]

        material = inside.copy()
        if inside.any():
            material[inside] &= ~self._void[simplex[inside]]
        if self._mask is not None:
            h, w = self._mask.shape[:2]
            xi = np.clip(np.round(xy[:, 0]).astype(np.int64), 0, w - 1)
            yi = np.clip(np.round(xy[:, 1]).astype(np.int64), 0, h - 1)
            material &= self._mask[yi, xi] >= 0.5
        # A corner that is not material at frame 0 (born dead) makes the
        # triangle unmeasurable from the start.
        alive0 = self.alive(0)
        material &= np.where(inside, alive0[corners].all(axis=1), False)
        return SamplePlan(
            xy=xy, corners=corners, weights=weights,
            inside=inside, material=material,
        )

    # -- per-frame node state ---------------------------------------------

    def alive(self, frame: int) -> NDArray[np.bool_]:
        """Nodes that are material on *frame*.

        Frame 0: everything except nodes the transform marks dead on frame 1
        *and* that sit on a frame-0 void pixel -- the born-dead nodes of an
        initial notch. A node that is material at frame 0 and dies on frame 1
        is alive here and consumed there.
        """
        cached = self._alive.get(frame)
        if cached is not None:
            return cached
        if frame == 0:
            out = np.ones(self._n, dtype=bool)
            if self._result.result_disp:
                u1 = self._disp(1)
                if u1 is not None:
                    dead1 = ~np.isfinite(u1[0])
                    out = ~(dead1 & ~self._node_on_material())
        else:
            uv = self._disp(frame)
            out = (np.zeros(self._n, dtype=bool) if uv is None
                   else np.isfinite(uv[0]) & np.isfinite(uv[1]))
        self._alive[frame] = out
        return out

    def _node_on_material(self) -> NDArray[np.bool_]:
        if self._mask is None:
            return np.ones(self._n, dtype=bool)
        h, w = self._mask.shape[:2]
        xi = np.clip(np.round(self._nodes[:, 0]).astype(np.int64), 0, w - 1)
        yi = np.clip(np.round(self._nodes[:, 1]).astype(np.int64), 0, h - 1)
        return self._mask[yi, xi] >= 0.5

    def _disp(self, frame: int) -> tuple[NDArray, NDArray] | None:
        """Cumulative (u, v) in pixels on *frame*; zeros on frame 0."""
        if frame == 0:
            z = np.zeros(self._n, dtype=np.float64)
            return z, z
        idx = frame - 1
        disp = self._result.result_disp
        if idx < 0 or idx >= len(disp):
            return None
        fr = disp[idx]
        U = fr.U_accum if getattr(fr, "U_accum", None) is not None else fr.U
        if U is None:
            return None
        U = np.asarray(U, dtype=np.float64)
        return U[0::2], U[1::2]

    def node_field(
        self, field: str, frame: int, *, axes: str = IMAGE_AXES
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]] | None:
        """(values in pixel units, per-node validity), or None if absent.

        ``axes="world"`` reads ``disp_v`` positive up, as the node export
        writes it. The sign is applied to the nodes, before any statistic, so
        a maximum stays a maximum -- negating a reduced value would turn the
        image-axes maximum into the world-axes minimum.
        """
        validate_field(field)
        if axes not in (IMAGE_AXES, WORLD_AXES):
            raise ValueError(f"axes must be 'image' or 'world', got {axes!r}.")
        if is_strain_field(field):
            strains = self._result.result_strain
            if not strains:
                return None
            # Frame 0 is zero strain by definition. It takes frame 1's trim so
            # a region's valid fraction does not jump between frames 0 and 1.
            idx = max(frame - 1, 0)
            if idx >= len(strains):
                return None
            sr = strains[idx]
            vals = getattr(sr, field, None)
            if vals is None:
                return None
            vals = np.asarray(vals, dtype=np.float64)
            if frame == 0:
                vals = np.zeros(self._n, dtype=np.float64)
            valid = self.alive(frame) & np.isfinite(vals)
            trim = getattr(sr, "strain_valid", None)
            if trim is not None and len(trim) == self._n:
                valid &= np.asarray(trim, dtype=bool)
            return vals, valid
        uv = self._disp(frame)
        if uv is None:
            return None
        u, v = uv
        if field == "disp_u":
            vals = u
        elif field == "disp_v":
            vals = -v if axes == WORLD_AXES else v
        else:
            vals = np.hypot(u, v)
        return vals, self.alive(frame) & np.isfinite(vals)

    # -- sampling -------------------------------------------------------------

    def sample(
        self,
        plan: SamplePlan,
        field: str,
        frame: int,
        *,
        barrier: NDArray[np.bool_] | None = None,
        axes: str = IMAGE_AXES,
    ) -> Samples:
        """Read *plan* on *frame*. ``barrier`` is an extra per-simplex void."""
        got = self.node_field(field, frame, axes=axes)
        status = np.full(plan.size, SampleStatus.OUTSIDE, dtype=np.uint8)
        values = np.full(plan.size, np.nan, dtype=np.float64)
        if got is None:
            return Samples(values, status)
        vals, valid = got
        return self._gather(plan, vals, valid, frame, barrier)

    def _gather(
        self,
        plan: SamplePlan,
        vals: NDArray[np.float64],
        valid: NDArray[np.bool_],
        frame: int,
        barrier: NDArray[np.bool_] | None,
    ) -> Samples:
        status = np.full(plan.size, SampleStatus.OUTSIDE, dtype=np.uint8)
        values = np.full(plan.size, np.nan, dtype=np.float64)
        m = plan.material
        if not m.any():
            return Samples(values, status)

        corners = plan.corners[m]
        alive = self.alive(frame)[corners].all(axis=1)
        if barrier is not None:
            alive &= ~barrier[m]
        ok = alive & valid[corners].all(axis=1)

        st = np.where(
            ~alive, SampleStatus.CONSUMED,
            np.where(ok, SampleStatus.VALID, SampleStatus.UNRELIABLE),
        ).astype(np.uint8)
        status[m] = st

        v = np.full(int(m.sum()), np.nan, dtype=np.float64)
        if ok.any():
            v[ok] = np.einsum(
                "ij,ij->i", plan.weights[m][ok], vals[corners[ok]]
            )
        values[m] = v
        return Samples(values, status)

    def _barrier(self, mask: NDArray | None) -> NDArray[np.bool_] | None:
        """Per-simplex void for an extra per-frame mask (Python API route).

        Nodes on a void pixel of that mask count as consumed, and so does any
        triangle whose edge crosses it -- the analogue of what the cumulative
        transform does to a real run.
        """
        if mask is None:
            return None
        mask = np.asarray(mask)
        h, w = mask.shape[:2]
        xi = np.clip(np.round(self._nodes[:, 0]).astype(np.int64), 0, w - 1)
        yi = np.clip(np.round(self._nodes[:, 1]).astype(np.int64), 0, h - 1)
        on_void = mask[yi, xi] < 0.5
        crossing = cross_crack_simplices(
            self._tri.simplices, self._nodes[self._tri_ids], mask
        )
        return (crossing | on_void[self._simplices].any(axis=1)) & ~self._void

    def _barrier_for(self, plan: SamplePlan, frame_mask) -> NDArray | None:
        per_simplex = self._barrier(frame_mask)
        if per_simplex is None:
            return None
        simplex = self._tri.find_simplex(plan.xy)
        out = np.zeros(plan.size, dtype=bool)
        hit = simplex >= 0
        out[hit] = per_simplex[simplex[hit]]
        return out

    # -- capabilities -----------------------------------------------------

    @staticmethod
    def supports(probe: Probe, statistic: str) -> bool:
        """Whether *probe* produces *statistic* (and so joins that chart)."""
        return statistic in allowed_reductions(probe.kind)

    # -- series -------------------------------------------------------------

    def series(
        self,
        probe: Probe,
        field: str | None,
        statistic: str,
        *,
        pixel_size: float = 1.0,
        length_unit: str = "px",
        min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
        frames: Sequence[int] | None = None,
        masks: Sequence[NDArray | None] | None = None,
        axes: str = IMAGE_AXES,
    ) -> TimeSeries:
        """One curve: *probe* reduced to one number per frame.

        ``field`` is ignored for gauge quantities. ``masks``, when given, is
        one extra barrier mask per entry of *frames* (Python API route for
        runs whose ``U_accum`` carries no dead nodes).
        """
        if not self.supports(probe, statistic):
            raise ValueError(
                f"{statistic!r} does not apply to a {probe.kind} probe."
            )
        idxs = list(frames) if frames is not None else list(range(self.n_frames))
        if statistic in GAUGE_QUANTITIES:
            return self._gauge_series(
                probe, statistic, idxs, pixel_size, length_unit, masks
            )
        if field is None:
            raise ValueError(f"{statistic!r} needs a field.")
        validate_field(field)
        plan = self.plan(probe.geometry)
        scale = 1.0 if is_strain_field(field) else float(pixel_size)
        unit = ("" if statistic == "valid_fraction"
                else field_unit(field, length_unit))

        values = np.full(len(idxs), np.nan, dtype=np.float64)
        fractions = np.zeros(len(idxs), dtype=np.float64)
        status: list[FrameStatus] = []
        for i, frame in enumerate(idxs):
            got = self.node_field(field, frame, axes=axes)
            if got is None:
                status.append(FrameStatus.NOT_COMPUTED)
                continue
            vals, valid = got
            barrier = (self._barrier_for(plan, masks[i])
                       if masks is not None and i < len(masks) else None)
            s = self._gather(plan, vals, valid, frame, barrier)
            fractions[i] = s.valid_fraction
            crack = s.consumed
            if not s.material.any():
                status.append(FrameStatus.NO_DATA)
                continue
            if statistic == "valid_fraction":
                # A measure of the data, not a measurement made from it: the
                # threshold must not suppress it, and an empty frame is 0.
                values[i] = s.valid_fraction
                status.append(FrameStatus.CRACK if crack else FrameStatus.OK)
                continue
            if not s.valid.any():
                if crack:
                    status.append(FrameStatus.CRACK)
                elif (s.status == SampleStatus.UNRELIABLE).any():
                    status.append(FrameStatus.UNRELIABLE)
                else:
                    status.append(FrameStatus.NO_DATA)
                continue
            if probe.kind != "point" and s.valid_fraction < min_valid_fraction:
                status.append(FrameStatus.BELOW_THRESHOLD)
                continue
            values[i] = self._reduce(plan, s, vals, valid, frame, statistic) * scale
            status.append(FrameStatus.CRACK if crack else FrameStatus.OK)

        return TimeSeries(
            frames=np.asarray(idxs, dtype=np.int64),
            values=values,
            valid_fraction=fractions,
            status=status,
            unit=unit,
            axes=axes,
        )

    def _reduce(
        self,
        plan: SamplePlan,
        s: Samples,
        vals: NDArray[np.float64],
        valid: NDArray[np.bool_],
        frame: int,
        statistic: str,
    ) -> float:
        good = s.values[s.valid]
        if not good.size:
            return math.nan
        if statistic == "value":
            return float(good[0])
        if statistic == "mean":
            return float(np.mean(good))
        if statistic == "median":
            return float(np.median(good))
        if statistic == "std":
            return float(np.std(good))
        if statistic in ("max", "min"):
            pool = good
            if plan.node_ids is not None and plan.node_ids.size:
                ids = plan.node_ids
                ok = valid[ids] & self.alive(frame)[ids]
                if ok.any():
                    pool = np.concatenate([good, vals[ids][ok]])
            return float(np.max(pool) if statistic == "max" else np.min(pool))
        raise ValueError(f"Unknown statistic {statistic!r}.")

    def _gauge_series(
        self,
        probe: Probe,
        quantity: str,
        idxs: Sequence[int],
        pixel_size: float,
        length_unit: str,
        masks,
    ) -> TimeSeries:
        geom: LineGeom = probe.geometry  # type: ignore[assignment]
        ends = self.plan(PointGeom(geom.x0, geom.y0)), self.plan(
            PointGeom(geom.x1, geom.y1))
        path = self.plan(geom)
        l0 = geom.length()
        ex, ey = geom.direction()
        unit = length_unit if quantity in _LENGTH_GAUGES else ""
        scale = float(pixel_size) if quantity in _LENGTH_GAUGES else 1.0

        values = np.full(len(idxs), np.nan, dtype=np.float64)
        fractions = np.zeros(len(idxs), dtype=np.float64)
        status: list[FrameStatus] = []
        for i, frame in enumerate(idxs):
            uv = self._disp(frame)
            if uv is None:
                status.append(FrameStatus.NOT_COMPUTED)
                continue
            u, v = uv
            ok_nodes = self.alive(frame) & np.isfinite(u) & np.isfinite(v)
            frame_mask = masks[i] if masks is not None and i < len(masks) else None
            d = []
            for end in ends:
                barrier = self._barrier_for(end, frame_mask)
                su = self._gather(end, u, ok_nodes, frame, barrier)
                sv = self._gather(end, v, ok_nodes, frame, barrier)
                d.append((su.values[0], sv.values[0]))
            crossing = self._gather(
                path, u, ok_nodes, frame, self._barrier_for(path, frame_mask)
            ).consumed
            (du0, dv0), (du1, dv1) = d
            if not all(math.isfinite(c) for c in (du0, dv0, du1, dv1)):
                status.append(FrameStatus.ENDPOINT_LOST)
                continue
            fractions[i] = 1.0
            jx, jy = du1 - du0, dv1 - dv0
            if quantity in ("strain", "true_strain", "elongation"):
                length = math.hypot(
                    (geom.x1 + du1) - (geom.x0 + du0),
                    (geom.y1 + dv1) - (geom.y0 + dv0),
                )
                if quantity == "strain":
                    val = (length - l0) / l0
                elif quantity == "true_strain":
                    val = math.log(length / l0) if length > 0 else math.nan
                else:
                    val = length - l0
            elif quantity == "cod":
                val = jx * ex + jy * ey
            elif quantity == "cod_sliding":
                val = jx * (-ey) + jy * ex
            else:
                val = math.hypot(jx, jy)
            values[i] = val * scale
            status.append(FrameStatus.CRACK if crossing else FrameStatus.OK)

        return TimeSeries(
            frames=np.asarray(idxs, dtype=np.int64),
            values=values,
            valid_fraction=fractions,
            status=status,
            unit=unit,
        )

    # -- line views -----------------------------------------------------------

    def profile(
        self,
        probe: Probe,
        field: str,
        frame: int,
        *,
        pixel_size: float = 1.0,
        length_unit: str = "px",
        axes: str = IMAGE_AXES,
    ) -> Profile:
        k = self.kymograph(probe, field, frames=[frame], pixel_size=pixel_size,
                           length_unit=length_unit, axes=axes)
        return Profile(
            distance=k.distance, values=k.values[0], status=k.status[0],
            unit=k.unit, distance_unit=k.distance_unit,
        )

    def kymograph(
        self,
        probe: Probe,
        field: str,
        *,
        frames: Sequence[int] | None = None,
        pixel_size: float = 1.0,
        length_unit: str = "px",
        axes: str = IMAGE_AXES,
    ) -> Kymograph:
        if probe.kind != "line":
            raise ValueError("Profiles and kymographs need a line probe.")
        validate_field(field)
        plan = self.plan(probe.geometry)
        idxs = list(frames) if frames is not None else list(range(self.n_frames))
        scale = 1.0 if is_strain_field(field) else float(pixel_size)
        values = np.full((len(idxs), plan.size), np.nan, dtype=np.float64)
        status = np.full((len(idxs), plan.size), SampleStatus.OUTSIDE,
                         dtype=np.uint8)
        for i, frame in enumerate(idxs):
            s = self.sample(plan, field, frame, axes=axes)
            values[i] = s.values * scale
            status[i] = s.status
        return Kymograph(
            frames=np.asarray(idxs, dtype=np.int64),
            distance=plan.distance * float(pixel_size),
            values=values,
            status=status,
            unit=field_unit(field, length_unit),
            distance_unit=length_unit,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _with(plan: SamplePlan, **changes) -> SamplePlan:
    from dataclasses import replace

    return replace(plan, **changes)


def _median_spacing(nodes: NDArray[np.float64]) -> float:
    """Typical node spacing, when the run does not record its mesh step."""
    from scipy.spatial import cKDTree

    if len(nodes) < 2:
        return 1.0
    d, _ = cKDTree(nodes).query(nodes, k=2)
    spacing = float(np.median(d[:, 1]))
    return spacing if spacing > 0 else 1.0


def _in_area(geom: AreaGeom, pts: NDArray[np.float64]) -> NDArray[np.bool_]:
    x, y = pts[:, 0], pts[:, 1]
    if geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if geom.shape == "polygon":
        return _points_in_polygon(x, y, geom.data)  # type: ignore[arg-type]
    raise ValueError(f"Unknown area shape {geom.shape!r}.")


def _area_bounds(geom: AreaGeom) -> tuple[float, float, float, float]:
    if geom.shape == "rect":
        return tuple(geom.data)  # type: ignore[return-value]
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return (cx - r, cy - r, cx + r, cy + r)
    pts = np.asarray(geom.data, dtype=np.float64)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _area_centroid(geom: AreaGeom) -> tuple[float, float]:
    if geom.shape == "circle":
        return geom.data[0], geom.data[1]  # type: ignore[return-value]
    x0, y0, x1, y1 = _area_bounds(geom)
    if geom.shape == "rect":
        return (x0 + x1) / 2, (y0 + y1) / 2
    pts = np.asarray(geom.data, dtype=np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _area_grid(geom: AreaGeom, spacing: float) -> NDArray[np.float64]:
    """Cell-centred grid inside *geom* -- an area-weighted sample.

    Averaging nodes instead weights every node equally, which biases the
    result toward refined zones -- and refinement happens at cracks and edges.
    A region too small to hold one grid point is read at its centroid.
    """
    x0, y0, x1, y1 = _area_bounds(geom)
    h = max(float(spacing), 1e-6)
    # Cap the count for a huge region on a fine mesh; coarsen evenly.
    area = max((x1 - x0) * (y1 - y0), 0.0)
    if area / (h * h) > _MAX_AREA_SAMPLES:
        h = math.sqrt(area / _MAX_AREA_SAMPLES)
    xs = np.arange(x0 + h / 2, x1, h)
    ys = np.arange(y0 + h / 2, y1, h)
    if xs.size and ys.size:
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pts = pts[_in_area(geom, pts)]
        if len(pts):
            return pts
    cx, cy = _area_centroid(geom)
    return np.array([[cx, cy]], dtype=np.float64)


def _points_in_polygon(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    vertices: Sequence[tuple[float, float]],
) -> NDArray[np.bool_]:
    """Even-odd ray casting, vectorised over the query points."""
    verts = np.asarray(vertices, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        straddles = (yi > y) != (yj > y)
        denom = np.where(yj != yi, yj - yi, 1.0)
        crossing_x = (xj - xi) * (y - yi) / denom + xi
        inside ^= straddles & (x < crossing_x)
        j = i
    return inside


__all__ = [
    "AnalysisEngine", "DEFAULT_MIN_VALID_FRACTION", "GAUGE_QUANTITIES",
    "Kymograph", "Profile", "SPATIAL_STATISTICS", "SamplePlan", "SampleStatus",
    "Samples",
]
