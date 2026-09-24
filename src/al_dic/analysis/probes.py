"""Probe geometry and identity.

A probe is a shape placed on the reference image plus the metadata needed to
show it and tell it apart. It carries no results: reading a probe -- which
samples are valid, and the series they give -- is
``al_dic.analysis.engine``; editing one on screen is
``al_dic.analysis.geometry_edit``.

Coordinates are frame-0 (reference) image pixels, origin top-left, x = column,
y = row -- the same system as ``DICMesh.coordinates_fem``, so no probe ever
needs a coordinate transform. Geometry stays in pixels even when the project
displays physical units; the conversion happens at read-out, so changing the
unit setting cannot invalidate a probe someone already saved.

Probes are frozen. Editing goes through ``dataclasses.replace`` and
``ProbeSet.replace``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal, Sequence

ProbeKind = Literal["point", "line", "area"]
AreaShape = Literal["rect", "circle", "polygon"]

#: Statistics over a probe's samples on one frame, in display order.
SPATIAL_STATISTICS: tuple[str, ...] = (
    "mean", "median", "max", "min", "std", "valid_fraction",
)
#: Quantities from a line's two endpoints, in display order. The field does
#: not apply: engineering and true (log) strain, elongation, and the crack
#: opening's components along and across the gauge and its magnitude.
GAUGE_QUANTITIES: tuple[str, ...] = (
    "strain", "true_strain", "elongation",
    "cod", "cod_sliding", "cod_magnitude",
)
#: What a single point stands in for: a one-sample region. Not ``std`` or
#: ``valid_fraction``, which say nothing about one sample.
POINT_STATISTICS: tuple[str, ...] = ("value", "mean", "median", "max", "min")

_COLOUR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

#: Assigned in order to new probes. Chosen to stay distinguishable on the dark
#: theme and, as far as eight colours can, under common colour-vision
#: deficiencies.
DEFAULT_COLOURS: tuple[str, ...] = (
    "#ef4444", "#22c55e", "#3b82f6", "#eab308",
    "#a855f7", "#06b6d4", "#f97316", "#ec4899",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointGeom:
    """A single location."""

    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PointGeom":
        return PointGeom(x=float(d["x"]), y=float(d["y"]))


@dataclass(frozen=True)
class LineGeom:
    """A straight gauge between two endpoints. Not a polyline."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.length() <= 0.0:
            raise ValueError(
                "Line probe has zero length; its two endpoints coincide."
            )

    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    def direction(self) -> tuple[float, float]:
        """Unit vector from the first endpoint to the second."""
        n = self.length()
        return ((self.x1 - self.x0) / n, (self.y1 - self.y0) / n)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x0": float(self.x0), "y0": float(self.y0),
            "x1": float(self.x1), "y1": float(self.y1),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LineGeom":
        return LineGeom(
            x0=float(d["x0"]), y0=float(d["y0"]),
            x1=float(d["x1"]), y1=float(d["y1"]),
        )


@dataclass(frozen=True)
class AreaGeom:
    """A closed region: an axis-aligned rectangle, a circle or a polygon.

    ``shape`` is spelled ``"polygon"`` everywhere -- there is deliberately no
    ``"poly"`` alias. The reference implementation validates one spelling and
    sends the other, which makes its polygon probes unusable end to end while
    its tests still pass.
    """

    shape: AreaShape
    data: tuple[float, ...] | tuple[tuple[float, float], ...]

    @staticmethod
    def rect(x0: float, y0: float, x1: float, y1: float) -> "AreaGeom":
        """Normalised so a rectangle dragged in any direction is the same one."""
        lo_x, hi_x = sorted((float(x0), float(x1)))
        lo_y, hi_y = sorted((float(y0), float(y1)))
        if hi_x <= lo_x or hi_y <= lo_y:
            raise ValueError("Rectangle probe has zero width or height.")
        return AreaGeom(shape="rect", data=(lo_x, lo_y, hi_x, hi_y))

    @staticmethod
    def circle(cx: float, cy: float, radius: float) -> "AreaGeom":
        if radius <= 0.0:
            raise ValueError("Circle probe radius must be positive.")
        return AreaGeom(shape="circle", data=(float(cx), float(cy), float(radius)))

    @staticmethod
    def polygon(vertices: Sequence[tuple[float, float]]) -> "AreaGeom":
        if len(vertices) < 3:
            raise ValueError("Polygon probe needs at least three vertices.")
        return AreaGeom(
            shape="polygon",
            data=tuple((float(x), float(y)) for x, y in vertices),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.shape == "polygon":
            data: Any = [[float(x), float(y)] for x, y in self.data]  # type: ignore[misc]
        else:
            data = [float(v) for v in self.data]  # type: ignore[union-attr]
        return {"shape": self.shape, "data": data}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AreaGeom":
        shape = d["shape"]
        raw = d["data"]
        if shape == "rect":
            return AreaGeom.rect(*(float(v) for v in raw))
        if shape == "circle":
            return AreaGeom.circle(*(float(v) for v in raw))
        if shape == "polygon":
            return AreaGeom.polygon([(float(x), float(y)) for x, y in raw])
        raise ValueError(
            f"Unknown area shape {shape!r}; expected rect, circle or polygon."
        )


Geometry = PointGeom | LineGeom | AreaGeom

_GEOM_FOR_KIND: dict[str, type] = {
    "point": PointGeom,
    "line": LineGeom,
    "area": AreaGeom,
}


def allowed_reductions(kind: ProbeKind) -> frozenset[str]:
    """Reductions that mean something for *kind*.

    Asking for one that does not is an error rather than something to ignore
    silently, which is what the reference does.
    """
    if kind == "point":
        # A point is a one-sample region, so it can join a chart of region or
        # line means -- the same quantity on the same axis.
        return frozenset(POINT_STATISTICS)
    if kind == "line":
        return frozenset(SPATIAL_STATISTICS) | frozenset(GAUGE_QUANTITIES)
    if kind == "area":
        return frozenset(SPATIAL_STATISTICS)
    raise ValueError(f"Unknown probe kind {kind!r}.")


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    """A placed probe: geometry plus the metadata that identifies it."""

    id: int
    kind: ProbeKind
    geometry: Geometry
    label: str
    color: str
    visible: bool = True

    def __post_init__(self) -> None:
        expected = _GEOM_FOR_KIND.get(self.kind)
        if expected is None:
            raise ValueError(f"Unknown probe kind {self.kind!r}.")
        if not isinstance(self.geometry, expected):
            raise ValueError(
                f"Probe kind {self.kind!r} needs {expected.__name__}, "
                f"got {type(self.geometry).__name__}."
            )
        if not _COLOUR_RE.match(self.color):
            raise ValueError(
                f"Probe colour must be '#RRGGBB', got {self.color!r}."
            )


def probe_to_dict(probe: Probe) -> dict[str, Any]:
    """JSON-safe form. See ``docs`` in ``al_dic.gui.session`` for where it lands."""
    return {
        "id": int(probe.id),
        "kind": probe.kind,
        "geometry": probe.geometry.to_dict(),
        "label": probe.label,
        "color": probe.color,
        "visible": bool(probe.visible),
    }


def probe_from_dict(d: dict[str, Any]) -> Probe:
    kind = d.get("kind")
    geom_cls = _GEOM_FOR_KIND.get(kind)  # type: ignore[arg-type]
    if geom_cls is None:
        raise ValueError(
            f"Unknown probe kind {kind!r}; expected point, line or area."
        )
    return Probe(
        id=int(d["id"]),
        kind=kind,  # type: ignore[arg-type]
        geometry=geom_cls.from_dict(d["geometry"]),
        label=str(d["label"]),
        color=str(d["color"]),
        visible=bool(d.get("visible", True)),
    )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

@dataclass
class ProbeSet:
    """Ordered probes with unique ids.

    Ids are unique across every kind and are never reused, so a probe can be
    deleted, referenced or coloured by id alone.
    """

    _probes: list[Probe] = field(default_factory=list)
    _next_id: int = 1

    def __iter__(self) -> Iterator[Probe]:
        return iter(self._probes)

    def __len__(self) -> int:
        return len(self._probes)

    def get(self, probe_id: int) -> Probe:
        for p in self._probes:
            if p.id == probe_id:
                return p
        raise KeyError(f"No probe with id {probe_id}.")

    def of_kind(self, kind: ProbeKind) -> list[Probe]:
        return [p for p in self._probes if p.kind == kind]

    def add(
        self,
        kind: ProbeKind,
        geometry: Geometry,
        *,
        label: str | None = None,
        color: str | None = None,
    ) -> Probe:
        probe_id = self._next_id
        self._next_id += 1
        probe = Probe(
            id=probe_id,
            kind=kind,
            geometry=geometry,
            label=label if label else f"P{probe_id}",
            color=color or self._free_colour(probe_id),
        )
        self._probes.append(probe)
        return probe

    def _free_colour(self, probe_id: int) -> str:
        """First palette colour no current probe uses; cycle once all are.

        Following the id counter instead made colours repeat on one chart as
        soon as probes had been deleted.
        """
        used = {p.color.lower() for p in self._probes}
        for colour in DEFAULT_COLOURS:
            if colour.lower() not in used:
                return colour
        return DEFAULT_COLOURS[(probe_id - 1) % len(DEFAULT_COLOURS)]

    def replace(self, probe: Probe) -> None:
        """Swap the probe with the same id for *probe*, keeping its position."""
        for i, existing in enumerate(self._probes):
            if existing.id == probe.id:
                self._probes[i] = probe
                return
        raise KeyError(f"No probe with id {probe.id}.")

    def remove(self, probe_id: int) -> None:
        for i, existing in enumerate(self._probes):
            if existing.id == probe_id:
                del self._probes[i]
                return
        raise KeyError(f"No probe with id {probe_id}.")

    def clear(self) -> None:
        self._probes.clear()

    def to_list(self) -> list[dict[str, Any]]:
        return [probe_to_dict(p) for p in self._probes]

    def to_payload(self) -> dict[str, Any]:
        """What a session file stores: the probes and the id counter.

        The counter matters: rebuilt from the surviving ids alone, a deleted
        probe's id comes back after a reload.
        """
        return {"items": self.to_list(), "next_id": int(self._next_id)}

    @staticmethod
    def from_list(
        items: Sequence[dict[str, Any]], next_id: int | None = None
    ) -> "ProbeSet":
        if not isinstance(items, (list, tuple)):
            raise ValueError("Probe items must be a list.")
        probes = []
        for d in items:
            if not isinstance(d, dict):
                raise ValueError(f"A probe must be an object, got {d!r}.")
            probes.append(probe_from_dict(d))
        ids = [p.id for p in probes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Probe ids must be unique; duplicate in {ids}.")
        floor = max(ids, default=0) + 1
        start = max(int(next_id), floor) if next_id is not None else floor
        return ProbeSet(_probes=probes, _next_id=start)

    @staticmethod
    def from_payload(payload: Any) -> "ProbeSet":
        """Accept ``to_payload`` output, or the bare list early files used."""
        if payload is None:
            return ProbeSet()
        if isinstance(payload, dict):
            return ProbeSet.from_list(
                payload.get("items", []), payload.get("next_id")
            )
        return ProbeSet.from_list(payload)


__all__ = [
    "GAUGE_QUANTITIES", "POINT_STATISTICS", "SPATIAL_STATISTICS",
    "AreaGeom", "AreaShape", "DEFAULT_COLOURS", "Geometry", "LineGeom",
    "PointGeom", "Probe", "ProbeKind", "ProbeSet", "allowed_reductions",
    "probe_from_dict", "probe_to_dict", "replace",
]
