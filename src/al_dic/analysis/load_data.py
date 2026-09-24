"""A testing machine's record, read from CSV and matched to the image frames.

A tensile test has two clocks: the machine's, which logs load, and the
camera's. To pair a stress with an extensometer strain the load has to be
known at every frame. Two ways to get there:

* by frame number -- the machine was triggered by the camera, or logs its
  frame counter, so a column of the CSV names the frame of each row;
* by time -- both record time from their own start, and the camera started
  ``offset_s`` seconds into the machine's record. The load is interpolated at
  each frame's time, ``(frame index) / fps + offset_s``, and left undefined
  outside the record rather than extrapolated.

Machines write CSV in many dialects. The reader copes with the common ones:
comma, semicolon or tab separators; decimal commas where the separator is
not a comma; a title line above the header; a units row below it (whose
units then join the names: "Load" over "(kN)" becomes "Load (kN)").

Units are fixed where they meet: load in N (kN converted), area in mm², so
stress comes out in MPa (N/mm²).
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from numpy.typing import NDArray

SyncMode = Literal["frame", "time"]
LoadUnit = Literal["N", "kN"]

_DELIMITERS = (",", ";", "\t")
_DECIMAL_COMMA = re.compile(r"^\s*[-+]?\d+,\d+\s*$")
_UNIT_TO_N = {"N": 1.0, "kN": 1000.0}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadTable:
    """Named numeric columns, NaN where a cell was not a number."""

    names: tuple[str, ...]
    columns: tuple[NDArray[np.float64], ...]
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.names) != len(self.columns):
            raise ValueError("A load table needs one name per column.")
        if len({len(c) for c in self.columns}) > 1:
            raise ValueError("Every column of a load table must be as long.")

    @property
    def n_rows(self) -> int:
        return len(self.columns[0]) if self.columns else 0

    def column(self, name: str) -> NDArray[np.float64]:
        try:
            return self.columns[self.names.index(name)]
        except ValueError:
            raise ValueError(f"The load file has no column {name!r}.") from None

    def subset(self, names: Iterable[str]) -> "LoadTable":
        wanted = tuple(dict.fromkeys(names))
        return LoadTable(wanted, tuple(self.column(n) for n in wanted), self.source)


def _delimiter(lines: list[str]) -> str:
    """The separator most lines agree on."""
    sample = lines[:40]
    best, best_score = ",", -1
    for candidate in _DELIMITERS:
        counts = [line.count(candidate) for line in sample]
        used = [c for c in counts if c > 0]
        if not used:
            continue
        mode = max(set(used), key=used.count)
        score = sum(1 for c in counts if c == mode)
        if score > best_score:
            best, best_score = candidate, score
    return best


def _names(header: list[list[str]], width: int) -> tuple[str, ...]:
    """Column names from the rows above the numbers.

    Of the header rows as wide as the data, the last is taken as units when
    there are two or more, and the one above it as names.
    """
    full = [row for row in header if len(row) == width]
    names = [c.strip() for c in full[-2]] if len(full) >= 2 else (
        [c.strip() for c in full[-1]] if full else [""] * width)
    units = [c.strip().strip("()[]").strip() for c in full[-1]] if len(full) >= 2 \
        else [""] * width
    out: list[str] = []
    for j in range(width):
        name = names[j] if j < len(names) else ""
        unit = units[j] if j < len(units) else ""
        text = name or f"Column {j + 1}"
        if unit and name:
            text = f"{name} ({unit})"
        base, k = text, 2
        while text in out:
            text = f"{base} ({k})"
            k += 1
        out.append(text)
    return tuple(out)


def parse_load_table(text: str, source: str = "") -> LoadTable:
    """Parse CSV *text* from a testing machine into a ``LoadTable``."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The load file is empty.")
    delimiter = _delimiter(lines)
    rows = [row for row in csv.reader(lines, delimiter=delimiter)]
    decimal_comma = delimiter != "," and any(
        _DECIMAL_COMMA.match(cell) for row in rows for cell in row)

    def number(cell: str) -> float | None:
        s = cell.strip()
        if decimal_comma:
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    def is_numeric(row: list[str]) -> bool:
        cells = [c for c in row if c.strip()]
        found = sum(number(c) is not None for c in cells)
        return bool(cells) and found >= max(1, (len(cells) + 1) // 2)

    first = next((i for i, row in enumerate(rows) if is_numeric(row)), None)
    if first is None:
        raise ValueError("The load file holds no numbers.")
    data = [row for row in rows[first:] if is_numeric(row)]
    width = max(len(row) for row in data)
    names = _names(rows[:first], width)
    columns = []
    for j in range(width):
        values = [number(row[j]) if j < len(row) else None for row in data]
        columns.append(np.array([np.nan if v is None else v for v in values],
                                dtype=np.float64))
    return LoadTable(names, tuple(columns), source)


def read_load_table(path: str | Path) -> LoadTable:
    """Read a machine's CSV file; UTF-8 (with or without BOM), else Latin-1."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return parse_load_table(text, Path(path).name)


# ---------------------------------------------------------------------------
# Matching to frames
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadSync:
    """Which columns hold what, and how their rows meet the frames."""

    mode: SyncMode
    load_column: str
    frame_column: str | None = None
    time_column: str | None = None
    offset_s: float = 0.0           # machine time at the reference image
    load_unit: LoadUnit = "N"
    frame_base: int = 1             # the frame column's number for image 1

    def __post_init__(self) -> None:
        if self.mode not in ("frame", "time"):
            raise ValueError(f"Unknown sync mode {self.mode!r}; use frame or time.")
        if self.mode == "frame" and not self.frame_column:
            raise ValueError("Syncing by frame needs a frame column.")
        if self.mode == "time" and not self.time_column:
            raise ValueError("Syncing by time needs a time column.")
        if self.load_unit not in _UNIT_TO_N:
            raise ValueError(f"Unknown load unit {self.load_unit!r}; use N or kN.")
        if self.frame_base not in (0, 1):
            raise ValueError("The frame column counts from 0 or from 1.")

    @property
    def key_column(self) -> str:
        """The column that places a row in time: frame or time."""
        return self.frame_column if self.mode == "frame" else self.time_column  # type: ignore[return-value]


@dataclass(frozen=True)
class LoadData:
    """A machine record ready for the analysis: columns, mapping and A0."""

    table: LoadTable
    sync: LoadSync
    area_mm2: float | None = None
    source: str = field(default="")

    def __post_init__(self) -> None:
        for name in (self.sync.key_column, self.sync.load_column):
            self.table.column(name)             # raises, naming the column
        if self.area_mm2 is not None and not self.area_mm2 > 0:
            raise ValueError("The cross-section area must be positive.")
        if not self.source and self.table.source:
            object.__setattr__(self, "source", self.table.source)

    def load_n(self, n_frames: int, frame_rate: float) -> NDArray[np.float64]:
        """Load in N at each of *n_frames* frames; NaN where not covered."""
        load = self.table.column(self.sync.load_column) * _UNIT_TO_N[self.sync.load_unit]
        key = self.table.column(self.sync.key_column)
        ok = np.isfinite(key) & np.isfinite(load)
        if self.sync.mode == "frame":
            index = np.round(key[ok]).astype(np.int64) - self.sync.frame_base
            inside = (index >= 0) & (index < n_frames)
            sums = np.bincount(index[inside], weights=load[ok][inside], minlength=n_frames)
            counts = np.bincount(index[inside], minlength=n_frames)
            # Repeated rows for one frame are averaged.
            return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)[:n_frames]
        if not frame_rate > 0:
            raise ValueError("Syncing by time needs the camera's frame rate.")
        times, first = np.unique(key[ok], return_index=True)
        if len(times) == 0:
            return np.full(n_frames, np.nan)
        frame_times = np.arange(n_frames) / float(frame_rate) + self.sync.offset_s
        return np.interp(frame_times, times, load[ok][first], left=np.nan, right=np.nan)

    def stress_mpa(self, n_frames: int, frame_rate: float) -> NDArray[np.float64] | None:
        """Engineering stress F / A0 in MPa, or None without an area."""
        if self.area_mm2 is None:
            return None
        return self.load_n(n_frames, frame_rate) / self.area_mm2

    # -- persistence ---------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe form holding only the columns the mapping uses."""
        used = self.table.subset((self.sync.key_column, self.sync.load_column))
        return {
            "source": self.source,
            "sync": asdict(self.sync),
            "area_mm2": self.area_mm2,
            "columns": {
                name: [None if not np.isfinite(v) else float(v) for v in values]
                for name, values in zip(used.names, used.columns)
            },
        }

    @staticmethod
    def from_payload(payload: Any) -> "LoadData":
        try:
            columns = payload["columns"]
            names = tuple(str(n) for n in columns)
            arrays = tuple(
                np.array([np.nan if v is None else float(v) for v in columns[n]],
                         dtype=np.float64)
                for n in columns)
            area = payload.get("area_mm2")
            return LoadData(
                table=LoadTable(names, arrays, str(payload.get("source", ""))),
                sync=LoadSync(**payload["sync"]),
                area_mm2=None if area is None else float(area),
                source=str(payload.get("source", "")),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Unreadable load data: {exc}") from exc


__all__ = [
    "LoadData", "LoadSync", "LoadTable", "parse_load_table", "read_load_table",
]
