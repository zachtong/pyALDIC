"""Faithful (de)serialization of :class:`PipelineResult` to a NumPy archive.

A generic recursive encoder walks the dataclass tree (``PipelineResult`` and
every nested record), routing ndarrays into the archive and everything else
into a compact JSON manifest.  Arrays are **deduplicated by content**, so the
per-frame meshes -- identical across every frame in an accumulative run -- are
stored once instead of N times (a several-hundred-fold saving on big runs).

Records added to ``PipelineResult`` after 0.10.0 are stored beside the
manifest's ``root`` rather than inside it (``_SIDE_RECORDS``), so that older
builds, which decode only ``root``, still open the session without them.

Security: the archive is opened with ``allow_pickle=False`` (data only), so
loading an untrusted session file can never execute code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import Field, fields, is_dataclass, replace
from typing import Any, Callable

import numpy as np

from al_dic.core.data_structures import (
    DICMesh,
    DICPara,
    FrameResult,
    FrameSchedule,
    FrameTrackingStats,
    GridxyROIRange,
    PipelineResult,
    RefGridExtent,
    StrainResult,
    TrackingDiagnostics,
)

# Bump when the on-disk encoding changes incompatibly.
SERIALIZE_VERSION = 1
_MANIFEST_KEY = "__manifest__"


def _build_registry() -> dict[str, type]:
    """Map dataclass name -> class for reconstruction.

    Optional records (seed propagation) are imported lazily so a missing or
    refactored solver module never breaks core session loading.
    """
    classes: list[type] = [
        PipelineResult,
        DICPara,
        DICMesh,
        FrameResult,
        StrainResult,
        GridxyROIRange,
        FrameSchedule,
        TrackingDiagnostics,
        FrameTrackingStats,
        RefGridExtent,
    ]
    try:  # pragma: no cover - only present on seed-propagation runs
        from al_dic.solver.seed_prop_pipeline import ReseedEvent
        classes.append(ReseedEvent)
    except Exception:
        pass
    try:  # pragma: no cover
        from al_dic.solver.seed_propagation import Seed, SeedSet
        classes.extend([Seed, SeedSet])
    except Exception:
        pass
    return {c.__name__: c for c in classes}


_REGISTRY = _build_registry()

# PipelineResult fields added after 0.10.0, stored beside "root" in the
# manifest (field name -> record type). Builds up to 0.10.0 decode only
# "root" and reject record types they do not know, so a record inside
# "root" would make every session with results unreadable there.
_SIDE_RECORDS: dict[str, type] = {"tracking_diagnostics": TrackingDiagnostics}


def _array_key(arr: np.ndarray) -> str:
    """Content hash (dtype + shape + bytes) -> dedup key for an array."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return "arr_" + h.hexdigest()


def _omitted(f: Field, value: Any) -> bool:
    """A ``None`` field whose default is ``None`` is left out of the manifest.

    Decoding falls back to the default, so the round trip is unchanged, and
    a build that predates a field can still decode a record where it is
    ``None``: its decoder passes every manifest field to the constructor
    and would fail on an unknown one. This is what keeps the side records
    (``_SIDE_RECORDS``) out of ``root``.
    """
    return value is None and f.default is None


def _encode(obj: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Recursively encode *obj* to a JSON-able value, collecting arrays."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        key = _array_key(obj)
        if key not in arrays:
            arrays[key] = obj
        return {"__ndarray__": key}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [_encode(v, arrays) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        name = type(obj).__name__
        if name not in _REGISTRY:
            raise TypeError(
                f"Dataclass {name!r} is not in the serialization registry; "
                "add it to _build_registry()."
            )
        return {
            "__dataclass__": name,
            "fields": {
                f.name: _encode(getattr(obj, f.name), arrays)
                for f in fields(obj)
                if not _omitted(f, getattr(obj, f.name))
            },
        }
    raise TypeError(
        f"Cannot serialize object of type {type(obj).__name__!r} "
        "for a pyALDIC session."
    )


def _decode(val: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Inverse of :func:`_encode`."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, list):
        return [_decode(v, arrays) for v in val]
    if isinstance(val, dict):
        if "__ndarray__" in val:
            return arrays[val["__ndarray__"]]
        if "__tuple__" in val:
            return tuple(_decode(v, arrays) for v in val["__tuple__"])
        if "__dataclass__" in val:
            return _decode_dataclass(val, arrays)
        raise ValueError(f"Unrecognized encoded object with keys {list(val)}")
    raise ValueError(f"Cannot decode value of type {type(val).__name__!r}")


def _decode_dataclass(val: dict, arrays: dict[str, np.ndarray]) -> Any:
    """Rebuild a registered dataclass; unknown names or fields are a
    ``ValueError`` (a session error in the GUI), never a ``TypeError``."""
    name = val["__dataclass__"]
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown dataclass {name!r} in session archive.")
    known = {f.name for f in fields(cls) if f.init}
    unknown = sorted(set(val["fields"]) - known)
    if unknown:
        raise ValueError(
            f"Unknown field(s) {unknown} of {name!r} in session archive "
            "(saved by a newer pyALDIC?)."
        )
    kwargs = {k: _decode(v, arrays) for k, v in val["fields"].items()}
    return cls(**kwargs)


def serialize_result(
    result: PipelineResult,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return ``(arrays, manifest)`` for a :class:`PipelineResult`.

    *arrays* maps content-hash keys to the (deduplicated) numpy arrays;
    *manifest* is a JSON-able dict describing the object tree: ``root``,
    plus each side record (``_SIDE_RECORDS``) that is not ``None``.
    """
    arrays: dict[str, np.ndarray] = {}
    root = _encode(replace(result, **dict.fromkeys(_SIDE_RECORDS)), arrays)
    manifest: dict[str, Any] = {
        "serialize_version": SERIALIZE_VERSION, "root": root,
    }
    for name in _SIDE_RECORDS:
        value = getattr(result, name)
        if value is not None:
            manifest[name] = _encode(value, arrays)
    return arrays, manifest


def deserialize_result(
    arrays: dict[str, np.ndarray], manifest: dict
) -> PipelineResult:
    """Reconstruct a :class:`PipelineResult` from ``(arrays, manifest)``."""
    version = manifest.get("serialize_version")
    if version != SERIALIZE_VERSION:
        raise ValueError(
            f"Unsupported result serialize_version {version!r} "
            f"(this build reads {SERIALIZE_VERSION})."
        )
    obj = _decode(manifest["root"], arrays)
    if not isinstance(obj, PipelineResult):
        raise ValueError("Session archive root is not a PipelineResult.")
    side = _decode_side_records(manifest, arrays)
    return replace(obj, **side) if side else obj


def _decode_side_records(
    manifest: dict, arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """The side records present in *manifest*, type-checked."""
    side: dict[str, Any] = {}
    for name, cls in _SIDE_RECORDS.items():
        if manifest.get(name) is None:
            continue
        value = _decode(manifest[name], arrays)
        if not isinstance(value, cls):
            raise ValueError(
                f"Session record {name!r} is not a {cls.__name__}."
            )
        side[name] = value
    return side


def save_result_npz(
    file: Any,
    result: PipelineResult,
    *,
    compress: bool = True,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Write *result* to *file* (path or file-like) as a .npz archive.

    The JSON manifest is embedded as a uint8 array under ``__manifest__``.
    Writes the arrays one at a time (like ``np.savez`` does internally) so a
    *progress* callback can report a real fraction on gigabyte-scale saves,
    instead of a single opaque blocking call.
    """
    import zipfile
    from numpy.lib.format import write_array

    arrays, manifest = serialize_result(result)
    payload: dict[str, np.ndarray] = dict(arrays)
    payload[_MANIFEST_KEY] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    total = len(payload)
    done = 0
    with zipfile.ZipFile(file, "w", compression=compression, allowZip64=True) as zf:
        for key, arr in payload.items():
            # ``np.savez`` stores each array as ``<key>.npy``; matching that
            # naming lets ``np.load`` read the archive back transparently.
            with zf.open(key + ".npy", "w", force_zip64=True) as fid:
                write_array(fid, np.asanyarray(arr), allow_pickle=False)
            done += 1
            if progress is not None:
                progress(done / total)


def estimated_nbytes(result: PipelineResult) -> int:
    """Uncompressed, deduplicated size (bytes) of *result*'s arrays.

    Cheap (no I/O) -- used to show the user an approximate session size before
    a save. The on-disk file is smaller after zlib compression.
    """
    arrays, _ = serialize_result(result)
    return int(sum(int(a.nbytes) for a in arrays.values()))


def load_result_npz(file: Any) -> PipelineResult:
    """Read a :class:`PipelineResult` from a .npz archive (path or file-like)."""
    with np.load(file, allow_pickle=False) as npz:
        if _MANIFEST_KEY not in npz.files:
            raise ValueError(
                "Not a pyALDIC result archive (missing manifest)."
            )
        manifest = json.loads(bytes(npz[_MANIFEST_KEY].tobytes()).decode("utf-8"))
        arrays = {k: npz[k] for k in npz.files if k != _MANIFEST_KEY}
    return deserialize_result(arrays, manifest)
