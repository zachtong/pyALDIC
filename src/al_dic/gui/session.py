"""Session save / load for the GUI.

A session captures everything the user needs to resume a project:
image folder + per-frame Region-of-Interest masks + DIC parameters +
physical units. Pipeline *results* are not saved (the user re-runs
with the restored parameters — re-running is cheap compared to re-
configuring).

File format
-----------
Single JSON file with extension ``.aldic.json``. Mask images are
PNG-compressed and base64-encoded inline. Single-file sessions are
easier to share than a directory tree.

Schema is versioned (``schema_version`` key). Unknown versions raise
:class:`SessionError` — we never silently load a newer schema.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from al_dic.gui.app_state import AppState, RunState

SCHEMA_VERSION = 1


class SessionError(Exception):
    """Raised when a session file cannot be parsed or applied cleanly."""


@dataclass
class SessionData:
    """In-memory representation of a parsed session file.

    Separated from ``apply_session`` so parsing can be unit-tested
    without touching ``AppState``.
    """

    schema_version: int
    image_folder: str | None
    image_files: list[str]
    per_frame_rois: dict[int, NDArray[np.bool_]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    physical_units: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Mask (de)serialization
# ---------------------------------------------------------------------

def _encode_mask(mask: NDArray[np.bool_]) -> str:
    """Encode a boolean mask as base64-PNG for embedding in JSON."""
    # Convert bool -> uint8 (0 or 255) for PNG
    u8 = mask.astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise SessionError("cv2.imencode failed to encode mask as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_mask(b64_png: str) -> NDArray[np.bool_]:
    """Decode a base64-PNG string into a boolean mask."""
    raw = base64.b64decode(b64_png.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SessionError("cv2.imdecode failed to decode base64 PNG mask")
    return img > 127


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

# Parameters we serialize from AppState. Kept explicit (rather than
# introspecting AppState) so schema changes are reviewable and stable.
_PARAM_KEYS = (
    "subset_size",
    "subset_step",
    "search_range",
    "tracking_mode",
    "inc_ref_mode",
    "inc_ref_interval",
    "inc_custom_refs",
    "use_admm",
    "admm_max_iter",
    "init_guess_mode",
    "fft_reset_interval",
    "fft_auto_expand",
    "seed_ncc_threshold",
    # Note: state.seeds (list[SeedRecord]) is intentionally NOT persisted.
    # Seeds reference mesh node indices which are tied to ROI/winsize/step;
    # round-tripping through a session with different mesh params would yield
    # invalid indices. Users re-place seeds after loading a session.
    "refine_inner",
    "refine_outer",
    "refinement_level",
)

_PHYSICAL_KEYS = (
    "use_physical_units",
    "pixel_size",
    "pixel_unit",
    "frame_rate",
)


def save_session(path: Path, state: AppState) -> None:
    """Write the current AppState to a session file at ``path``.

    Overwrites the file if it already exists. Raises :class:`SessionError`
    on I/O failure.
    """
    path = Path(path)

    rois_payload: dict[str, str] = {
        # JSON only allows string keys, so frame indices go as str
        str(int(fidx)): _encode_mask(mask)
        for fidx, mask in state.per_frame_rois.items()
    }

    params = {k: _get_state_field(state, k) for k in _PARAM_KEYS}
    physical = {k: _get_state_field(state, k) for k in _PHYSICAL_KEYS}

    doc = {
        "schema_version": SCHEMA_VERSION,
        "image_folder": (
            str(state.image_folder) if state.image_folder is not None else None
        ),
        "image_files": list(state.image_files),
        "per_frame_rois": rois_payload,
        "params": params,
        "physical_units": physical,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except OSError as e:
        raise SessionError(f"Failed to write session to {path}: {e}") from e


def _get_state_field(state: AppState, key: str) -> Any:
    """Read a field from AppState, returning a JSON-serializable value.

    Returns ``None`` for missing attributes so older AppStates (or
    future ones with renamed fields) do not crash the save.
    """
    value = getattr(state, key, None)
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_session(path: Path) -> SessionData:
    """Parse a session file into a :class:`SessionData`.

    Raises :class:`SessionError` if the file cannot be read, is not
    valid JSON, or reports an unsupported schema version.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SessionError(f"Cannot open session file {path}: {e}") from e

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SessionError(
            f"Session file {path} is not valid JSON: {e}"
        ) from e

    if not isinstance(doc, dict):
        raise SessionError("Session file must contain a JSON object")

    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SessionError(
            f"Unsupported session schema version {version!r}; "
            f"this build writes version {SCHEMA_VERSION}."
        )

    # Decode ROIs — accept both int-like and str keys
    rois: dict[int, NDArray[np.bool_]] = {}
    for k, v in (doc.get("per_frame_rois") or {}).items():
        try:
            fidx = int(k)
        except (TypeError, ValueError) as e:
            raise SessionError(
                f"Invalid frame index {k!r} in per_frame_rois"
            ) from e
        rois[fidx] = _decode_mask(v)

    return SessionData(
        schema_version=version,
        image_folder=doc.get("image_folder"),
        image_files=list(doc.get("image_files") or []),
        per_frame_rois=rois,
        params=dict(doc.get("params") or {}),
        physical_units=dict(doc.get("physical_units") or {}),
    )


# ---------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------

def apply_session(session: SessionData, state: AppState, image_ctrl) -> None:
    """Apply a parsed :class:`SessionData` to the running ``AppState``.

    Re-loads images from ``session.image_folder`` via ``image_ctrl`` so
    the canvas has pixel data to render. If the folder is missing or no
    longer contains the expected files, the images are left empty and
    the caller should surface a warning — we do not raise here so the
    user can still recover the parameters / masks.
    """
    # Reset any running pipeline to IDLE before re-loading
    state.set_run_state(RunState.IDLE)
    state.results = None

    # Image folder re-load (best effort)
    if session.image_folder:
        folder = Path(session.image_folder)
        if folder.exists():
            try:
                image_ctrl.load_folder(folder)
            except Exception as e:  # surface to log, do not fail whole load
                state.log_message.emit(
                    f"Could not re-load images from {folder}: "
                    f"{type(e).__name__}: {e}",
                    "warn",
                )
        else:
            state.log_message.emit(
                f"Image folder '{folder}' no longer exists; session "
                "parameters and Regions of Interest were restored but "
                "images must be re-selected manually.",
                "warn",
            )

    # Per-frame ROIs
    state.per_frame_rois = dict(session.per_frame_rois)

    # Params — whitelist-driven assignment so unknown keys are dropped
    for k in _PARAM_KEYS:
        if k in session.params:
            try:
                state.set_param(k, session.params[k])
            except Exception:
                # set_param may not exist for all fields; fall back
                setattr(state, k, session.params[k])

    for k in _PHYSICAL_KEYS:
        if k in session.physical_units:
            setattr(state, k, session.physical_units[k])

    # Broadcast changes so widgets re-sync
    state.params_changed.emit()
    state.roi_changed.emit()
    state.physical_units_changed.emit()
