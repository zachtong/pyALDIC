"""Session save / load for the GUI.

A session captures everything the user needs to resume a project: image folder
+ per-frame Regions of Interest + DIC parameters + physical units + the
current *view state* (which field/frame, colormaps, ranges) and -- new in
schema 2 -- the computed **pipeline results**, so a long run is not lost when
the GUI is closed.

File format
-----------
Schema 2 writes a single ``.aldic`` **bundle** (a zip) containing:

    session.json   -- human-readable config + view state + fingerprint
    results.npz    -- faithful PipelineResult archive (only when results exist)

Schema 1 wrote a plain ``.aldic.json`` file (config only, no results). Both are
accepted on load: :func:`load_session` sniffs the file (zip vs. JSON) rather
than trusting the extension.

Schema is versioned (``schema_version``). Unknown versions raise
:class:`SessionError` -- we never silently load a newer schema.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import PipelineResult
from al_dic.gui.app_state import AppState, RunState
from al_dic.io.session_serialize import load_result_npz, save_result_npz

SCHEMA_VERSION = 3
_CONFIG_NAME = "session.json"
_RESULTS_NAME = "results.npz"

ProgressFn = Callable[[float, str], None]


class SessionError(Exception):
    """Raised when a session file cannot be parsed or applied cleanly."""


@dataclass
class SessionData:
    """In-memory representation of a parsed session file.

    Separated from ``apply_session`` so parsing can be unit-tested without
    touching ``AppState``.
    """

    schema_version: int
    image_folder: str | None
    image_files: list[str]
    per_frame_rois: dict[int, NDArray[np.bool_]] = field(default_factory=dict)
    refine_brush_mask: NDArray[np.bool_] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    physical_units: dict[str, Any] = field(default_factory=dict)
    view_state: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    # Serialised probe definitions; empty for sessions written before schema 3.
    probes: list[dict[str, Any]] = field(default_factory=list)
    results: PipelineResult | None = None


# ---------------------------------------------------------------------
# Mask (de)serialization
# ---------------------------------------------------------------------

def _encode_mask(mask: NDArray[np.bool_]) -> str:
    """Encode a boolean mask as base64-PNG for embedding in JSON."""
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
# Config (params / physical units / view state)
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

# View state so the user lands back on exactly the page they left.
_VIEW_SCALAR_KEYS = (
    "current_frame",
    "display_field",
    "show_deformed",
    "show_background",
    "hidden_bg_color",
    "overlay_alpha",
    "show_mesh",
    "mesh_line_color",
    "mesh_line_width",
    "show_subset_window",
)


def _build_view_state(state: AppState) -> dict[str, Any]:
    """Snapshot the current display settings (field, frame, colormaps...)."""
    vs: dict[str, Any] = {
        k: _get_state_field(state, k) for k in _VIEW_SCALAR_KEYS
    }
    # Per-field colormap + range. Access the private store directly (same
    # package); the public get_field_state() would create entries we don't
    # want to force into existence here.
    field_states = getattr(state, "_field_states", {})
    vs["field_states"] = {
        name: {
            "auto": bool(fs.auto),
            "vmin": float(fs.vmin),
            "vmax": float(fs.vmax),
            "colormap": str(fs.colormap),
        }
        for name, fs in field_states.items()
    }
    return vs


def _apply_view_state(vs: dict[str, Any], state: AppState) -> None:
    """Restore display settings.  current_frame is applied by the caller once
    images are loaded (it is clamped to the image count)."""
    for name, d in (vs.get("field_states") or {}).items():
        fs = state.get_field_state(name)
        fs.auto = bool(d.get("auto", fs.auto))
        fs.vmin = float(d.get("vmin", fs.vmin))
        fs.vmax = float(d.get("vmax", fs.vmax))
        fs.colormap = str(d.get("colormap", fs.colormap))
    # Drive the restore off the same list the save side writes, so a key added
    # to _VIEW_SCALAR_KEYS cannot be saved and then silently never restored.
    # current_frame is excluded here: the caller applies it after images load
    # so it can be clamped to the image count.
    for k in _VIEW_SCALAR_KEYS:
        if k in ("current_frame", "display_field"):
            continue
        if k in vs and vs[k] is not None:
            setattr(state, k, vs[k])
    # Set the display field through its setter (after field states are
    # restored) so display_changed fires and the colormap combo, colour-range
    # controls and canvas all re-sync to the restored field + its colours.
    df = vs.get("display_field")
    if df:
        state.set_display_field(str(df))


def _build_config(state: AppState, has_results: bool,
                  fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Assemble the JSON config document (schema 2)."""
    rois_payload: dict[str, str] = {
        str(int(fidx)): _encode_mask(mask)
        for fidx, mask in state.per_frame_rois.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "image_folder": (
            str(state.image_folder) if state.image_folder is not None else None
        ),
        "image_files": list(state.image_files),
        "per_frame_rois": rois_payload,
        # The painted refinement zones feed BrushRegionCriterion, so losing
        # them silently changes the mesh -- and the results -- on a re-run of
        # a restored session.
        "refine_brush_mask": (
            _encode_mask(state.refine_brush_mask)
            if state.refine_brush_mask is not None else None
        ),
        # Post-processing probes. A probe someone placed and named is work,
        # and losing it on save/reopen would be as surprising as losing a
        # Region of Interest.
        "probes": state.probes.to_list(),
        "params": {k: _get_state_field(state, k) for k in _PARAM_KEYS},
        "physical_units": {k: _get_state_field(state, k) for k in _PHYSICAL_KEYS},
        "view_state": _build_view_state(state),
        "has_results": has_results,
        "fingerprint": fingerprint,
    }


def _fingerprint(state: AppState) -> dict[str, Any]:
    """Small summary used to warn on load if things clearly don't match."""
    fp: dict[str, Any] = {
        "n_images": len(state.image_files),
        "image_names": [Path(f).name for f in state.image_files],
    }
    res = state.results
    if res is not None:
        fp["n_frames"] = len(res.result_disp)
        fp["n_nodes"] = int(res.dic_mesh.coordinates_fem.shape[0])
        fp["has_strain"] = bool(res.result_strain)
    return fp


def _get_state_field(state: AppState, key: str) -> Any:
    """Read a field from AppState as a JSON-serializable value."""
    value = getattr(state, key, None)
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_session(
    path: Path,
    state: AppState,
    *,
    include_results: bool = True,
    progress: ProgressFn | None = None,
) -> None:
    """Write the current AppState to an ``.aldic`` bundle at ``path``.

    When *include_results* is True and results exist, the full PipelineResult
    is written into the bundle so the run can be restored without recomputing.
    Overwrites atomically (temp file + rename). Raises :class:`SessionError`.
    """
    def _report(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    path = Path(path)
    results = state.results if include_results else None
    _report(0.02, "Preparing session…")
    config = _build_config(state, has_results=results is not None,
                           fingerprint=_fingerprint(state))

    tmp = path.with_name(path.name + ".tmp")
    results_tmp: Path | None = None
    try:
        if results is not None:
            # Stream results to a temp .npz first (low peak memory), then add
            # it to the bundle without a second full copy in RAM. Per-array
            # progress maps onto 10-80% so the bar tracks a gigabyte-scale save.
            _report(0.10, "Serializing results…")
            results_tmp = path.with_name(path.name + ".results.tmp.npz")
            save_result_npz(
                str(results_tmp), results, compress=True,
                progress=lambda f: _report(0.10 + 0.70 * f, "Serializing results…"),
            )

        _report(0.85, "Writing bundle…")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr(_CONFIG_NAME, json.dumps(config, indent=2))
            if results_tmp is not None:
                # results.npz is already zlib-compressed internally.
                zf.write(str(results_tmp), _RESULTS_NAME)
        tmp.replace(path)
        _report(1.0, "Done.")
    except OSError as e:
        raise SessionError(f"Failed to write session to {path}: {e}") from e
    finally:
        for p in (tmp, results_tmp):
            if p is not None and p.exists() and p != path:
                try:
                    p.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_session(path: Path, *, progress: ProgressFn | None = None) -> SessionData:
    """Parse a session file (v2 ``.aldic`` bundle or v1 ``.aldic.json``).

    Format is detected from the file contents (zip vs. JSON), not the
    extension. Raises :class:`SessionError` on any parse/read failure or an
    unsupported schema version.
    """
    def _report(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    path = Path(path)
    _report(0.02, "Opening session…")
    try:
        is_zip = zipfile.is_zipfile(path)
    except OSError as e:
        raise SessionError(f"Cannot open session file {path}: {e}") from e

    if is_zip:
        return _load_bundle(path, _report)
    return _load_legacy_json(path)


def _load_legacy_json(path: Path) -> SessionData:
    """Load a schema-1 plain-JSON session (config only, no results)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SessionError(f"Cannot open session file {path}: {e}") from e
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SessionError(f"Session file {path} is not valid JSON: {e}") from e
    return _parse_config(doc)


def _load_bundle(path: Path, report: ProgressFn) -> SessionData:
    """Load a schema-2 ``.aldic`` bundle (config + optional results.npz)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if _CONFIG_NAME not in names:
                raise SessionError(
                    f"'{path.name}' is not a pyALDIC session (no {_CONFIG_NAME})."
                )
            report(0.10, "Reading configuration…")
            doc = json.loads(zf.read(_CONFIG_NAME).decode("utf-8"))
            data = _parse_config(doc)

            if _RESULTS_NAME in names:
                report(0.30, "Restoring results…")
                # Extract to a temp file so np.load can read arrays lazily
                # instead of holding the whole archive in RAM.
                # A temp file in the system temp dir, not beside the input:
                # reading a session must not require write permission on the
                # folder holding it (read-only share, mounted image, USB stick).
                _fd, _name = tempfile.mkstemp(suffix=".npz")
                os.close(_fd)
                tmp = Path(_name)
                try:
                    with zf.open(_RESULTS_NAME) as src, open(tmp, "wb") as dst:
                        _copy_stream(src, dst)
                    data.results = load_result_npz(str(tmp))
                finally:
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
    except zipfile.BadZipFile as e:
        raise SessionError(f"Session file {path} is a corrupt archive: {e}") from e
    except (OSError, ValueError, KeyError) as e:
        raise SessionError(f"Failed to read session {path}: {e}") from e
    report(1.0, "Loaded.")
    return data


def _copy_stream(src, dst, chunk: int = 1 << 20) -> None:
    while True:
        buf = src.read(chunk)
        if not buf:
            break
        dst.write(buf)


def _parse_config(doc: Any) -> SessionData:
    """Parse a config JSON dict (schema 1 or 2) into a SessionData."""
    if not isinstance(doc, dict):
        raise SessionError("Session config must be a JSON object.")
    version = doc.get("schema_version")
    if version not in (1, 2, 3):
        raise SessionError(
            f"Unsupported session schema version {version!r}; "
            f"this build reads versions 1 to 3."
        )
    rois: dict[int, NDArray[np.bool_]] = {}
    for k, v in (doc.get("per_frame_rois") or {}).items():
        try:
            fidx = int(k)
        except (TypeError, ValueError) as e:
            raise SessionError(f"Invalid frame index {k!r} in per_frame_rois") from e
        rois[fidx] = _decode_mask(v)

    # Absent in sessions written before brush zones were persisted.
    brush_payload = doc.get("refine_brush_mask")
    brush = _decode_mask(brush_payload) if brush_payload else None

    # Absent before schema 3. A malformed entry is worth reporting rather
    # than dropping: the geometry is user work, not a derived cache.
    probes_payload = doc.get("probes") or []
    if not isinstance(probes_payload, list):
        raise SessionError("Session 'probes' must be a list.")

    return SessionData(
        schema_version=version,
        image_folder=doc.get("image_folder"),
        image_files=list(doc.get("image_files") or []),
        per_frame_rois=rois,
        refine_brush_mask=brush,
        probes=probes_payload,
        params=dict(doc.get("params") or {}),
        physical_units=dict(doc.get("physical_units") or {}),
        view_state=dict(doc.get("view_state") or {}),
        fingerprint=dict(doc.get("fingerprint") or {}),
        results=None,
    )


# ---------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------

def _resolve_image_folder(
    session: SessionData, session_path: str | Path | None,
) -> str | None:
    """Best-effort location of the session's image folder.

    Order: the absolute folder saved in the session; then -- for the common
    "moved the whole project" case -- folders relative to the .aldic file: a
    subfolder with the saved folder's name, then the .aldic's own directory.
    A relative candidate is accepted only if it actually holds the session's
    first image (matched by basename), so an unrelated directory is never
    silently adopted.  Returns the folder path, or ``None`` if unresolved.
    """
    saved = session.image_folder
    if saved and Path(saved).is_dir():
        return str(saved)
    if not session_path or not session.image_files:
        return None
    base = Path(session_path).parent
    first = Path(session.image_files[0]).name
    candidates: list[Path] = []
    if saved:
        candidates.append(base / Path(saved).name)
    candidates.append(base)
    for cand in candidates:
        try:
            if cand.is_dir() and (cand / first).exists():
                return str(cand)
        except OSError:
            continue
    return None


def apply_session(
    session: SessionData,
    state: AppState,
    image_ctrl,
    session_path: str | Path | None = None,
    locate_folder_cb=None,
) -> None:
    """Apply a parsed :class:`SessionData` to the running ``AppState``.

    Re-loads images (best effort, relocating them if the project moved),
    restores parameters / Regions of Interest / physical units / view state,
    and -- for schema-2 sessions -- restores the computed results so the fields
    display immediately without recomputation.

    Args:
        session_path: Path of the .aldic file, used to relocate images that
            moved together with the project.
        locate_folder_cb: Optional ``(missing_folder) -> str | None`` callback
            the GUI supplies to prompt the user for the image folder when it
            cannot be located automatically.  Called BEFORE results are
            restored, because loading images clears results -- doing it after
            would discard the just-restored fields.
    """
    state.set_run_state(RunState.IDLE)
    state.results = None

    # Probes first: they are plain geometry with no dependency on images or
    # results, so restoring them cannot fail for reasons the user has to fix.
    from al_dic.analysis.probes import ProbeSet

    try:
        state.probes = ProbeSet.from_list(session.probes)
    except (ValueError, KeyError, TypeError) as exc:
        raise SessionError(f"Session contains an unreadable probe: {exc}") from exc

    # Image folder re-load.  Locate the images (they may have moved with the
    # project) and load them BEFORE results are restored below: image loading
    # clears results via set_image_files(), so the order is load-then-restore.
    if session.image_folder:
        folder = _resolve_image_folder(session, session_path)
        if folder is None and locate_folder_cb is not None:
            picked = locate_folder_cb(session.image_folder)
            if picked and Path(picked).is_dir():
                folder = str(picked)
        if folder is not None:
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
                f"Image folder '{session.image_folder}' no longer exists and "
                "could not be relocated; session parameters, Regions of "
                "Interest and results were restored, but background images "
                "must be re-selected manually.",
                "warn",
            )

    # Per-frame ROIs, and the painted refinement zones that go with them.
    state.per_frame_rois = dict(session.per_frame_rois)
    state.refine_brush_mask = (
        session.refine_brush_mask.copy()
        if session.refine_brush_mask is not None else None
    )

    # Params — whitelist-driven assignment so unknown keys are dropped
    for k in _PARAM_KEYS:
        if k in session.params:
            try:
                state.set_param(k, session.params[k])
            except Exception:
                setattr(state, k, session.params[k])

    for k in _PHYSICAL_KEYS:
        if k in session.physical_units:
            setattr(state, k, session.physical_units[k])

    # Results BEFORE view state, so the display_changed redraw triggered while
    # restoring the view has the fields to draw.
    if session.results is not None:
        state.results = session.results

    # View state (colormaps / ranges / display field / mesh appearance)
    if session.view_state:
        _apply_view_state(session.view_state, state)

    # current_frame after images are loaded so it clamps correctly
    cur = session.view_state.get("current_frame")
    if cur is not None and state.image_files:
        state.set_current_frame(int(cur))

    # Broadcast changes so widgets re-sync
    state.params_changed.emit()
    state.roi_changed.emit()
    state.physical_units_changed.emit()
    state.results_changed.emit()

    # A schema-2 session carries computed results, so the pipeline is
    # effectively DONE.  Transition the run state last (mirrors
    # pipeline_controller._on_finished) so results-gated UI -- the Export
    # button (enabled only in DONE) and the auto-opened strain window --
    # reflects the restored results without forcing a recompute.  Sessions
    # without results (schema-1 / displacement-only saves) stay IDLE.
    if state.results is not None:
        state.set_run_state(RunState.DONE)
