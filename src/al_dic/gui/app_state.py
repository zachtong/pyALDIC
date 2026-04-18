"""Central application state with Qt Signals.

All GUI panels subscribe to signals -- no direct panel-to-panel communication.
Workers emit signals back to the main thread; controllers update AppState.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field as _dc_field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from al_dic.core.data_structures import PipelineResult


class RunState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


@dataclass
class FieldColorState:
    """Per-field color range and colormap settings."""
    auto: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    colormap: str = "turbo"


@dataclass
class SeedRecord:
    """One user-placed (or auto-warped) seed for seed_propagation mode.

    Attributes:
        node_idx: Index into the current preview mesh's coordinates_fem.
            Updated by SeedController.re_snap_seeds() when ROI / winsize /
            step changes (Q3-B in the Phase 5 plan).
        region_id: Connected-component region id from precompute_node_regions
            on the current ROI mask.
        is_warped: True iff this seed was carried over from a previous
            reference frame via warp_seeds_to_new_ref (purple in canvas);
            False if user placed it manually (yellow).
        ncc_peak: Most recent bootstrap NCC value (populated after a run).
            None before any pipeline execution.
        xy_canvas: Image-pixel (x, y) where the user originally clicked.
            Source of truth for re-snap: when the mesh changes, find the
            new nearest node to xy_canvas and update node_idx. None for
            seeds added programmatically (e.g., from Python REPL).
    """

    node_idx: int
    region_id: int
    is_warped: bool = False
    ncc_peak: float | None = None
    xy_canvas: tuple[float, float] | None = None



def _default_colormap(field_name: str) -> str:
    """Return the default colormap for a given field name."""
    return "turbo"


class AppState(QObject):
    """Singleton state container for the entire GUI."""

    _instance: AppState | None = None

    # --- Signals ---
    images_changed = Signal()
    current_frame_changed = Signal(int)
    roi_changed = Signal()
    params_changed = Signal()
    run_state_changed = Signal(RunState)
    progress_updated = Signal(float, str)
    results_changed = Signal()
    display_changed = Signal()
    log_message = Signal(str, str)  # (message, level)
    # Fatal errors that should be surfaced as a modal dialog in addition to
    # being written to the console log. Emitted as (title, message) where
    # message is user-facing (plain English, no tracebacks). Stack traces
    # should still go through log_message with level "error".
    fatal_error = Signal(str, str)
    physical_units_changed = Signal()
    # Emitted when state.seeds is mutated (add/remove/re-snap/clear).
    # Canvas overlay subscribes to redraw the seed markers.
    seeds_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._init_state()

    def _init_state(self) -> None:
        # Images
        self.image_folder: Path | None = None
        self.image_files: list[str] = []
        self.current_frame: int = 0
        # ROI — per-frame system (frame_idx -> bool mask)
        self.per_frame_rois: dict[int, NDArray[np.bool_]] = {}
        # Per-frame masks in deformed coordinates (optional, e.g. from segmentation)
        # Mapping: frame_idx -> bool mask.  When set, deformed display uses these
        # directly instead of warping the reference roi_mask.
        self.deformed_masks: dict[int, NDArray[np.bool_]] | None = None
        # Parameters
        self.subset_size: int = 40
        self.subset_step: int = 16
        self.search_range: int = 20
        self.tracking_mode: str = "accumulative"
        self.inc_ref_mode: str = "every_frame"
        self.inc_ref_interval: int = 5
        self.inc_custom_refs: list[int] = []
        # Solver selection
        self.use_admm: bool = True       # True = AL-DIC (ADMM), False = Local DIC
        self.admm_max_iter: int = 3      # ADMM iteration count (1-10)
        # Initial guess parameters
        # Modes: "previous" | "fft_ref_update" | "fft_every" | "fft_reset_n"
        # | "seed_propagation"
        self.init_guess_mode: str = "previous"
        self.fft_reset_interval: int = 5
        self.fft_auto_expand: bool = True
        # Seed propagation parameters (only relevant when
        # init_guess_mode == "seed_propagation").
        self.seeds: list[SeedRecord] = []
        self.seed_ncc_threshold: float = 0.70
        # Computation
        self.run_state: RunState = RunState.IDLE
        self.progress: float = 0.0
        self.progress_message: str = ""
        self.elapsed_seconds: float = 0.0
        self.results: PipelineResult | None = None
        # Display
        self.display_field: str = "disp_u"
        self.show_deformed: bool = True
        self.roi_editing: bool = False
        # Per-field color state (colormap + auto/manual range).
        # Access via get_field_state() or the proxy properties below.
        self._field_states: dict[str, FieldColorState] = {}
        self.overlay_alpha: float = 0.7
        self.show_mesh: bool = True
        self.show_subset_window: bool = False
        # Mesh line appearance (user-configurable)
        self.mesh_line_color: str = "#ffffff"
        self.mesh_line_width: int = 1
        # Physical units
        self.use_physical_units: bool = False
        self.pixel_size: float = 1.0    # value in units of pixel_unit per px
        self.pixel_unit: str = "mm"     # unit string: nm, μm, mm, cm, m, inch
        self.frame_rate: float = 1.0    # fps
        # Mesh refinement (Item 1: inner/outer boundary refinement)
        self.refine_inner: bool = False
        self.refine_outer: bool = False
        # User-painted brush refinement mask in frame-0 image coordinates.
        # When set, BrushRegionCriterion is added to the refinement policy
        # and is auto-warped to subsequent reference frames inside pipeline.
        self.refine_brush_mask: NDArray[np.bool_] | None = None
        # Refinement level: 1=light, 2=medium, 3=heavy.
        # min_element_size = max(4, subset_step // 2**level)
        self.refinement_level: int = 1

    @property
    def roi_mask(self) -> NDArray[np.bool_] | None:
        """Backward-compatible access to frame-0 ROI mask."""
        return self.per_frame_rois.get(0)

    # ------------------------------------------------------------------
    # Per-field color state accessors
    # ------------------------------------------------------------------

    def _get_or_create_field_state(self, field_name: str) -> FieldColorState:
        """Return the FieldColorState for *field_name*, creating it on first access."""
        if field_name not in self._field_states:
            self._field_states[field_name] = FieldColorState(
                colormap=_default_colormap(field_name)
            )
        return self._field_states[field_name]

    def get_field_state(self, field_name: str) -> FieldColorState:
        """Return the stored color state for the given field (public API)."""
        return self._get_or_create_field_state(field_name)

    # Proxy properties: read/write through the *current* display_field's state.
    # All existing code that uses state.colormap / color_auto / color_min / color_max
    # continues to work without modification.

    @property
    def colormap(self) -> str:
        return self._get_or_create_field_state(self.display_field).colormap

    @colormap.setter
    def colormap(self, value: str) -> None:
        self._get_or_create_field_state(self.display_field).colormap = value

    @property
    def color_auto(self) -> bool:
        return self._get_or_create_field_state(self.display_field).auto

    @color_auto.setter
    def color_auto(self, value: bool) -> None:
        self._get_or_create_field_state(self.display_field).auto = value

    @property
    def color_min(self) -> float:
        return self._get_or_create_field_state(self.display_field).vmin

    @color_min.setter
    def color_min(self, value: float) -> None:
        self._get_or_create_field_state(self.display_field).vmin = float(value)

    @property
    def color_max(self) -> float:
        return self._get_or_create_field_state(self.display_field).vmax

    @color_max.setter
    def color_max(self, value: float) -> None:
        self._get_or_create_field_state(self.display_field).vmax = float(value)

    @classmethod
    def instance(cls) -> AppState:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        self._init_state()

    # --- Setters (emit signals) ---
    def set_image_files(self, files: list[str]) -> None:
        # Clear previous results and ROI when loading new images
        had_results = self.results is not None
        had_roi = bool(self.per_frame_rois) or self.refine_brush_mask is not None
        self.results = None
        self.deformed_masks = None
        self.per_frame_rois = {}
        self.refine_brush_mask = None
        self.roi_editing = False
        self.run_state = RunState.IDLE
        self.progress = 0.0
        self.progress_message = ""
        self.elapsed_seconds = 0.0
        self.image_files = list(files)
        self.current_frame = 0
        if had_results:
            self.results_changed.emit()
            self.run_state_changed.emit(RunState.IDLE)
        if had_roi:
            self.roi_changed.emit()
        self.images_changed.emit()

    def set_current_frame(self, idx: int) -> None:
        if not self.image_files:
            return
        self.current_frame = max(0, min(idx, len(self.image_files) - 1))
        self.current_frame_changed.emit(self.current_frame)

    def set_roi_mask(self, mask: NDArray[np.bool_] | None) -> None:
        """Set or clear the frame-0 ROI mask (backward-compatible entry point)."""
        if mask is None:
            self.per_frame_rois.pop(0, None)
        else:
            self.per_frame_rois[0] = mask
        self.roi_changed.emit()

    def set_frame_roi(
        self, frame: int, mask: NDArray[np.bool_] | None
    ) -> None:
        """Set or clear the ROI mask for a specific frame."""
        if mask is None:
            self.per_frame_rois.pop(frame, None)
        else:
            self.per_frame_rois[frame] = mask
        self.roi_changed.emit()

    def set_refine_brush_mask(
        self, mask: NDArray[np.bool_] | None
    ) -> None:
        """Set or clear the user-painted brush refinement mask.

        The mask is always defined in frame-0 image coordinates; the
        pipeline auto-warps it to subsequent reference frames inside a
        single Run.
        """
        self.refine_brush_mask = mask
        self.roi_changed.emit()

    def get_effective_roi(
        self, frame: int, *, is_ref_frame: bool = False
    ) -> NDArray[np.bool_] | None:
        """Return the effective ROI mask for *frame*.

        Resolution order:
        1. Own mask in per_frame_rois[frame] — always wins.
        2. If *is_ref_frame* and frame != 0: inherit from per_frame_rois[0].
        3. Otherwise: None.
        """
        own = self.per_frame_rois.get(frame)
        if own is not None:
            return own
        if is_ref_frame:
            return self.per_frame_rois.get(0)
        return None

    def set_run_state(self, state: RunState) -> None:
        self.run_state = state
        self.run_state_changed.emit(state)

    def set_progress(self, fraction: float, message: str = "") -> None:
        self.progress = fraction
        self.progress_message = message
        self.progress_updated.emit(fraction, message)

    def set_results(self, results: PipelineResult) -> None:
        self.results = results
        self.results_changed.emit()

    def set_colormap(self, cmap: str) -> None:
        self.colormap = cmap
        self.display_changed.emit()

    def set_display_field(self, field_name: str) -> None:
        self.display_field = field_name
        self.display_changed.emit()

    def set_color_range(
        self, auto: bool, vmin: float = 0.0, vmax: float = 1.0
    ) -> None:
        self.color_auto = auto
        self.color_min = vmin
        self.color_max = vmax
        self.display_changed.emit()

    def set_show_mesh(self, show: bool) -> None:
        self.show_mesh = show
        self.display_changed.emit()

    def set_show_subset_window(self, show: bool) -> None:
        self.show_subset_window = show
        self.display_changed.emit()

    def set_overlay_alpha(self, alpha: float) -> None:
        self.overlay_alpha = max(0.0, min(1.0, float(alpha)))
        self.display_changed.emit()

    def set_physical_units(
        self,
        enabled: bool,
        pixel_size: float,
        frame_rate: float,
        pixel_unit: str = "mm",
    ) -> None:
        self.use_physical_units = bool(enabled)
        self.pixel_size = max(0.0, float(pixel_size))
        self.pixel_unit = str(pixel_unit)
        self.frame_rate = max(0.0, float(frame_rate))
        self.physical_units_changed.emit()
        self.display_changed.emit()

    def set_refine_inner(self, on: bool) -> None:
        self.refine_inner = on
        self.params_changed.emit()

    def set_refine_outer(self, on: bool) -> None:
        self.refine_outer = on
        self.params_changed.emit()

    def set_refinement_level(self, level: int) -> None:
        # Always clamp to the currently-valid range. UI will mirror this.
        max_level = self.compute_max_refinement_level()
        self.refinement_level = max(1, min(int(level), max_level))
        self.params_changed.emit()

    def compute_refinement_min_size(self) -> int:
        """Compute the actual min element size from level + subset_step.

        Floor at 2 px: this is the true mathematical lower bound from
        ``qrefine_r`` (edge midpoints divide by 2 — anything below 2 px
        produces fractional pixel coordinates and breaks ``mark_edge``).
        """
        return max(2, self.subset_step // (2 ** self.refinement_level))

    def compute_max_refinement_level(self) -> int:
        """Largest valid refinement level for the current params.

        Two constraints, both must hold:
          1. Integer/geometry: ``min_size = subset_step / 2^level >= 2``
             → ``level <= log2(subset_step / 2)``.
          2. Physical/statistical: node spacing should not be smaller than
             ``subset_size / 4`` (otherwise adjacent IC-GN subsets overlap
             by >75% and information becomes severely redundant)
             → ``level <= log2(subset_size / 4)``.

        Returns at least 1 so the dropdown always has Light available.
        """
        # subset_step is always a power of 2 (UI restricts to {4,8,16,32,64})
        step_limit = max(1, int(math.log2(max(2, self.subset_step) / 2)))
        # subset_size is the internal even value (display = subset_size + 1).
        # Use floor(log2(subset_size / 4)).
        size_limit = max(1, int(math.log2(max(4, self.subset_size) / 4)))
        return min(step_limit, size_limit)

    def set_param(self, name: str, value: object) -> None:
        if hasattr(self, name):
            setattr(self, name, value)
            self.params_changed.emit()
