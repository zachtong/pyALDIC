"""Central application state with Qt Signals.

All GUI panels subscribe to signals -- no direct panel-to-panel communication.
Workers emit signals back to the main thread; controllers update AppState.
"""

from __future__ import annotations

import enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from staq_dic.core.data_structures import PipelineResult


class RunState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


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

    def __init__(self) -> None:
        super().__init__()
        self._init_state()

    def _init_state(self) -> None:
        # Images
        self.image_folder: Path | None = None
        self.image_files: list[str] = []
        self.current_frame: int = 0
        # ROI
        self.roi_mask: NDArray[np.bool_] | None = None
        # Parameters
        self.subset_size: int = 40
        self.subset_step: int = 16
        self.search_range: int = 10
        self.tracking_mode: str = "incremental"
        # Computation
        self.run_state: RunState = RunState.IDLE
        self.progress: float = 0.0
        self.progress_message: str = ""
        self.elapsed_seconds: float = 0.0
        self.results: PipelineResult | None = None
        # Display
        self.display_field: str = "disp_u"
        self.color_auto: bool = True
        self.color_min: float = 0.0
        self.color_max: float = 1.0

    @classmethod
    def instance(cls) -> AppState:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        self._init_state()

    # --- Setters (emit signals) ---
    def set_image_files(self, files: list[str]) -> None:
        self.image_files = list(files)
        self.current_frame = 0
        self.images_changed.emit()

    def set_current_frame(self, idx: int) -> None:
        if not self.image_files:
            return
        self.current_frame = max(0, min(idx, len(self.image_files) - 1))
        self.current_frame_changed.emit(self.current_frame)

    def set_roi_mask(self, mask: NDArray[np.bool_] | None) -> None:
        self.roi_mask = mask
        self.roi_changed.emit()

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

    def set_param(self, name: str, value: object) -> None:
        if hasattr(self, name):
            setattr(self, name, value)
            self.params_changed.emit()
