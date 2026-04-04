"""Pipeline execution controller -- QThread worker with pause/stop."""

from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import FrameSchedule
from staq_dic.core.pipeline import run_aldic
from staq_dic.gui.app_state import AppState, RunState


class PipelineWorker(QThread):
    """Runs run_aldic() in a background thread."""

    progress = Signal(float, str)       # (fraction, message)
    log = Signal(str, str)              # (message, level)
    finished_result = Signal(object)    # PipelineResult or None on error

    def __init__(
        self,
        para,
        images: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> None:
        super().__init__()
        self._para = para
        self._images = [img.copy() for img in images]
        self._masks = [m.copy() for m in masks]

        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially

    def run(self) -> None:
        self.log.emit("Starting DIC analysis...", "info")
        t0 = time.perf_counter()
        try:
            result = run_aldic(
                para=self._para,
                images=self._images,
                masks=self._masks,
                progress_fn=self._on_progress,
                stop_fn=self._should_stop,
                compute_strain=False,
            )
            elapsed = time.perf_counter() - t0
            self.log.emit(f"Analysis complete in {elapsed:.1f}s", "info")
            self.finished_result.emit(result)
        except RuntimeError as e:
            if "abort" in str(e).lower() or "stop" in str(e).lower():
                self.log.emit("Analysis stopped by user.", "warn")
            else:
                self.log.emit(f"Error: {e}", "error")
            self.finished_result.emit(None)
        except Exception as e:
            self.log.emit(f"Unexpected error: {e}", "error")
            self.finished_result.emit(None)

    def _on_progress(self, fraction: float, message: str) -> None:
        self._pause_event.wait()  # block if paused
        self.progress.emit(fraction, message)

    def _should_stop(self) -> bool:
        return self._stop_requested

    def request_stop(self) -> None:
        self._stop_requested = True
        self._pause_event.set()  # unblock if paused

    def request_pause(self) -> None:
        self._pause_event.clear()

    def request_resume(self) -> None:
        self._pause_event.set()


class PipelineController:
    """Coordinates pipeline execution between UI and worker thread."""

    def __init__(self, state: AppState, image_ctrl) -> None:
        self._state = state
        self._image_ctrl = image_ctrl
        self._worker: PipelineWorker | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        """Build config from AppState and launch worker."""
        state = self._state
        if state.run_state == RunState.RUNNING:
            return

        if len(state.image_files) < 2:
            state.log_message.emit("Need at least 2 images.", "error")
            return
        if state.roi_mask is None:
            state.log_message.emit("Define a Region of Interest first.", "error")
            return

        # Build DICPara from GUI state
        schedule = FrameSchedule.from_mode(
            state.tracking_mode, len(state.image_files)
        )
        para = dicpara_default(
            winsize=state.subset_size,
            winstepsize=state.subset_step,
            size_of_fft_search_region=state.search_range,
            reference_mode=state.tracking_mode,
            frame_schedule=schedule,
        )

        # Load all images
        images = [
            self._image_ctrl.read_image(i)
            for i in range(len(state.image_files))
        ]
        mask = state.roi_mask.astype(np.float64)
        masks = [mask] * len(images)

        # Launch worker
        self._worker = PipelineWorker(para, images, masks)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(
            lambda msg, lvl: state.log_message.emit(msg, lvl)
        )
        self._worker.finished_result.connect(self._on_finished)

        self._start_time = time.perf_counter()
        state.set_run_state(RunState.RUNNING)
        self._worker.start()

    def stop(self) -> None:
        if self._worker:
            self._worker.request_stop()

    def pause(self) -> None:
        if self._worker and self._state.run_state == RunState.RUNNING:
            self._worker.request_pause()
            self._state.set_run_state(RunState.PAUSED)

    def resume(self) -> None:
        if self._worker and self._state.run_state == RunState.PAUSED:
            self._worker.request_resume()
            self._state.set_run_state(RunState.RUNNING)

    def _on_progress(self, fraction: float, message: str) -> None:
        self._state.set_progress(fraction, message)
        self._state.elapsed_seconds = time.perf_counter() - self._start_time

    def _on_finished(self, result) -> None:
        self._worker = None
        if result is not None:
            self._state.set_results(result)
            self._state.set_run_state(RunState.DONE)
        else:
            self._state.set_run_state(RunState.IDLE)
