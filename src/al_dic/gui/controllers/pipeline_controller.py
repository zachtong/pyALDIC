"""Pipeline execution controller -- QThread worker with pause/stop."""

from __future__ import annotations

import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import FrameSchedule, GridxyROIRange
from al_dic.core.pipeline import run_aldic
from al_dic.gui.app_state import AppState, RunState
from al_dic.mesh.refinement import RefinementPolicy, build_refinement_policy
from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedQualityError,
    SeedWarpFailure,
)


# User-facing messages for each SeedPropagationError subclass. Keyed by
# exception type so subclasses are matched exactly; the generic
# SeedPropagationError base is the fallback for any untagged subclass
# (e.g., 'window out of image bounds' — raised but not via a subclass).
_SEED_ERROR_MESSAGES: dict[type, tuple[str, str]] = {
    SeedNCCBelowThreshold: (
        "Starting Point match quality too low",
        "A Starting Point could not be reliably matched between the "
        "reference and the deformed frame. This usually means the local "
        "texture is too uniform for cross-correlation.\n\n"
        "Suggested actions:\n"
        "  - Move the point to a more textured area (visible speckle).\n"
        "  - Lower the NCC threshold (currently defaults to 0.70).\n"
        "  - Or switch back to FFT mode in ADVANCED > Initial Guess.",
    ),
    SeedICGNDiverged: (
        "Starting Point local solver did not converge",
        "A Starting Point's IC-GN iteration hit the max without "
        "converging. Common causes: seed too close to a mask edge, "
        "insufficient local texture, or an oversized subset window.\n\n"
        "Suggested actions:\n"
        "  - Move the Starting Point inward, away from mask boundaries.\n"
        "  - Try a smaller subset size.\n"
        "  - Or switch back to FFT mode.",
    ),
    SeedQualityError: (
        "Starting Point flagged as unreliable after run",
        "After the frame completed, a Starting Point was marked as a "
        "statistical outlier by the bad-point detector. Because the "
        "propagation uses this seed as a root, the whole frame's result "
        "is suspect.\n\n"
        "Suggested actions:\n"
        "  - Move the Starting Point to a cleaner textured region.\n"
        "  - Place multiple Starting Points for redundancy.",
    ),
    SeedWarpFailure: (
        "Could not carry Starting Points to the new reference",
        "When the reference frame changed, one or more Starting Points "
        "could not be mapped to the new reference. The material point "
        "may have moved outside the new ROI.\n\n"
        "Suggested actions:\n"
        "  - Place Starting Points well inside the ROI interior.\n"
        "  - Ensure the new reference's ROI covers the warped position.",
    ),
    MissingSeedForRegion: (
        "Region without a Starting Point",
        "At least one connected mask region has no Starting Point. "
        "The propagation cannot cross mask boundaries, so every region "
        "needs at least one seed.\n\n"
        "Suggested actions:\n"
        "  - Add a Starting Point in each yellow region before running.\n"
        "  - Or unify the mask so only one connected region exists.",
    ),
}


def _build_masks(
    per_frame_rois: dict[int, np.ndarray],
    n_frames: int,
    img_shape: tuple[int, int],
    ref_frame_set: set[int],
) -> list[np.ndarray]:
    """Build per-frame mask list for the pipeline.

    Strategy:
    - Frame with own per_frame_roi: use it
    - Ref frame without own ROI: inherit from frame 0
    - Non-ref frame without own ROI: all-ones (no mask)
    """
    mask0 = per_frame_rois.get(0)
    masks: list[np.ndarray] = []
    for i in range(n_frames):
        if i in per_frame_rois:
            masks.append(per_frame_rois[i].astype(np.float64))
        elif i in ref_frame_set and mask0 is not None:
            masks.append(mask0.astype(np.float64))
        else:
            masks.append(np.ones(img_shape, dtype=np.float64))
    return masks


def _confirm_incomplete_ref_rois(
    state: AppState, ref_frame_set: set[int]
) -> bool:
    """Warn the user when some reference frames lack a per-frame ROI mask.

    Returns ``True`` if the run should proceed (no missing refs, or user
    explicitly confirmed), ``False`` if the user declined.

    The check excludes frame 0 because its ROI is a hard prerequisite
    that PipelineController.start() validates earlier.
    """
    missing = sorted(
        f for f in ref_frame_set
        if f != 0 and f not in state.per_frame_rois
    )
    if not missing:
        return True

    # Convert to 1-based frame numbers for display (matches GUI labelling)
    display_frames = [f + 1 for f in missing]
    if len(display_frames) <= 10:
        frames_str = ", ".join(str(f) for f in display_frames)
    else:
        head = ", ".join(str(f) for f in display_frames[:10])
        frames_str = f"{head}, ... ({len(display_frames)} frames total)"

    text = (
        "<b>Incomplete Region of Interest coverage in incremental mode"
        "</b><br><br>"
        f"You are running incremental mode, but {len(display_frames)} "
        "reference frame(s) do not have their own Region of Interest "
        "mask:<br>"
        f"<i>Frame {frames_str}</i><br><br>"
        "These reference frames will <b>inherit frame 1's Region of "
        "Interest mask</b>, which is geometrically not strictly correct "
        "because the material moves between frames. Results near the "
        "Region of Interest boundary may be inaccurate.<br><br>"
        "For best accuracy, define a per-frame Region of Interest mask "
        "for each reference frame.<br><br>"
        "<b>Continue with the inherited masks?</b>"
    )

    parent = QApplication.activeWindow()
    button = QMessageBox.question(
        parent,
        "Incomplete Region of Interest Coverage",
        text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return button == QMessageBox.StandardButton.Yes


class PipelineWorker(QThread):
    """Runs run_aldic() in a background thread."""

    progress = Signal(float, str)       # (fraction, message)
    log = Signal(str, str)              # (message, level)
    finished_result = Signal(object)    # PipelineResult or None on error
    fatal_error = Signal(str, str)      # (title, message) — modal-worthy

    def __init__(
        self,
        para,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        refinement_policy: RefinementPolicy | None = None,
    ) -> None:
        super().__init__()
        self._para = para
        self._images = [img.copy() for img in images]
        self._masks = [m.copy() for m in masks]
        self._refinement_policy = refinement_policy

        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially

    def run(self) -> None:
        self.log.emit("Starting DIC analysis...", "info")
        self.log.emit(
            f"  images={len(self._images)}, "
            f"shape={self._images[0].shape}, "
            f"masks={len(self._masks)}, "
            f"mask_shape={self._masks[0].shape}",
            "info",
        )
        self.log.emit(
            f"  winsize={self._para.winsize}, "
            f"step={self._para.winstepsize}, "
            f"search={self._para.size_of_fft_search_region}, "
            f"mode={self._para.reference_mode}",
            "info",
        )
        t0 = time.perf_counter()
        try:
            result = run_aldic(
                para=self._para,
                images=self._images,
                masks=self._masks,
                progress_fn=self._on_progress,
                stop_fn=self._should_stop,
                compute_strain=False,
                refinement_policy=self._refinement_policy,
            )
            elapsed = time.perf_counter() - t0
            self.log.emit(f"Analysis complete in {elapsed:.1f}s", "success")
            self.finished_result.emit(result)
        except RuntimeError as e:
            if "abort" in str(e).lower() or "stop" in str(e).lower():
                self.log.emit("Analysis stopped by user.", "warn")
            else:
                tb = traceback.format_exc()
                self.log.emit(f"RuntimeError: {e}", "error")
                self.log.emit(tb, "error")
                print(tb, flush=True)
                self.fatal_error.emit(
                    "DIC analysis failed",
                    f"The pipeline raised a runtime error:\n\n{e}\n\n"
                    "See the console log for the full traceback.",
                )
            self.finished_result.emit(None)
        except SeedPropagationError as e:
            # Starting-Points specific failure — show a tailored dialog
            # with suggested actions before falling through to the
            # generic error path. Exception subclass lookup matches the
            # most specific type first; unrecognised subclasses fall
            # back to the base-class message.
            tb = traceback.format_exc()
            self.log.emit(f"{type(e).__name__}: {e}", "error")
            self.log.emit(tb, "error")
            print(tb, flush=True)
            title, suggestions = _SEED_ERROR_MESSAGES.get(
                type(e),
                (
                    "Starting Points error",
                    "Seed propagation raised an error.\n\n"
                    "Suggested actions:\n"
                    "  - Re-check Starting Point positions.\n"
                    "  - See console log for details.",
                ),
            )
            self.fatal_error.emit(
                title, f"{suggestions}\n\nDetails: {e}",
            )
            self.finished_result.emit(None)
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {type(e).__name__}: {e}", "error")
            self.log.emit(tb, "error")
            print(tb, flush=True)
            self.fatal_error.emit(
                "DIC analysis failed",
                f"{type(e).__name__}: {e}\n\n"
                "See the console log for the full traceback.",
            )
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

    def _fatal(self, title: str, message: str, detail: str = "") -> None:
        """Log and raise a modal-worthy error.

        The message must be user-facing (no tracebacks). ``detail`` is an
        optional extra line sent only to the console log.
        """
        self._state.log_message.emit(message, "error")
        if detail:
            self._state.log_message.emit(detail, "error")
        self._state.fatal_error.emit(title, message)

    def start(self) -> None:
        """Build config from AppState and launch worker."""
        state = self._state
        if state.run_state == RunState.RUNNING:
            return

        # Bug B: ensure the previous worker has truly finished before we
        # drop our reference to it and create a new one. Without an
        # explicit wait(), Qt may destroy the previous QThread before
        # its native thread has exited, producing a hard crash:
        #   "QThread: Destroyed while thread is still running"
        # The previous run normally cleared self._worker in _on_finished,
        # but if the user spams Run quickly the wrapper QObject can
        # outlive the IDLE state transition for one event-loop tick.
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.request_stop()
                self._worker.wait(5000)
            self._worker = None

        if len(state.image_files) < 2:
            self._fatal(
                "Cannot start analysis",
                "Need at least 2 images. Drop an image folder into the "
                "Images panel before running DIC."
            )
            return
        if state.roi_mask is None:
            self._fatal(
                "Cannot start analysis",
                "Define a Region of Interest first. Use the Region of "
                "Interest toolbar to draw or import one for frame 1."
            )
            return

        try:
            # Exit ROI editing mode when starting DIC
            state.roi_editing = False
            state.display_changed.emit()

            state.log_message.emit(self.tr("Building pipeline configuration..."), "info")

            # Build DICPara from GUI state
            n_images = len(state.image_files)
            solver_label = "AL-DIC" if state.use_admm else "Local DIC"
            admm_info = f", admm_iter={state.admm_max_iter}" if state.use_admm else ""
            state.log_message.emit(
                f"  {n_images} images, subset={state.subset_size}, "
                f"step={state.subset_step}, search={state.search_range}, "
                f"mode={state.tracking_mode}, solver={solver_label}{admm_info}",
                "info",
            )

            # Build FrameSchedule from current settings.
            # Only build an explicit schedule for incremental mode;
            # accumulative is handled by reference_mode alone (no schedule needed).
            schedule = None
            if state.tracking_mode == "incremental":
                if state.inc_ref_mode == "every_frame":
                    schedule = FrameSchedule.from_mode("incremental", n_images)
                elif state.inc_ref_mode == "every_n":
                    schedule = FrameSchedule.from_every_n(
                        state.inc_ref_interval, n_images
                    )
                elif state.inc_ref_mode == "custom":
                    schedule = FrameSchedule.from_custom(
                        state.inc_custom_refs, n_images
                    )
                else:
                    schedule = FrameSchedule.from_mode("incremental", n_images)

            # Validate frame 0 ROI exists
            roi_mask_0 = state.per_frame_rois.get(0)
            if roi_mask_0 is None:
                self._fatal(
                    "Cannot start analysis",
                    "Frame 1 has no Region of Interest. Draw one on the "
                    "first frame before running DIC."
                )
                return

            # Compute gridxy_roi_range from ROI mask bounding box
            roi_mask = roi_mask_0
            rows, cols = np.where(roi_mask)
            if len(rows) == 0:
                self._fatal(
                    "Cannot start analysis",
                    "The Region of Interest for frame 1 is empty "
                    "(no pixels selected). Redraw it or Clear + retry."
                )
                return
            roi_range = GridxyROIRange(
                gridx=(int(cols.min()), int(cols.max())),
                gridy=(int(rows.min()), int(rows.max())),
            )

            # Validate ROI is large enough for the configured mesh
            roi_width = roi_range.gridx[1] - roi_range.gridx[0]
            roi_height = roi_range.gridy[1] - roi_range.gridy[0]
            min_dim = 2 * state.subset_step
            if roi_width < min_dim or roi_height < min_dim:
                self._fatal(
                    "Region of Interest too small",
                    f"The Region of Interest is "
                    f"{roi_width}\u00d7{roi_height} px, but the current "
                    f"Subset Step ({state.subset_step}) requires at least "
                    f"{min_dim}\u00d7{min_dim} px. Either enlarge the "
                    f"Region of Interest or reduce Subset Step."
                )
                return

            state.log_message.emit(
                f"  ROI range: x=[{roi_range.gridx[0]}, {roi_range.gridx[1]}], "
                f"y=[{roi_range.gridy[0]}, {roi_range.gridy[1]}]",
                "info",
            )

            # winsize_min must be <= winstepsize; auto-clamp for small steps
            winsize_min = min(8, state.subset_step)

            # Map GUI init_guess_mode to DICPara fields
            seed_set_for_para = None
            if state.init_guess_mode == "fft_every":
                dic_init_guess = "fft"
                dic_fft_reset = 0
            elif state.init_guess_mode == "fft_reset_n":
                dic_init_guess = "previous"
                dic_fft_reset = max(1, state.fft_reset_interval)
            elif state.init_guess_mode == "seed_propagation":
                dic_init_guess = "seed_propagation"
                dic_fft_reset = 0
                if not state.seeds:
                    state.fatal_error.emit(
                        "Starting Points required",
                        "Place at least one Starting Point on the canvas "
                        "before running (select the 'Starting Points' mode "
                        "in ADVANCED → Initial Guess, then click 'Place "
                        "Starting Points').",
                    )
                    return
                # Count regions that are unseeded. Same logic as
                # SeedController.all_regions_seeded, but computed
                # pipeline-side as a defence-in-depth check independent
                # of any UI state that may have skipped the gate.
                seeded_regions = {s.region_id for s in state.seeds}
                unseeded = [
                    s.region_id
                    for s in state.seeds
                    if s.region_id < 0
                ]
                all_regions = set(seeded_regions)
                # If seed records reference region ids outside what we
                # just collected (e.g. negative), drop the run.
                if unseeded:
                    state.fatal_error.emit(
                        "Invalid seed region ids",
                        "One or more Starting Points reference an invalid "
                        "region. Re-place the points in the ROI.",
                    )
                    return
                from al_dic.solver.seed_propagation import Seed, SeedSet
                seed_set_for_para = SeedSet(
                    seeds=tuple(
                        Seed(node_idx=s.node_idx, region_id=s.region_id)
                        for s in state.seeds
                    ),
                    ncc_threshold=state.seed_ncc_threshold,
                )
            else:  # "previous"
                dic_init_guess = "previous"
                dic_fft_reset = 0

            para = dicpara_default(
                winsize=state.subset_size,
                winstepsize=state.subset_step,
                winsize_min=winsize_min,
                size_of_fft_search_region=state.search_range,
                reference_mode=state.tracking_mode,
                frame_schedule=schedule,
                gridxy_roi_range=roi_range,
                init_guess_mode=dic_init_guess,
                fft_reset_interval=dic_fft_reset,
                fft_auto_expand_search=state.fft_auto_expand,
                use_global_step=state.use_admm,
                admm_max_iter=state.admm_max_iter,
                seed_set=seed_set_for_para,
            )

            # Load all images
            state.log_message.emit(self.tr("Loading images..."), "info")
            images = [
                self._image_ctrl.read_image(i)
                for i in range(n_images)
            ]
            state.log_message.emit(
                f"  Loaded {len(images)} images, shape={images[0].shape}",
                "info",
            )

            # Build per-frame mask list
            mask_pixels = int(np.sum(roi_mask_0))
            state.log_message.emit(
                f"  ROI mask: {roi_mask_0.shape}, "
                f"{mask_pixels} pixels "
                f"({100*mask_pixels/roi_mask_0.size:.1f}%)",
                "info",
            )

            ref_frame_set = schedule.ref_frame_set if schedule is not None else {0}

            # In incremental mode every reference frame ideally has its own
            # ROI mask: a mask defined on frame 0 is geometrically wrong for
            # frame K if the material has moved between them.  When some
            # ref frames are missing a mask we currently fall back to
            # frame 0's mask (see _build_masks); ask the user to confirm
            # before doing so.
            if (
                state.tracking_mode == "incremental"
                and not _confirm_incomplete_ref_rois(
                    state, ref_frame_set
                )
            ):
                state.log_message.emit(
                    "Run cancelled: define per-frame Regions of Interest "
                    "for the missing reference frames or accept the "
                    "inherited frame-1 mask in the next run.",
                    "warn",
                )
                return

            masks = _build_masks(
                state.per_frame_rois, n_images,
                roi_mask_0.shape, ref_frame_set,
            )

            n_custom = sum(
                1 for i in range(n_images)
                if i in state.per_frame_rois and i != 0
            )
            if n_custom > 0:
                state.log_message.emit(
                    f"  {n_custom} frame(s) with custom ROI masks",
                    "info",
                )

            # Build refinement policy from GUI state. The factory returns
            # None when no refinement levers are active, which makes
            # run_aldic skip all refinement (uniform mesh fast path).
            brush_mask_f64 = (
                state.refine_brush_mask.astype(np.float64)
                if state.refine_brush_mask is not None
                else None
            )
            refinement_policy = build_refinement_policy(
                refine_inner_boundary=state.refine_inner,
                refine_outer_boundary=state.refine_outer,
                refinement_mask=brush_mask_f64,
                min_element_size=state.compute_refinement_min_size(),
                # half_win is the IC-GN window half-width in pixels.
                # state.subset_size already stores the even internal value
                # (display 41 -> internal 40), and half_win = winsize / 2.
                half_win=max(1, state.subset_size // 2),
            )
            if refinement_policy is not None:
                bits = []
                if state.refine_inner:
                    bits.append("inner")
                if state.refine_outer:
                    bits.append("outer")
                if brush_mask_f64 is not None:
                    bits.append("brush")
                state.log_message.emit(
                    f"  Refinement: {'+'.join(bits)} "
                    f"(level={state.refinement_level}, "
                    f"min_size={state.compute_refinement_min_size()} px)",
                    "info",
                )

            # All validation passed -- this run is committed.  Drop the
            # previous run's outputs *before* the new worker starts so the
            # canvas does not render a hybrid view of the OLD field/mesh
            # clipped by the NEW ROI while the new worker is computing.
            # Doing it here (rather than at the top of start()) preserves
            # old results when validation fails earlier in this method.
            #
            # ``results_changed`` triggers VizController.clear_all() (wired
            # in app.py) and CanvasArea._refresh_overlay/_refresh_mesh_overlay,
            # which both fall back to the "no results" preview path.
            had_results = state.results is not None
            state.results = None
            state.deformed_masks = None
            # NOTE: do NOT reset state.show_deformed here. The user's
            # "show on deformed frame" preference must persist across
            # re-runs -- otherwise the checkbox stays checked but the
            # display silently reverts to reference-frame coordinates.
            # The new run's deformed_masks will be populated by
            # _on_finished -> set_results, at which point the canvas
            # picks the right coordinates based on this preserved flag.
            if had_results:
                state.results_changed.emit()

            # Launch worker
            self._worker = PipelineWorker(
                para, images, masks, refinement_policy=refinement_policy
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.log.connect(
                lambda msg, lvl: state.log_message.emit(msg, lvl)
            )
            # Defensive: some test stubs omit fatal_error. hasattr
            # keeps the pipeline bootable when a stub is incomplete
            # rather than triggering the very dialog that reports
            # 'worker has no attribute fatal_error' on the user's
            # desktop during regression runs.
            if hasattr(self._worker, "fatal_error"):
                self._worker.fatal_error.connect(state.fatal_error)
            self._worker.finished_result.connect(self._on_finished)

            # Reset progress from any previous run
            state.set_progress(0.0, "Starting...")
            state.elapsed_seconds = 0.0

            self._start_time = time.perf_counter()
            state.set_run_state(RunState.RUNNING)
            self._worker.start()

        except Exception as e:
            tb = traceback.format_exc()
            self._fatal(
                "Failed to start analysis",
                f"{type(e).__name__}: {e}\n\n"
                "See the console log for the full traceback.",
                detail=tb,
            )
            print(tb, flush=True)

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
        # Enforce monotonic progress (pipeline sections can have overlapping
        # fraction ranges across frames, causing backwards jumps).
        fraction = max(fraction, self._state.progress)
        self._state.set_progress(fraction, message)
        self._state.elapsed_seconds = time.perf_counter() - self._start_time

    def _on_finished(self, result) -> None:
        self._worker = None
        try:
            if result is not None:
                self._state.log_message.emit(
                    f"Results received: {len(result.result_disp)} frames",
                    "success",
                )
                self._state.set_results(result)
                self._state.set_run_state(RunState.DONE)
                # Auto-navigate to frame 1 (first result frame) so user
                # sees the displacement overlay immediately.
                self._state.set_current_frame(1)
            else:
                self._state.set_run_state(RunState.IDLE)
        except Exception as e:
            tb = traceback.format_exc()
            self._fatal(
                "Failed to process results",
                f"{type(e).__name__}: {e}\n\n"
                "See the console log for the full traceback.",
                detail=tb,
            )
            print(tb, flush=True)
            self._state.set_run_state(RunState.IDLE)
