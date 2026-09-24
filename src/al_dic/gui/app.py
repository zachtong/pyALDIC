"""AL-DIC GUI application entry point."""

import logging
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QHBoxLayout,
    QProgressDialog,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.controllers.pipeline_controller import PipelineController
from al_dic.gui.controllers.roi_controller import ROIController
from al_dic.gui.controllers.viz_controller import VizController
from al_dic.gui.icons import icon_app
from al_dic.gui.panels.canvas_area import CanvasArea
from al_dic.gui.panels.left_sidebar import LeftSidebar
from al_dic.gui.panels.right_sidebar import RightSidebar
from al_dic.gui.theme import COLORS, build_stylesheet


from al_dic.gui.window_chrome import enable_dark_title_bar


class MainWindow(QMainWindow):
    """Three-column main window: left sidebar | canvas | right sidebar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyALDIC")
        self.setWindowIcon(icon_app())
        # Minimum size tracks the natural layout floor (left sidebar 320 +
        # canvas toolbar ~520 + right sidebar 280 = 1120) instead of being
        # padded above it, so the window can shrink onto smaller screens.
        self.setMinimumSize(1120, 760)
        enable_dark_title_bar(self)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # State and controllers
        self._state = AppState.instance()
        self._image_ctrl = ImageController(self._state)
        self._roi_ctrl: ROIController | None = None
        # Separate buffer for the brush refinement mask (frame 0 only).
        self._brush_ctrl: ROIController | None = None

        # Seed-propagation controller (created unconditionally; idle until
        # the user selects init_guess_mode='seed_propagation').
        from al_dic.gui.controllers.seed_controller import SeedController
        self._seed_ctrl = SeedController(self)

        # Left sidebar — image loading + ROI toolbar
        self._left_sidebar = LeftSidebar(self._image_ctrl)
        layout.addWidget(self._left_sidebar, stretch=0)

        # Visualization controller (two-level cache)
        self._viz_ctrl = VizController()

        # Center canvas
        self._canvas_area = CanvasArea(self._image_ctrl, viz_ctrl=self._viz_ctrl)
        layout.addWidget(self._canvas_area, stretch=1)

        # Pipeline controller
        self._pipeline_ctrl = PipelineController(self._state, self._image_ctrl)

        # Right sidebar -- run controls, progress, display, console
        self._right_sidebar = RightSidebar(self._pipeline_ctrl)
        layout.addWidget(self._right_sidebar, stretch=0)

        # Lazy strain post-processing window. Created on first request,
        # then reused as a singleton until MainWindow closes.
        self._strain_window = None
        self._right_sidebar.open_strain_window_requested.connect(
            self._on_open_strain_window
        )

        # Prompt user to open strain window when a pipeline run completes.
        self._state.run_state_changed.connect(self._on_run_state_changed)

        # Wire ROI toolbar signals
        roi_tb = self._left_sidebar.roi_toolbar
        roi_tb.draw_requested.connect(self._on_draw_requested)
        roi_tb.clear_requested.connect(self._on_roi_clear)
        roi_tb.import_requested.connect(self._on_roi_import)
        roi_tb.save_requested.connect(self._on_roi_save)
        roi_tb.invert_requested.connect(self._on_roi_invert)
        roi_tb.batch_import_requested.connect(self._on_batch_import)
        roi_tb.brush_requested.connect(self._on_brush_requested)
        roi_tb.brush_clear_requested.connect(self._on_brush_clear)
        # Live brush radius update — applies to whichever mode (paint or
        # erase) is currently active without re-opening the popup menu.
        roi_tb.brush_radius_changed.connect(
            self._canvas_area.canvas.set_brush_radius
        )

        # When canvas finishes drawing, deactivate toolbar highlight
        self._canvas_area.canvas.drawing_finished.connect(roi_tb.deactivate)

        # Starting-Points (seed propagation) wiring
        self._canvas_area.attach_seed_controller(self._seed_ctrl)
        init_guess = self._left_sidebar.init_guess_widget
        init_guess.set_seed_controller(self._seed_ctrl)
        init_guess.request_place_seeds.connect(self._on_request_place_seeds)
        init_guess.request_auto_place_seeds.connect(
            self._on_request_auto_place_seeds,
        )
        init_guess.request_clear_seeds.connect(
            self._on_request_clear_seeds,
        )
        # Any init-method action jumps the canvas back to frame-0 ROI
        # editing so the user sees a consistent 'setup' view.
        init_guess.init_mode_user_changed.connect(
            self._enter_frame0_setup_view,
        )
        self._right_sidebar.set_seed_controller(self._seed_ctrl)
        # Sync the 'Place Starting Points' button state with the canvas
        # tool: pressed + label changed while the tool is 'seed',
        # released otherwise (incl. after Esc on the canvas).
        self._canvas_area.canvas.tool_changed.connect(
            self._on_canvas_tool_changed_for_button,
        )
        # Auto-place Starting Points when the prerequisites are all met
        # (seed_propagation mode, >= 2 images, an ROI with regions, and
        # no user-placed seeds yet). Keeps the default mode usable
        # without forcing every user to learn the Place / Auto-place
        # workflow before they can click Run.
        self._state.roi_changed.connect(self._maybe_auto_place_seeds)
        self._state.params_changed.connect(self._maybe_auto_place_seeds)
        self._state.images_changed.connect(self._maybe_auto_place_seeds)

        # Per-frame ROI editing from image list
        self._left_sidebar._image_list.roi_edit_requested.connect(
            self._on_roi_edit_for_frame
        )
        self._left_sidebar._image_list.roi_import_for_frames.connect(
            self._on_roi_import_for_frames
        )

        # Initialize ROI controller when images are loaded
        self._state.images_changed.connect(self._init_roi_controller)

        # When the user navigates frames during ROI editing, reload the
        # stamping buffer so the next draw operation targets the new frame.
        self._state.current_frame_changed.connect(self._on_frame_changed_for_roi)

        # Defensive: if an external path (batch import, invert, clear)
        # mutates per_frame_rois[current_frame] while the user is in
        # editing mode, reload the stamping buffer so the next draw
        # stamp operates on the fresh mask.
        self._state.roi_changed.connect(self._on_roi_changed_reload)

        # Refresh cyan brush overlay whenever state.refine_brush_mask
        # changes (clear, restore after re-load, etc).
        self._state.roi_changed.connect(
            self._canvas_area.canvas.update_refine_overlay
        )
        # Also refresh on frame navigation so the brush overlay hides
        # when the user is not on frame 0 (the brush mask lives in
        # frame-0 coordinates and would otherwise bleed through later
        # frames).
        self._state.current_frame_changed.connect(
            lambda _idx: self._canvas_area.canvas.update_refine_overlay()
        )

        # Clear viz caches when results change (new pipeline run)
        self._state.results_changed.connect(self._viz_ctrl.clear_all)

        # Brush refinement is a pre-Run input that only makes sense on
        # frame 0 with no results.  Drop the active brush tool whenever
        # either condition flips so the user can't keep painting on a
        # stale or wrong-frame canvas.
        self._state.current_frame_changed.connect(
            self._drop_brush_tool_if_invalid
        )

        # File menu: session save / load
        self._build_menu_bar()

    # ------------------------------------------------------------------
    # Menu bar — session save / load
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        """Create the File menu with session save / load actions."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu(self.tr("&File"))

        open_session_action = QAction(self.tr("Open Session…"), self)
        open_session_action.setShortcut(QKeySequence.StandardKey.Open)
        open_session_action.triggered.connect(self._on_open_session)
        file_menu.addAction(open_session_action)

        save_session_action = QAction(self.tr("Save Session…"), self)
        save_session_action.setShortcut(QKeySequence.StandardKey.Save)
        save_session_action.triggered.connect(self._on_save_session)
        file_menu.addAction(save_session_action)

        from al_dic.gui import file_association
        if file_association.is_supported():
            file_menu.addSeparator()
            assoc_action = QAction(
                self.tr("Associate .aldic files with pyALDIC…"), self)
            assoc_action.setToolTip(self.tr(
                "Register .aldic so double-clicking a session file opens "
                "pyALDIC (current user only, no admin rights needed)."))
            assoc_action.triggered.connect(self._on_register_association)
            file_menu.addAction(assoc_action)

        file_menu.addSeparator()
        quit_action = QAction(self.tr("Quit"), self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Settings > Language submenu
        from al_dic.i18n import SUPPORTED_LANGUAGES, LanguageManager

        settings_menu = menu_bar.addMenu(self.tr("&Settings"))
        language_menu = settings_menu.addMenu(self.tr("Language"))
        current = LanguageManager.saved_preference()
        for code, display_name in SUPPORTED_LANGUAGES.items():
            act = QAction(display_name, self)
            act.setCheckable(True)
            act.setChecked(code == current)
            act.triggered.connect(
                lambda _checked=False, c=code: self._on_language_selected(c))
            language_menu.addAction(act)

    def _on_language_selected(self, lang_code: str) -> None:
        """Persist the chosen language and prompt for a restart.

        Phase-1 strategy: write the preference, show an info dialog,
        wait for the next app launch. A live runtime switch will arrive
        once every widget implements retranslate_ui() + changeEvent().
        """
        from al_dic.i18n import LanguageManager, SUPPORTED_LANGUAGES

        app = QApplication.instance()
        lang_mgr: LanguageManager | None = getattr(
            app, "_pyaldic_lang_mgr", None)
        if lang_mgr is not None:
            lang_mgr.load(lang_code)

        from al_dic.i18n import tr_args

        QMessageBox.information(
            self,
            self.tr("Language changed"),
            tr_args(
                self.tr(
                    "Language set to %1. Please restart pyALDIC for all "
                    "widgets to pick up the new language."),
                SUPPORTED_LANGUAGES.get(lang_code, lang_code),
            ),
        )

    def _on_save_session(self) -> None:
        """Save-session dialog: write config + Regions of Interest (+ results)."""
        from al_dic.gui.session import save_session
        from al_dic.gui.session_worker import format_bytes
        from al_dic.io.session_serialize import estimated_nbytes

        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Session"),
            "",
            self.tr("pyALDIC Session") + " (*.aldic);;"
            + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        if not path.endswith(".aldic"):
            path = path + ".aldic"

        # Offer to include the (potentially large) computed results.
        include_results = False
        if self._state.results is not None:
            try:
                est = format_bytes(estimated_nbytes(self._state.results))
            except Exception:
                est = self.tr("large")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(self.tr("Include Results?"))
            box.setText(self.tr(
                "Include the computed results in this session?"))
            from al_dic.i18n import tr_args
            box.setInformativeText(tr_args(self.tr(
                "Including results (about %1 uncompressed) lets you reopen the "
                "session without recomputing. Choose No to save a small "
                "configuration-only file for sharing."), est))
            box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Yes)
            choice = box.exec()
            if choice == QMessageBox.StandardButton.Cancel:
                return
            include_results = choice == QMessageBox.StandardButton.Yes

        self._run_session_op(
            self.tr("Saving Session"),
            lambda cb: save_session(Path(path), self._state,
                                    include_results=include_results, progress=cb),
            on_done=lambda _r: self._state.log_message.emit(
                f"Session saved to {path}", "success"),
            fail_title=self.tr("Save Session Failed"),
        )

    def _on_open_session(self) -> None:
        """Open-session dialog: parse the bundle, then apply to AppState."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Session"),
            "",
            self.tr("pyALDIC Session") + " (*.aldic *.aldic.json);;"
            + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        self.open_session_path(path)

    def open_session_path(self, path: str) -> None:
        """Load and apply a session from *path* (used by the menu and by the
        command line / file association)."""
        from al_dic.gui.session import load_session

        self._run_session_op(
            self.tr("Loading Session"),
            lambda cb: load_session(Path(path), progress=cb),
            on_done=lambda session: self._apply_loaded_session(session, path),
            fail_title=self.tr("Open Session Failed"),
        )

    def _prompt_locate_images(self, missing_folder) -> str | None:
        """Ask the user to locate a session's images when the saved folder is
        gone (e.g. the project was moved).  Returns the chosen folder or None."""
        from al_dic.i18n import tr_args
        QMessageBox.information(
            self,
            self.tr("Locate Session Images"),
            tr_args(
                self.tr(
                    "The image folder saved with this session was not found:\n"
                    "%1\n\nResults were restored. To show the background "
                    "images, select the folder that now contains them."
                ),
                str(missing_folder),
            ),
        )
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select Image Folder"), "",
        )
        return folder or None

    def _apply_loaded_session(self, session, path: str) -> None:
        """Apply a parsed session on the main thread (GUI-touching)."""
        from al_dic.gui.session import SessionError, apply_session

        try:
            apply_session(
                session, self._state, self._image_ctrl,
                session_path=path,
                locate_folder_cb=self._prompt_locate_images,
            )
        except SessionError as e:
            QMessageBox.critical(self, self.tr("Open Session Failed"), str(e))
            return
        # Mirror the restored ROI into the editing buffer.  apply_session
        # assigns per_frame_rois directly and the signals it emits reach the
        # buffer only while roi_editing is on, which it is not right after a
        # load -- leaving the buffer empty while AppState holds the ROI.  Save
        # would then report an empty mask, and an edit would overwrite the
        # restored ROI with whatever was drawn on the empty buffer.
        self._load_roi_buffer_for_current_frame()
        n_roi = len(session.per_frame_rois)
        had_results = session.results is not None
        msg = f"Session loaded from {path} ({n_roi} Region(s) of Interest"
        msg += ", results restored)" if had_results else ")"
        self._state.log_message.emit(msg, "success")

    def _run_session_op(self, title: str, fn, on_done, fail_title: str) -> None:
        """Run *fn(progress_cb)* on a worker thread behind a progress dialog."""
        from al_dic.gui.session_worker import SessionOpWorker

        dlg = QProgressDialog(title + "…", "", 0, 100, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)          # save/load cannot be safely aborted
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setValue(0)

        worker = SessionOpWorker(fn, self)
        self._session_worker = worker      # keep a reference alive

        def _progress(frac: float, message: str) -> None:
            dlg.setValue(max(0, min(100, int(frac * 100))))
            dlg.setLabelText(message)

        def _finish() -> None:
            dlg.reset()
            worker.deleteLater()
            self._session_worker = None

        def _ok(result) -> None:
            _finish()
            on_done(result)

        def _err(message: str) -> None:
            _finish()
            QMessageBox.critical(self, fail_title, message)

        worker.progress.connect(_progress)
        worker.done.connect(_ok)
        worker.failed.connect(_err)
        dlg.show()
        worker.start()

    def _on_register_association(self) -> None:
        """Register the .aldic file type (Windows, current user)."""
        from al_dic.gui import file_association

        try:
            file_association.register_association()
        except Exception as e:  # noqa: BLE001 - report any failure to the user
            QMessageBox.warning(
                self, self.tr("File Association Failed"),
                self.tr("Could not register .aldic files: ") + str(e))
            return
        QMessageBox.information(
            self, self.tr("File Association"),
            self.tr("Done. Double-clicking a .aldic file will now open "
                    "pyALDIC and restore that session."))

    # ------------------------------------------------------------------

    def _init_roi_controller(self) -> None:
        """Create ROI + brush controllers matching the loaded image dimensions."""
        if not self._state.image_files:
            return
        try:
            rgb = self._image_ctrl.read_image_rgb(0)
            h, w = rgb.shape[:2]
            self._roi_ctrl = ROIController((h, w))
            self._canvas_area.canvas.set_roi_controller(self._roi_ctrl)
            # Sibling buffer for the brush refinement mask.  Restore from
            # AppState if a previous Run left a brush mask in place.
            self._brush_ctrl = ROIController((h, w))
            if self._state.refine_brush_mask is not None:
                if self._state.refine_brush_mask.shape == (h, w):
                    self._brush_ctrl.mask = self._state.refine_brush_mask.copy()
                else:
                    # Image dims changed — drop the stale brush mask
                    self._state.set_refine_brush_mask(None)
            self._canvas_area.canvas.set_brush_controller(self._brush_ctrl)
            self._canvas_area.canvas.update_refine_overlay()
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _enter_roi_editing(self) -> None:
        """Switch to ROI editing mode — show ROI overlay, hide field overlay."""
        self._state.roi_editing = True
        self._state.display_changed.emit()
        # Auto-scroll the left sidebar to the ROI section so the
        # controls are immediately visible, no matter how far the user
        # had scrolled through the settings.
        self._left_sidebar.focus_roi_section()

    def _on_roi_edit_for_frame(self, frame: int) -> None:
        """Enter ROI editing mode for a specific frame.

        Single source of truth: ``current_frame`` *is* the editing frame.
        Switching frames here keeps the image list selection, progress bar,
        and canvas display all in sync with what the user is editing.
        """
        state = self._state
        if not state.image_files:
            return
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return
        state.set_current_frame(frame)
        self._load_roi_buffer_for_current_frame()
        state.roi_editing = True
        state.display_changed.emit()

    def _load_roi_buffer_for_current_frame(self) -> None:
        """Mirror per_frame_rois[current_frame] into the ROI controller buffer.

        Called on entry to ROI editing and whenever current_frame changes
        during editing.
        """
        if self._roi_ctrl is None:
            return
        state = self._state
        existing = state.per_frame_rois.get(state.current_frame)
        if existing is not None:
            self._roi_ctrl.mask = existing.copy()
        else:
            self._roi_ctrl.clear()

    def _on_frame_changed_for_roi(self, _frame: int) -> None:
        """Reload the ROI controller buffer for the new current frame.

        Only matters during ROI editing -- the canvas always paints the
        overlay from per_frame_rois[current_frame] regardless.  But the
        in-memory buffer must mirror the new frame so the next stamp
        operation (draw / invert / clear) starts from the right base.
        """
        if self._state.roi_editing:
            self._load_roi_buffer_for_current_frame()

    def _on_roi_changed_reload(self) -> None:
        """Refresh the working ROI buffer when per_frame_rois mutates.

        Only runs during ROI editing -- outside editing mode the buffer
        is irrelevant and must be left alone (otherwise batch import
        paths that the user isn't actively editing would stomp the
        in-memory buffer).
        """
        if self._state.roi_editing and self._roi_ctrl is not None:
            self._load_roi_buffer_for_current_frame()

    def _drop_brush_tool_if_invalid(self, *_args) -> None:
        """Reset the canvas to ``select`` if brush is no longer paintable.

        The only condition that invalidates an active brush session is
        navigating away from frame 0 -- brush coordinates only make sense
        on the reference frame.  Painting after a completed Run is allowed
        (consistent with ROI / mesh-parameter edits).

        We emit ``drawing_finished`` so the toolbar Refine button
        highlight clears; otherwise the cursor stays in cross-hair mode.
        """
        canvas = self._canvas_area.canvas
        if canvas._current_tool != "brush":
            return
        if self._state.current_frame == 0:
            return
        canvas.set_tool("select")
        canvas.drawing_finished.emit()

    def _enter_frame0_setup_view(self) -> None:
        """Jump the canvas to frame 0 + ROI editing (= Edit button).

        Invoked for any 'setup' action: init-guess method change,
        Place Starting Points, Auto-place. Makes the sidebar action
        and the canvas stay in sync without the user having to
        remember to click Edit on frame 0 first.
        """
        state = self._state
        if not state.image_files:
            return
        # Jump to frame 0 so the user sees the reference frame on
        # which ROI + seeds live.
        if state.current_frame != 0:
            state.set_current_frame(0)
        # Load the frame-0 ROI buffer and flip roi_editing = True —
        # mirrors the _on_draw_requested path without toggling a
        # specific drawing tool (rect/polygon/etc).
        self._load_roi_buffer_for_current_frame()
        if not state.roi_editing:
            self._enter_roi_editing()

    def _on_request_place_seeds(self) -> None:
        """User clicked 'Place Starting Points' in the init-guess panel."""
        canvas = self._canvas_area.canvas
        if canvas._current_tool == "seed":
            # Toggle off — return to pan
            canvas.set_tool("pan")
            return
        # Take user to frame 0 editing for a consistent setup view.
        self._enter_frame0_setup_view()
        canvas.set_tool("seed")

    def _on_canvas_tool_changed_for_button(self, tool: str) -> None:
        """Mirror canvas tool state on the 'Place Starting Points' button."""
        init_guess = self._left_sidebar.init_guess_widget
        init_guess.set_seed_mode_active(tool == "seed")

    def _maybe_auto_place_seeds(self) -> None:
        """Auto-place Starting Points in any region that doesn't have one.

        Fires on roi_changed / params_changed / images_changed. Fills
        only unseeded regions so:
          - a manual seed in region A is preserved when the user adds
            region B later,
          - multi-region ROI edits progressively populate all regions,
          - a user who right-clicks to clear a region gets that region
            re-filled on the next ROI edit (this is arguably friendly;
            if not, they can switch mode to opt out).

        Conditions:
          - init_guess_mode == 'seed_propagation'
          - at least 2 images loaded
          - SeedController has at least one region under current mask
          - at least one of those regions is unseeded
        """
        state = self._state
        if state.init_guess_mode != "seed_propagation":
            return
        if len(state.image_files) < 2:
            return
        status = self._seed_ctrl.regions_status()
        if not status:
            return
        if all(has for _, has, _ in status):
            return  # every region already seeded — nothing to do
        try:
            ref_img = self._image_ctrl.read_image(0)
            def_img = self._image_ctrl.read_image(1)
        except Exception:
            return
        try:
            self._seed_ctrl.auto_place_seeds(
                ref_img, def_img,
                winsize=state.subset_size,
                search_radius=state.search_range,
                only_unseeded_regions=True,
            )
        except Exception:
            return

    def _on_request_auto_place_seeds(self) -> None:
        """User clicked 'Auto-place' in the init-guess seed sub-panel."""
        state = self._state
        n_images = len(state.image_files)
        if n_images < 2:
            state.fatal_error.emit(
                "Auto-place needs two frames",
                "Load at least two images before auto-placing Starting "
                "Points — the algorithm needs the reference plus one "
                "deformed frame to compute cross-correlations.",
            )
            return
        try:
            ref_img = self._image_ctrl.read_image(0)
            def_img = self._image_ctrl.read_image(1)
        except Exception as e:
            state.fatal_error.emit(
                "Could not read images for auto-place",
                f"{type(e).__name__}: {e}",
            )
            return
        placed = self._seed_ctrl.auto_place_seeds(
            ref_img, def_img,
            winsize=state.subset_size,
            search_radius=state.search_range,
            only_unseeded_regions=True,
        )
        # Show the setup view regardless of whether anything was placed —
        # gives the user consistent feedback about what the action did.
        self._enter_frame0_setup_view()
        if placed == 0:
            state.log_message.emit(
                "Auto-place: every region already has a Starting Point.",
                "info",
            )
            return
        state.log_message.emit(
            f"Auto-placed {placed} Starting Point(s) in unseeded regions.",
            "info",
        )

    def _on_request_clear_seeds(self) -> None:
        """User clicked 'Clear' in the init-guess seed sub-panel."""
        state = self._state
        n_before = len(state.seeds)
        if n_before == 0:
            state.log_message.emit(
                "No Starting Points to clear.", "info",
            )
            return
        self._seed_ctrl.clear_seeds()
        self._enter_frame0_setup_view()
        state.log_message.emit(
            f"Cleared {n_before} Starting Point(s).", "info",
        )

    def _on_draw_requested(self, shape: str, mode: str) -> None:
        """Activate one-shot drawing mode on the canvas.

        The toolbar Draw button always edits the *currently displayed* frame.
        To edit a different frame, navigate there first (image list click,
        arrow keys, or per-frame Edit button).
        """
        self._load_roi_buffer_for_current_frame()
        self._enter_roi_editing()
        canvas = self._canvas_area.canvas
        canvas.set_drawing_mode(mode)
        canvas.set_tool(shape)

    def _on_roi_clear(self) -> None:
        """Clear the ROI mask for the currently edited frame.

        Brush refinement only lives on frame 0; if the user clears
        frame 0's ROI, the painted brush region becomes meaningless
        (it was scoped to the now-deleted ROI), so cascade-delete it.
        """
        if self._roi_ctrl is not None:
            self._enter_roi_editing()
            self._roi_ctrl.clear()
            self._canvas_area.canvas.update_roi_overlay()
            state = self._state
            state.set_frame_roi(state.current_frame, None)
            if state.current_frame == 0:
                self._cascade_clear_brush()

    def _on_roi_import(self, path: str) -> None:
        """Import a mask file into the ROI controller for the current editing frame."""
        if self._roi_ctrl is None:
            return
        self._enter_roi_editing()
        try:
            self._roi_ctrl.import_mask(path)
            self._canvas_area.canvas.update_roi_overlay()
            state = self._state
            state.set_frame_roi(
                state.current_frame, self._roi_ctrl.mask.copy()
            )
        except IOError:
            pass

    def _no_own_roi_hint(self) -> str | None:
        """Explain an empty ROI buffer when the frame itself is the reason.

        The ROI toolbar acts on the current frame's *own* mask, while the
        solver falls back to frame 1's.  Reopening a session lands on
        whichever frame was last displayed, so a user who left off browsing
        results finds the toolbar dead on a frame that never had its own ROI
        -- "mask is empty" then points at the mask instead of the frame.

        Returns a ready-to-log explanation in that case, or None when the
        Region of Interest is genuinely undefined everywhere.
        """
        state = self._state
        if state.current_frame == 0:
            return None
        if state.per_frame_rois.get(state.current_frame) is not None:
            return None
        if state.per_frame_rois.get(0) is None:
            return None
        from al_dic.i18n import tr_args
        return tr_args(
            self.tr(
                "Frame %1 has no Region of Interest of its own — frame 1's is "
                "used for computation. Switch to frame 1 to edit it, or import "
                "a mask to give this frame its own."
            ),
            state.current_frame + 1,
        )

    def _on_roi_save(self) -> None:
        """Save the current ROI mask to a PNG file."""
        if self._roi_ctrl is None:
            self._state.log_message.emit(
                self.tr("No Region of Interest to save — load images first."),
                "warn",
            )
            return
        if not self._roi_ctrl.mask.any():
            self._state.log_message.emit(
                self._no_own_roi_hint()
                or self.tr("Region of Interest mask is empty."),
                "warn",
            )
            return
        self._enter_roi_editing()
        # An anchored suggestion, not a bare name: a relative path is resolved
        # against the working directory, which for a frozen app launched from a
        # shortcut or a file association is not the user's project folder.
        folder = self._state.image_folder
        suggested = str(Path(folder) / "roi_mask.png") if folder else "roi_mask.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Region of Interest Mask"),
            suggested,
            self.tr("PNG Images") + " (*.png);;"
            + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        try:
            self._roi_ctrl.save_mask(path)
            from al_dic.i18n import tr_args
            self._state.log_message.emit(
                tr_args(self.tr("Mask saved to %1"), path), "success"
            )
        except IOError as e:
            self._state.log_message.emit(f"Save failed: {e}", "error")

    def _on_roi_invert(self) -> None:
        """Invert the ROI mask for the currently edited frame."""
        if self._roi_ctrl is None:
            self._state.log_message.emit(
                self.tr("No Region of Interest to invert — load images first."),
                "warn"
            )
            return
        # On a frame with no ROI of its own the buffer is empty, and inverting
        # it would hand that frame the whole image as a brand-new per-frame
        # Region of Interest -- silently changing what gets correlated.
        hint = self._no_own_roi_hint()
        if hint is not None:
            self._state.log_message.emit(hint, "warn")
            return
        self._enter_roi_editing()
        self._roi_ctrl.invert()
        self._canvas_area.canvas.update_roi_overlay()
        state = self._state
        state.set_frame_roi(
            state.current_frame, self._roi_ctrl.mask.copy()
        )

    def _on_brush_requested(self, mode: str, radius: int) -> None:
        """Activate the brush refinement sub-tool.

        Brush painting is gated to frame 0 (the reference) — the
        pipeline auto-warps the mask to subsequent ref frames at run
        time, so any user input on later frames would be silently
        overwritten.  We require an existing ROI as a sanity check
        because brush refinement only makes sense inside a defined ROI.
        """
        state = self._state
        if not state.image_files:
            state.log_message.emit(self.tr("Load images first."), "warn")
            return
        if state.current_frame != 0:
            state.log_message.emit(
                "Brush painting requires the reference frame — switching to frame 1.",
                "info",
            )
            self._on_roi_edit_for_frame(0)
            # Fall through: ROI check and brush activation continue below
        if state.roi_mask is None:
            state.log_message.emit(
                self.tr("Define a Region of Interest on frame 1 first."),
                "warn",
            )
            return
        if self._brush_ctrl is None:
            self._init_roi_controller()
        if self._brush_ctrl is None:
            return
        # Sync controller buffer from current state in case external
        # paths (clear, restore) mutated state.refine_brush_mask.
        if state.refine_brush_mask is not None:
            self._brush_ctrl.mask = state.refine_brush_mask.copy()
        else:
            self._brush_ctrl.clear()

        state.roi_editing = True
        state.display_changed.emit()
        canvas = self._canvas_area.canvas
        canvas.set_brush_radius(radius)
        canvas.set_brush_mode(mode)
        canvas.set_tool("brush")
        canvas.update_refine_overlay()

    def _on_brush_clear(self) -> None:
        """Clear the brush refinement mask buffer and AppState field."""
        self._cascade_clear_brush()

    def _cascade_clear_brush(self) -> None:
        """Drop the brush refinement mask + buffer + canvas overlay.

        Shared helper used both by the explicit Clear Brush popup action
        and by the ROI-clear cascade paths (Brush#5).
        """
        if self._brush_ctrl is not None:
            self._brush_ctrl.clear()
        self._state.set_refine_brush_mask(None)
        self._canvas_area.canvas.update_refine_overlay()

    def _on_batch_import(self) -> None:
        """Open the batch mask import dialog and load assigned masks."""
        state = self._state
        if not state.image_files:
            state.log_message.emit(self.tr("Load images first."), "warn")
            return

        # Ensure ROI controller is initialized so we can get image dimensions
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return
        img_shape = self._roi_ctrl.shape

        # Determine which frames will actually consume a mask in the
        # pipeline: only the reference-frame set under the current
        # tracking schedule.  Non-reference frames are hidden from the
        # dialog because the solver never reads their masks (see
        # pipeline.py — only `masks[ref_idx]` enters `para.img_ref_mask`).
        from al_dic.core.data_structures import FrameSchedule
        n_images = len(state.image_files)
        if state.tracking_mode == "accumulative" or n_images < 2:
            required_frames = {0}
        else:
            if state.inc_ref_mode == "every_n":
                schedule = FrameSchedule.from_every_n(
                    state.inc_ref_interval, n_images
                )
            elif state.inc_ref_mode == "custom":
                schedule = FrameSchedule.from_custom(
                    state.inc_custom_refs, n_images
                )
            else:  # every_frame (default) or unknown
                schedule = FrameSchedule.from_mode("incremental", n_images)
            required_frames = schedule.ref_frame_set

        from al_dic.gui.dialogs.batch_import_dialog import BatchImportDialog
        from al_dic.i18n import tr_args

        dialog = BatchImportDialog(
            state.image_files,
            parent=self,
            required_frames=required_frames,
            img_shape=img_shape,
        )
        if dialog.exec() != BatchImportDialog.DialogCode.Accepted:
            return

        masks = dialog.load_masks(img_shape)
        for frame_idx, mask in masks.items():
            state.per_frame_rois[frame_idx] = mask
            state.log_message.emit(
                tr_args(
                    self.tr("  Imported mask for frame %1"), frame_idx
                ),
                "info",
            )
        n_masks = len(masks)
        state.log_message.emit(
            QCoreApplication.translate(
                "MainWindow",
                "Batch import: %n mask(s) loaded",
                "",
                n_masks,
            ),
            "success",
        )
        state.roi_changed.emit()

    def _on_roi_import_for_frames(self, mapping: dict) -> None:
        """Import mask files for specific frames (from context menu)."""
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return

        from al_dic.io.io_utils import read_mask_as_bool

        img_shape = self._roi_ctrl.shape
        state = self._state
        count = 0
        for frame_idx, path in mapping.items():
            try:
                mask = read_mask_as_bool(path, target_shape=img_shape)
                state.per_frame_rois[frame_idx] = mask
                count += 1
            except (FileNotFoundError, IOError) as e:
                state.log_message.emit(f"Failed to read: {e}", "warn")

        if count:
            state.log_message.emit(
                QCoreApplication.translate(
                    "App",
                    "Imported Region of Interest for %n frame(s)",
                    "",
                    count,
                ),
                "success",
            )
            state.roi_changed.emit()

    def _on_open_strain_window(self) -> None:
        """Show the strain post-processing window (lazy singleton)."""
        if self._state.results is None:
            self._state.log_message.emit(
                QCoreApplication.translate(
                    "App",
                    "Run DIC first -- no displacement results to post-process.",
                ),
                "warn",
            )
            return
        if self._strain_window is None:
            from al_dic.gui.strain_window import StrainWindow
            self._strain_window = StrainWindow(self._state, parent=None)
        self._strain_window.show()
        self._strain_window.raise_()
        self._strain_window.activateWindow()

    def _on_run_state_changed(self, new_state) -> None:
        """Auto-open the strain post-processing window when a run completes.

        Previously this prompted the user with a Yes/No dialog, which added
        an extra click for an almost-always-yes answer. The strain window is
        non-modal and the user can close it immediately if they only need
        displacement, so auto-opening is strictly less friction.
        """
        from al_dic.gui.app_state import RunState
        if new_state != RunState.DONE or self._state.results is None:
            return
        self._on_open_strain_window()

    def closeEvent(self, event) -> None:
        """Stop and join any running pipeline worker before closing.

        Bug B: closing the main window while a worker QThread is still
        running causes Qt to destroy the underlying QThread object
        before its native thread has exited, producing the
        "QThread: Destroyed while thread is still running" crash on
        Windows. Request a stop and wait for the worker to drain
        before letting the window go.
        """
        worker = self._pipeline_ctrl._worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            worker.wait(5000)
        # Cascade close: the strain window is a child top-level we own,
        # so it must close with the main window to keep lifecycle parity.
        if self._strain_window is not None:
            self._strain_window.close()
            self._strain_window = None
        super().closeEvent(event)


def user_data_dir() -> Path:
    """Per-user writable directory for logs and other app-owned state.

    Deliberately not next to the executable: a onedir bundle is routinely
    unzipped into Program Files, onto a network share, or anywhere else the
    user has no write permission -- and a macOS app bundle must not change
    once signed. Each platform's own place, never a bare folder in the home
    directory: %LOCALAPPDATA% on Windows, Application Support on macOS,
    $XDG_DATA_HOME on Linux. packaging/rthook_pyaldic.py mirrors this.
    """
    home = Path(os.path.expanduser("~"))
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or home)
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / "pyALDIC"


# Path of the log file, once one has been opened. None when not frozen.
_LOG_FILE: Path | None = None


def _configure_logging() -> None:
    """Send the logging stream to a file when there is no console.

    A windowed PyInstaller build has ``sys.stdout is None``, which makes every
    ``print`` a silent no-op and leaves ``logging``'s last-resort handler with
    nowhere to write. Without a file the application has no way at all to
    report a startup failure -- which is precisely when it fails.
    """
    global _LOG_FILE
    if not getattr(sys, "frozen", False):
        return
    try:
        log_dir = user_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "pyALDIC.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        )
    except OSError:
        return  # a read-only profile is not a reason to refuse to start
    _LOG_FILE = log_file


def _report_crash_dialog(exc_type, exc_value, tb_str: str) -> None:
    """Show the traceback in a dialog -- a windowed build has no other channel."""
    if QApplication.instance() is None:
        return
    from al_dic.i18n import tr_args

    box = QMessageBox(
        QMessageBox.Icon.Critical,
        QCoreApplication.translate("Application", "pyALDIC has hit an error"),
        QCoreApplication.translate(
            "Application",
            "An unexpected error occurred. The application may not behave "
            "correctly from here on, so saving your session and restarting "
            "is recommended.",
        ),
    )
    if _LOG_FILE is not None:
        box.setInformativeText(
            tr_args(
                QCoreApplication.translate(
                    "Application", "Details were written to %1"
                ),
                str(_LOG_FILE),
            )
        )
    box.setDetailedText(f"{exc_type.__name__}: {exc_value}" + chr(10) * 2 + tb_str)
    box.exec()


def _global_exception_hook(exc_type, exc_value, exc_tb):
    """Catch unhandled exceptions so the GUI doesn't silently crash."""
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"\n{'='*60}", flush=True)
    print("UNHANDLED EXCEPTION — this would normally crash the GUI:", flush=True)
    print(tb_str, flush=True)
    print(f"{'='*60}\n", flush=True)

    # The log file is the only channel that survives a windowed build.
    logging.getLogger("al_dic").critical("Unhandled exception: %s", tb_str)

    # Also try to log to GUI console if available
    try:
        state = AppState.instance()
        state.log_message.emit(f"CRASH: {exc_type.__name__}: {exc_value}", "error")
        state.log_message.emit(tb_str, "error")
    except Exception:
        pass

    # ...and a dialog, for failures that happen before the console widget
    # exists, or that leave it unreachable.
    try:
        _report_crash_dialog(exc_type, exc_value, tb_str)
    except Exception:
        pass


def main() -> None:
    """Launch the GUI application."""
    # A frozen Windows build re-executes this entry point in every child
    # process, so any future ProcessPool would fork-bomb without this. It is a
    # no-op everywhere else.
    import multiprocessing

    multiprocessing.freeze_support()

    _configure_logging()
    # Install global exception hook to prevent silent crashes
    sys.excepthook = _global_exception_hook

    app = QApplication(sys.argv)
    app.setOrganizationName("pyALDIC")
    app.setApplicationName("pyALDIC")
    app.setStyle("Fusion")  # required for QSS to work correctly
    app.setStyleSheet(build_stylesheet())

    # i18n: install translators before any widget is constructed so
    # tr() wrappers resolve correctly from the very first paint.
    from al_dic.i18n import LanguageManager
    from al_dic.utils.matplotlib_fonts import configure_matplotlib_fonts

    configure_matplotlib_fonts()
    lang_mgr = LanguageManager(app)
    lang_mgr.load(LanguageManager.resolve_language())
    # Keep a reference on the QApplication so widgets can reach the
    # manager for language-switch actions (see Settings > Language menu).
    app._pyaldic_lang_mgr = lang_mgr  # type: ignore[attr-defined]

    window = MainWindow()
    window.show()

    _start_kernel_warmup(window)

    # File association / command line: ``al-dic path/to/foo.aldic`` (or a
    # double-click once .aldic is registered) opens straight into that session.
    session_arg = _session_path_from_argv(sys.argv)
    if session_arg is not None:
        # Defer until the event loop is running so the progress dialog + worker
        # behave normally.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: window.open_session_path(session_arg))

    sys.exit(app.exec())


def _start_kernel_warmup(window: "MainWindow") -> None:
    """Compile the solver kernels in the background, shortly after first paint.

    See al_dic.gui.kernel_warmup for why: without it, the first *Run DIC
    Analysis* of an installation freezes the interface for tens of seconds with
    nothing to explain it.
    """
    from PySide6.QtCore import QTimer

    from al_dic.gui.kernel_warmup import START_DELAY_MS, KernelWarmup

    state = AppState.instance()
    warmup = KernelWarmup(window)
    # Parented to the window so the signal carrier lives as long as the
    # receiver. The worker itself is a daemon thread and outlives neither.
    window._kernel_warmup = warmup  # type: ignore[attr-defined]

    def _announce() -> None:
        state.log_message.emit(
            QCoreApplication.translate(
                "Application",
                "Preparing compute kernels in the background. The first "
                "analysis on a new installation takes longer than the rest.",
            ),
            "info",
        )
        warmup.start()

    def _report(seconds: float) -> None:
        from al_dic.i18n import tr_args

        state.log_message.emit(
            tr_args(
                QCoreApplication.translate(
                    "Application", "Compute kernels ready (%1 s)."
                ),
                f"{seconds:.0f}",
            ),
            "info",
        )

    warmup.compiled.connect(_report)
    QTimer.singleShot(START_DELAY_MS, _announce)


def _session_path_from_argv(argv: list[str]) -> str | None:
    """Return the first existing ``.aldic``/``.aldic.json`` path in *argv*."""
    for arg in argv[1:]:
        low = arg.lower()
        if low.endswith((".aldic", ".aldic.json")) and Path(arg).exists():
            return arg
    return None


if __name__ == "__main__":
    main()
