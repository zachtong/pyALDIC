"""AL-DIC GUI application entry point."""

import sys
import traceback
from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QHBoxLayout,
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
        self.setMinimumSize(1420, 800)
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
        file_menu = menu_bar.addMenu("&File")

        open_session_action = QAction("Open Session\u2026", self)
        open_session_action.setShortcut(QKeySequence.StandardKey.Open)
        open_session_action.triggered.connect(self._on_open_session)
        file_menu.addAction(open_session_action)

        save_session_action = QAction("Save Session\u2026", self)
        save_session_action.setShortcut(QKeySequence.StandardKey.Save)
        save_session_action.triggered.connect(self._on_save_session)
        file_menu.addAction(save_session_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
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

        QMessageBox.information(
            self,
            self.tr("Language changed"),
            self.tr(
                "Language set to %1. Please restart pyALDIC for all "
                "widgets to pick up the new language.").arg(
                SUPPORTED_LANGUAGES.get(lang_code, lang_code)),
        )

    def _on_save_session(self) -> None:
        """Save-session dialog: write state + Regions of Interest to JSON."""
        from al_dic.gui.session import SessionError, save_session

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            "",
            "pyALDIC Session (*.aldic.json);;All Files (*)",
        )
        if not path:
            return
        # Enforce extension so the file dialog's filter actually helps
        if not path.endswith(".aldic.json"):
            path = path + ".aldic.json"
        try:
            save_session(Path(path), self._state)
        except SessionError as e:
            QMessageBox.critical(self, "Save Session Failed", str(e))
            return
        self._state.log_message.emit(f"Session saved to {path}", "success")

    def _on_open_session(self) -> None:
        """Open-session dialog: parse JSON, then apply to AppState."""
        from al_dic.gui.session import (
            SessionError,
            apply_session,
            load_session,
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session",
            "",
            "pyALDIC Session (*.aldic.json);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            session = load_session(Path(path))
            apply_session(session, self._state, self._image_ctrl)
        except SessionError as e:
            QMessageBox.critical(self, "Open Session Failed", str(e))
            return
        self._state.log_message.emit(
            f"Session loaded from {path} "
            f"({len(session.per_frame_rois)} Region(s) of Interest restored)",
            "success",
        )

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

    def _on_roi_save(self) -> None:
        """Save the current ROI mask to a PNG file."""
        if self._roi_ctrl is None:
            self._state.log_message.emit(
                "No Region of Interest to save — load images first.", "warn"
            )
            return
        if not self._roi_ctrl.mask.any():
            self._state.log_message.emit(
                "Region of Interest mask is empty.", "warn"
            )
            return
        self._enter_roi_editing()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Region of Interest Mask",
            "roi_mask.png",
            "PNG Images (*.png);;All Files (*)",
        )
        if not path:
            return
        try:
            self._roi_ctrl.save_mask(path)
            self._state.log_message.emit(f"Mask saved to {path}", "success")
        except IOError as e:
            self._state.log_message.emit(f"Save failed: {e}", "error")

    def _on_roi_invert(self) -> None:
        """Invert the ROI mask for the currently edited frame."""
        if self._roi_ctrl is None:
            self._state.log_message.emit(
                "No Region of Interest to invert — load images first.",
                "warn"
            )
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
            state.log_message.emit("Load images first.", "warn")
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
                "Define a Region of Interest on frame 1 first.",
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
            state.log_message.emit("Load images first.", "warn")
            return

        from al_dic.gui.dialogs.batch_import_dialog import BatchImportDialog

        dialog = BatchImportDialog(state.image_files, parent=self)
        if dialog.exec() != BatchImportDialog.DialogCode.Accepted:
            return

        # Ensure ROI controller is initialized so we can get image dimensions
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return

        img_shape = self._roi_ctrl.shape
        masks = dialog.load_masks(img_shape)
        for frame_idx, mask in masks.items():
            state.per_frame_rois[frame_idx] = mask
            state.log_message.emit(
                f"  Imported mask for frame {frame_idx}", "info"
            )
        state.log_message.emit(
            f"Batch import: {len(masks)} masks loaded", "success"
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
                f"Imported Region of Interest for {count} "
                f"frame{'s' if count > 1 else ''}",
                "success",
            )
            state.roi_changed.emit()

    def _on_open_strain_window(self) -> None:
        """Show the strain post-processing window (lazy singleton)."""
        if self._state.results is None:
            self._state.log_message.emit(
                "Run DIC first -- no displacement results to post-process.",
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


def _global_exception_hook(exc_type, exc_value, exc_tb):
    """Catch unhandled exceptions so the GUI doesn't silently crash."""
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"\n{'='*60}", flush=True)
    print("UNHANDLED EXCEPTION — this would normally crash the GUI:", flush=True)
    print(tb_str, flush=True)
    print(f"{'='*60}\n", flush=True)

    # Also try to log to GUI console if available
    try:
        state = AppState.instance()
        state.log_message.emit(f"CRASH: {exc_type.__name__}: {exc_value}", "error")
        state.log_message.emit(tb_str, "error")
    except Exception:
        pass


def main() -> None:
    """Launch the GUI application."""
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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
