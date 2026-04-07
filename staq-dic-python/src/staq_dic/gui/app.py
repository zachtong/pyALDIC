"""STAQ-DIC GUI application entry point."""

import sys
import traceback

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QHBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.controllers.pipeline_controller import PipelineController
from staq_dic.gui.controllers.roi_controller import ROIController
from staq_dic.gui.controllers.viz_controller import VizController
from staq_dic.gui.panels.canvas_area import CanvasArea
from staq_dic.gui.panels.left_sidebar import LeftSidebar
from staq_dic.gui.panels.right_sidebar import RightSidebar
from staq_dic.gui.theme import COLORS, build_stylesheet


class MainWindow(QMainWindow):
    """Three-column main window: left sidebar | canvas | right sidebar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STAQ-DIC v0.1")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # State and controllers
        self._state = AppState.instance()
        self._image_ctrl = ImageController(self._state)
        self._roi_ctrl: ROIController | None = None

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

        # Wire ROI toolbar signals
        roi_tb = self._left_sidebar.roi_toolbar
        roi_tb.draw_requested.connect(self._on_draw_requested)
        roi_tb.clear_requested.connect(self._on_roi_clear)
        roi_tb.import_requested.connect(self._on_roi_import)
        roi_tb.save_requested.connect(self._on_roi_save)
        roi_tb.invert_requested.connect(self._on_roi_invert)
        roi_tb.batch_import_requested.connect(self._on_batch_import)

        # When canvas finishes drawing, deactivate toolbar highlight
        self._canvas_area.canvas.drawing_finished.connect(roi_tb.deactivate)

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

        # Clear viz caches when results change (new pipeline run)
        self._state.results_changed.connect(self._viz_ctrl.clear_all)

    def _init_roi_controller(self) -> None:
        """Create a ROI controller matching the loaded image dimensions."""
        if not self._state.image_files:
            return
        try:
            rgb = self._image_ctrl.read_image_rgb(0)
            h, w = rgb.shape[:2]
            self._roi_ctrl = ROIController((h, w))
            self._canvas_area.canvas.set_roi_controller(self._roi_ctrl)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _enter_roi_editing(self) -> None:
        """Switch to ROI editing mode — show ROI overlay, hide field overlay."""
        self._state.roi_editing = True
        self._state.display_changed.emit()

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
        """Clear the ROI mask for the currently edited frame."""
        if self._roi_ctrl is not None:
            self._enter_roi_editing()
            self._roi_ctrl.clear()
            self._canvas_area.canvas.update_roi_overlay()
            state = self._state
            state.set_frame_roi(state.current_frame, None)

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
                "No ROI to save — load images first.", "warn"
            )
            return
        if not self._roi_ctrl.mask.any():
            self._state.log_message.emit("ROI mask is empty.", "warn")
            return
        self._enter_roi_editing()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI Mask",
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
                "No ROI to invert — load images first.", "warn"
            )
            return
        self._enter_roi_editing()
        self._roi_ctrl.invert()
        self._canvas_area.canvas.update_roi_overlay()
        state = self._state
        state.set_frame_roi(
            state.current_frame, self._roi_ctrl.mask.copy()
        )

    def _on_batch_import(self) -> None:
        """Open the batch mask import dialog and load assigned masks."""
        state = self._state
        if not state.image_files:
            state.log_message.emit("Load images first.", "warn")
            return

        from staq_dic.gui.dialogs.batch_import_dialog import BatchImportDialog

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

        from staq_dic.io.io_utils import read_mask_as_bool

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
                f"Imported ROI for {count} frame{'s' if count > 1 else ''}",
                "success",
            )
            state.roi_changed.emit()


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
    app.setStyle("Fusion")  # required for QSS to work correctly
    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
