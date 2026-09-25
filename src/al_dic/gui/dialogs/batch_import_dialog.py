"""Batch ROI mask import dialog.

Three-panel layout for assigning mask image files to frames:
  - Left panel:   available mask images from a selected folder
  - Middle panel: frame assignments (frame index, image filename, mask filename)
  - Right panel:  live preview of the selected frame's image and mask

Supports auto-match by filename number, sequential assignment, and
manual selected-to-selected pairing.

When ``required_frames`` is supplied, only those frame indices appear
in the assignment tree; other frames are hidden because the pipeline
will never consume their masks (e.g. non-reference frames in every-N
incremental mode).

When ``img_shape`` is supplied, masks are pre-scanned and any whose
on-disk size does not match the image size are disabled and excluded
from auto-match / sequential / manual assignment.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPixmap,
    QShowEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.theme import COLORS
from al_dic.gui.theme_manager import themed
from al_dic.gui.window_geometry import fit_dialog_to_screen
from al_dic.i18n import tr_args

# Image file extensions accepted as mask files
_MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".jp2", ".webp"}

# Max number of decoded image / mask buffers held in the preview LRU cache.
# At 4K each grayscale buffer is ~16 MB; capping at 5 keeps the working
# set under ~100 MB so the dialog stays responsive on a typical desktop.
_PREVIEW_CACHE_SIZE = 5


class _PreviewView(QGraphicsView):
    """QGraphicsView with ctrl-less wheel zoom centred on the cursor."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        # Cap zoom so users can't accidentally scroll into oblivion.
        self._zoom = 0
        self._zoom_min = -10
        self._zoom_max = 30

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt API)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1 if delta > 0 else -1
        new_zoom = max(self._zoom_min, min(self._zoom_max, self._zoom + step))
        if new_zoom == self._zoom:
            return
        factor = 1.15 if step > 0 else 1.0 / 1.15
        self._zoom = new_zoom
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        """Reset the view transform to fit the current pixmap."""
        self._zoom = 0
        self.resetTransform()


class _MaskPreviewPanel(QWidget):
    """Live preview of a frame's image with optional mask overlay.

    Three view modes — image only, image+mask, mask only — plus an alpha
    slider and a small color palette let the user inspect any frame
    before accepting the batch import. Decoded buffers are LRU-cached
    so flicking through frames does not re-read the same file repeatedly.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image_cache: "OrderedDict[str, np.ndarray | None]" = OrderedDict()
        self._mask_cache:  "OrderedDict[str, np.ndarray | None]" = OrderedDict()
        self._image_path: str | None = None
        self._mask_path:  str | None = None
        self._frame_idx:  int | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 0, 0)

        title = QLabel(self.tr("Preview"))
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self._info_label = QLabel(self.tr("(no image)"))
        themed(self._info_label,
            "color: {TEXT_SECONDARY};"
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        self._scene = QGraphicsScene(self)
        self._view = _PreviewView(self)
        self._view.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        layout.addWidget(self._view, stretch=1)

        ctrls = QFormLayout()
        ctrls.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._view_mode = QComboBox()
        # userData carries a stable English key so logic doesn't depend
        # on translated visible text.
        self._view_mode.addItem(self.tr("Image only"),    userData="image")
        self._view_mode.addItem(self.tr("Image + Mask"),  userData="both")
        self._view_mode.addItem(self.tr("Mask only"),     userData="mask")
        self._view_mode.setCurrentIndex(1)  # default: Image + Mask
        self._view_mode.currentIndexChanged.connect(self._render)
        ctrls.addRow(self.tr("View:"), self._view_mode)

        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(50)
        self._alpha_slider.valueChanged.connect(self._render)
        ctrls.addRow(self.tr("Alpha:"), self._alpha_slider)

        # Mask overlay color palette. Each addItem call uses a literal
        # source string so pyside6-lupdate can extract it; the
        # disambiguation context disambiguates colour names from
        # unrelated uses of "Blue" / "Red" / etc elsewhere.
        self._color_combo = QComboBox()
        self._color_combo.addItem(
            self.tr("Blue", "Mask overlay color"),
            userData=QColor(59, 130, 246),
        )
        self._color_combo.addItem(
            self.tr("Red", "Mask overlay color"),
            userData=QColor(239, 68, 68),
        )
        self._color_combo.addItem(
            self.tr("Green", "Mask overlay color"),
            userData=QColor(34, 197, 94),
        )
        self._color_combo.addItem(
            self.tr("Yellow", "Mask overlay color"),
            userData=QColor(234, 179, 8),
        )
        self._color_combo.currentIndexChanged.connect(self._render)
        ctrls.addRow(self.tr("Mask color:"), self._color_combo)

        layout.addLayout(ctrls)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target(
        self,
        frame_idx: int | None,
        image_path: str | None,
        mask_path: str | None,
    ) -> None:
        """Switch the preview to a new frame / mask pair.

        Passing ``None`` for ``image_path`` clears the preview.
        """
        self._frame_idx = frame_idx
        self._image_path = image_path
        self._mask_path = mask_path
        self._update_info_label()
        self._render()

    def clear(self) -> None:
        """Clear the preview without dropping the LRU cache."""
        self.set_target(None, None, None)

    # ------------------------------------------------------------------
    # LRU cache helpers
    # ------------------------------------------------------------------

    def _cached(
        self,
        cache: "OrderedDict[str, np.ndarray | None]",
        path: str,
        loader,
    ) -> np.ndarray | None:
        """Return a cached buffer for *path*, decoding via *loader* on miss."""
        if path in cache:
            cache.move_to_end(path)
            return cache[path]
        try:
            buf = loader(path)
        except (FileNotFoundError, IOError, ValueError):
            buf = None
        cache[path] = buf
        if len(cache) > _PREVIEW_CACHE_SIZE:
            cache.popitem(last=False)
        return buf

    @staticmethod
    def _load_grayscale(path: str) -> np.ndarray:
        """Decode an image file to a uint8 grayscale array (H, W)."""
        from al_dic.io.io_utils import _read_unchanged, _to_grayscale, _to_uint8
        raw = _read_unchanged(path)
        gray = _to_grayscale(raw)
        return _to_uint8(gray)

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        """Decode a mask file to a boolean array (H, W)."""
        from al_dic.io.io_utils import read_mask_as_bool
        return read_mask_as_bool(path)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_info_label(self) -> None:
        if self._image_path is None:
            self._info_label.setText(self.tr("(no image)"))
            return
        img_name = Path(self._image_path).name
        mask_name = (
            Path(self._mask_path).name
            if self._mask_path is not None
            else self.tr("No mask assigned")
        )
        frame_str = "—" if self._frame_idx is None else f"{self._frame_idx:02d}"
        # Build "Frame XX — image_name | mask_name" via two placeholders
        # so translators can reorder the prefix.
        self._info_label.setText(tr_args(
            self.tr("Frame %1 — %2"),
            frame_str,
            f"{img_name} | {mask_name}",
        ))

    def _render(self) -> None:
        if self._image_path is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._view.reset_zoom()
            return

        gray = self._cached(
            self._image_cache, self._image_path, self._load_grayscale
        )
        if gray is None:
            self._info_label.setText(self.tr("Failed to load image"))
            self._pixmap_item.setPixmap(QPixmap())
            return

        mode = self._view_mode.currentData()  # "image" | "both" | "mask"
        mask = None
        if self._mask_path is not None and mode != "image":
            mask = self._cached(
                self._mask_cache, self._mask_path, self._load_mask
            )

        rgb = self._compose_rgb(gray, mask, mode)
        qimg = self._numpy_to_qimage(rgb)
        pixmap = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        # Fit the new pixmap on every target change; the user's wheel
        # zoom resets to fit-in-view rather than persisting across
        # different-sized images (no good answer for "what scale should
        # frame 5 inherit from frame 2 when they're different sizes?").
        self._view.reset_zoom()
        self._view.fitInView(
            self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio,
        )

    def _compose_rgb(
        self,
        gray: np.ndarray,
        mask: np.ndarray | None,
        mode: str,
    ) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB array for the chosen view mode."""
        h, w = gray.shape[:2]
        if mode == "mask" and mask is not None:
            out = np.zeros((h, w, 3), dtype=np.uint8)
            color = self._color_combo.currentData()
            out[mask] = (color.red(), color.green(), color.blue())
            return out

        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        if mode == "both" and mask is not None and mask.shape == gray.shape:
            alpha = self._alpha_slider.value() / 100.0
            color = self._color_combo.currentData()
            color_arr = np.array(
                (color.red(), color.green(), color.blue()),
                dtype=np.float32,
            )
            # In-place blend on the masked region only.
            blended = (
                (1.0 - alpha) * rgb[mask].astype(np.float32)
                + alpha * color_arr
            )
            rgb[mask] = blended.astype(np.uint8)
        return rgb

    @staticmethod
    def _numpy_to_qimage(rgb: np.ndarray) -> QImage:
        """Wrap an (H, W, 3) uint8 RGB array as a QImage (deep-copied)."""
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        # copy() detaches from the numpy buffer so Qt can outlive it.
        return QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888,
        ).copy()


class BatchImportDialog(QDialog):
    """Dialog for batch-importing mask files and assigning them to frames."""

    def __init__(
        self,
        image_files: list[str],
        parent=None,
        required_frames: set[int] | None = None,
        img_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Import Region of Interest Masks"))
        # Larger default size to accommodate the new preview column.
        # Three vertical strips at ~340 px each + padding ≈ 1100; height
        # picked so the preview area is at least 400 px tall.
        self.setMinimumSize(1100, 650)
        # Fit-to-screen guard: this modal, taskbar-less dialog must never open
        # off-screen.  Clamped once on first show via showEvent.
        self._screen_fitted = False

        self._image_files = list(image_files)
        # Frames the pipeline will actually consume masks for.  None
        # means "no filter — list every frame" (backward-compatible
        # default for callers that don't yet pass this argument).
        if required_frames is None:
            self._required_frames: set[int] = set(range(len(self._image_files)))
        else:
            self._required_frames = set(required_frames)
        # Expected mask shape (H, W) used by the pre-scan size check.
        # None disables the check entirely.
        self._img_shape = img_shape

        self._mask_folder = ""
        self._mask_files: list[str] = []
        # Per-mask actual shape; populated by _load_mask_files.  Files
        # that fail to decode get shape == None and are treated as
        # mismatched.
        self._mask_shapes: dict[str, tuple[int, int] | None] = {}
        # Set of mask paths whose shape != img_shape (or that failed
        # to decode).  Members are excluded from every assignment path.
        self._mismatched: set[str] = set()
        self._assignments: dict[int, str] = {}

        self._build_ui()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        # Keep this modal, taskbar-less dialog fully on screen so its title bar
        # and buttons always stay reachable.  Only on first show.
        if not self._screen_fitted:
            self._screen_fitted = True
            fit_dialog_to_screen(self, self.parentWidget())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Folder browser row ---
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel(self.tr("Mask Folder:")))
        self._folder_label = QLabel(self.tr("(none)"))
        themed(self._folder_label,
            "color: {TEXT_SECONDARY}; font-style: italic;"
        )
        folder_row.addWidget(self._folder_label, stretch=1)
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._on_browse)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # --- Size-mismatch warning (hidden until a mismatch is detected) ---
        self._size_warning = QLabel("")
        themed(self._size_warning,
            "color: {WARNING}; font-weight: bold;"
        )
        self._size_warning.setWordWrap(True)
        self._size_warning.hide()
        layout.addWidget(self._size_warning)

        # --- Three-panel area: masks | assignments | preview ---
        panels = QHBoxLayout()

        # Left panel: available mask files
        left = QVBoxLayout()
        left.addWidget(QLabel(self.tr("Available Masks")))
        self._mask_list = QListWidget()
        self._mask_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        # currentItemChanged drives the preview from the left column;
        # itemSelectionChanged would fire on every selection toggle and
        # spam the preview during shift-click multi-select.
        self._mask_list.currentItemChanged.connect(self._on_mask_focus)
        left.addWidget(self._mask_list)

        auto_btn = QPushButton(self.tr("Auto-Match by Name"))
        auto_btn.setToolTip(self.tr(
            "Match mask files to frames by number in filename"))
        auto_btn.clicked.connect(self._auto_match)
        left.addWidget(auto_btn)

        seq_btn = QPushButton(self.tr("Assign Sequential"))
        seq_btn.setToolTip(self.tr(
            "Assign masks to frames in order starting from frame 0"))
        seq_btn.clicked.connect(self._assign_sequential)
        left.addWidget(seq_btn)

        panels.addLayout(left, stretch=1)

        # Middle panel: frame assignments
        mid = QVBoxLayout()
        mid.addWidget(QLabel(self.tr("Frame Assignments")))
        self._assign_tree = QTreeWidget()
        self._assign_tree.setHeaderLabels([
            self.tr("Frame"), self.tr("Image"), self.tr("Mask")])
        self._assign_tree.setColumnCount(3)
        self._assign_tree.setRootIsDecorated(False)
        header = self._assign_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._assign_tree.setColumnWidth(0, 50)
        self._assign_tree.currentItemChanged.connect(self._on_frame_focus)
        mid.addWidget(self._assign_tree)

        assign_btn = QPushButton(self.tr("Assign Selected ->"))
        assign_btn.setToolTip(self.tr(
            "Pair selected mask(s) with selected frame(s)"))
        assign_btn.clicked.connect(self._assign_selected)
        mid.addWidget(assign_btn)

        clear_btn = QPushButton(self.tr("Clear All"))
        clear_btn.clicked.connect(self._clear_assignments)
        mid.addWidget(clear_btn)

        panels.addLayout(mid, stretch=1)

        # Right panel: live preview
        self._preview = _MaskPreviewPanel(self)
        panels.addWidget(self._preview, stretch=1)

        layout.addLayout(panels)

        # --- OK / Cancel ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._populate_frames()

    def _populate_frames(self) -> None:
        """Fill the assignment tree with one row per *required* image frame.

        When ``required_frames`` filters out a frame (e.g. non-reference
        frame in every-N incremental mode), it does not appear in the
        tree — the pipeline would never consume its mask anyway.
        """
        self._assign_tree.clear()
        for i, fpath in enumerate(self._image_files):
            if i not in self._required_frames:
                continue
            fname = Path(fpath).name
            item = QTreeWidgetItem([f"{i:02d}", fname, ""])
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            self._assign_tree.addTopLevelItem(item)

    # ------------------------------------------------------------------
    # Folder browsing and mask loading
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select Mask Folder")
        )
        if not folder:
            return
        self._mask_folder = folder
        self._folder_label.setText(folder)
        self._load_mask_files(folder)

    def _load_mask_files(self, folder: str) -> None:
        """Scan *folder* for image files, pre-check sizes, populate list."""
        p = Path(folder)
        self._mask_files = sorted(
            str(f)
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _MASK_EXTENSIONS
        )

        # Pre-scan: read each mask's actual on-disk shape so we can
        # disable mismatched files before the user assigns them.
        self._mask_shapes = {}
        self._mismatched.clear()
        if self._img_shape is not None:
            from al_dic.io.io_utils import _read_unchanged, _to_grayscale
            for mf in self._mask_files:
                try:
                    raw = _read_unchanged(mf)
                    gray = _to_grayscale(raw)
                    shape = gray.shape[:2]
                except (FileNotFoundError, IOError):
                    shape = None
                self._mask_shapes[mf] = shape
                if shape != self._img_shape:
                    self._mismatched.add(mf)

        # Drop any prior assignments that referenced now-mismatched files.
        if self._mismatched:
            self._assignments = {
                fr: path
                for fr, path in self._assignments.items()
                if path not in self._mismatched
            }

        self._mask_list.clear()
        for mf in self._mask_files:
            item = QListWidgetItem(Path(mf).name)
            item.setData(Qt.ItemDataRole.UserRole, mf)
            if mf in self._mismatched:
                # Visually deemphasize and explain via tooltip.  Disabled
                # items are skipped by every assignment path.
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                              & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QColor(COLORS.TEXT_MUTED))
                shape = self._mask_shapes.get(mf)
                if shape is None:
                    item.setToolTip(self.tr("Failed to read mask file."))
                else:
                    item.setToolTip(tr_args(
                        self.tr(
                            "Mismatched shape: %1×%2 (expected %3×%4)"
                        ),
                        shape[0], shape[1],
                        self._img_shape[0], self._img_shape[1],
                    ))
            self._mask_list.addItem(item)

        # Update the top-of-dialog warning summary.
        self._update_size_warning()
        self._refresh_display()

    # ------------------------------------------------------------------
    # Preview wiring
    # ------------------------------------------------------------------

    def _on_frame_focus(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None = None,
    ) -> None:
        """Update the preview when the user clicks/arrow-keys a frame row."""
        if current is None:
            self._preview.clear()
            return
        idx = current.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            self._preview.clear()
            return
        image_path = self._image_files[idx] if 0 <= idx < len(self._image_files) else None
        mask_path = self._assignments.get(idx)
        self._preview.set_target(idx, image_path, mask_path)

    def _on_mask_focus(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None = None,
    ) -> None:
        """Preview a mask file without committing to any assignment.

        Reuses whichever image frame is currently selected in the
        assignment tree (or frame 0 as a fallback). Mismatched masks
        are skipped — they can't be selected anyway, but ``current``
        may briefly land on one while keyboard navigation skips past.
        """
        if current is None:
            return
        mask_path = current.data(Qt.ItemDataRole.UserRole)
        if mask_path is None or mask_path in self._mismatched:
            return
        # Pick the currently-focused frame if any, else the first
        # required frame, else frame 0.
        tree_item = self._assign_tree.currentItem()
        if tree_item is not None:
            frame_idx = tree_item.data(0, Qt.ItemDataRole.UserRole)
        elif self._required_frames:
            frame_idx = min(self._required_frames)
        else:
            frame_idx = 0
        image_path = (
            self._image_files[frame_idx]
            if 0 <= frame_idx < len(self._image_files)
            else None
        )
        self._preview.set_target(frame_idx, image_path, mask_path)

    def _update_size_warning(self) -> None:
        """Show or hide the size-mismatch warning above the panels."""
        n = len(self._mismatched)
        if n == 0:
            self._size_warning.hide()
            return
        from PySide6.QtCore import QCoreApplication
        self._size_warning.setText(
            QCoreApplication.translate(
                "BatchImportDialog",
                "%n mask(s) have mismatched sizes and are disabled.",
                "",
                n,
            )
        )
        self._size_warning.show()

    # ------------------------------------------------------------------
    # Assignment strategies
    # ------------------------------------------------------------------

    def _auto_match(self) -> None:
        """Match masks to frames by extracting the last number in filename."""
        self._assignments.clear()
        for mf in self._mask_files:
            if mf in self._mismatched:
                continue
            numbers = re.findall(r"\d+", Path(mf).stem)
            if not numbers:
                continue
            idx = int(numbers[-1])
            if idx in self._required_frames:
                self._assignments[idx] = mf
        self._refresh_display()

    def _assign_sequential(self) -> None:
        """Assign masks to required frames in sorted order, 1:1."""
        self._assignments.clear()
        usable = [mf for mf in self._mask_files if mf not in self._mismatched]
        required_sorted = sorted(self._required_frames)
        for frame_idx, mf in zip(required_sorted, usable):
            self._assignments[frame_idx] = mf
        self._refresh_display()

    def _assign_selected(self) -> None:
        """Pair the selected mask(s) with the selected frame row(s).

        Three cases:
          * 1 mask + 1 frame: assign that pair.
          * 1 mask + N frames: broadcast the mask to every selected frame.
          * N masks + N frames: pair them in selection order.
          * N masks + 1 frame: rejected — a frame can only have one mask.
        """
        mask_items = [
            mi for mi in self._mask_list.selectedItems()
            if mi.data(Qt.ItemDataRole.UserRole) not in self._mismatched
        ]
        frame_items = self._assign_tree.selectedItems()
        if not mask_items or not frame_items:
            return

        # N masks -> 1 frame is ambiguous (which mask wins?).  Refuse
        # rather than silently overwriting, so the user gets a clear
        # signal that the relationship is one-mask-per-frame only.
        if len(mask_items) > 1 and len(frame_items) == 1:
            QMessageBox.warning(
                self,
                self.tr("Invalid assignment"),
                self.tr(
                    "A frame can only have one mask. Select exactly one "
                    "mask, or select multiple frames to assign one mask "
                    "to many."
                ),
            )
            return

        # 1 mask -> N frames: broadcast.
        if len(mask_items) == 1 and len(frame_items) > 1:
            path = mask_items[0].data(Qt.ItemDataRole.UserRole)
            for fi in frame_items:
                idx = fi.data(0, Qt.ItemDataRole.UserRole)
                if idx is not None and path:
                    self._assignments[idx] = path
        else:
            # 1:1 or N:N — pair in selection order.
            for mi, fi in zip(mask_items, frame_items):
                idx = fi.data(0, Qt.ItemDataRole.UserRole)
                path = mi.data(Qt.ItemDataRole.UserRole)
                if idx is not None and path:
                    self._assignments[idx] = path
        self._refresh_display()

    def _clear_assignments(self) -> None:
        """Remove all mask-to-frame assignments."""
        self._assignments.clear()
        self._refresh_display()

    # ------------------------------------------------------------------
    # Display refresh
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        """Update the Mask column in the assignment tree."""
        for i in range(self._assign_tree.topLevelItemCount()):
            item = self._assign_tree.topLevelItem(i)
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            mask_path = self._assignments.get(idx, "")
            name = Path(mask_path).name if mask_path else ""
            item.setText(2, name)
            color = QColor(COLORS.SUCCESS) if name else QColor(COLORS.TEXT_MUTED)
            item.setForeground(2, color)
        # Reflect newly-assigned (or cleared) masks in the preview for
        # the currently focused frame.  Safe to call with None.
        current = self._assign_tree.currentItem() if hasattr(self, "_preview") else None
        if current is not None:
            self._on_frame_focus(current)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_assignments(self) -> dict[int, str]:
        """Return the current frame-index-to-mask-path mapping."""
        return dict(self._assignments)

    def load_masks(self, img_shape: tuple[int, int]) -> dict[int, np.ndarray]:
        """Read and threshold all assigned mask files.

        Supports all bit depths (uint8, uint16, float) and common
        formats (tif, png, bmp, jpg, jp2, webp).  When ``img_shape``
        was supplied at construction, mismatched masks have already
        been excluded from ``self._assignments`` by ``_load_mask_files``
        and the pre-scan UI.

        Args:
            img_shape: (height, width) used as the resize target by
                :func:`read_mask_as_bool`.  Pre-scanned masks already
                match this shape, so the resize is a no-op for them.

        Returns:
            Mapping of frame_index -> boolean mask array.
        """
        from al_dic.io.io_utils import read_mask_as_bool

        result: dict[int, np.ndarray] = {}
        for idx, path in self._assignments.items():
            try:
                result[idx] = read_mask_as_bool(path, target_shape=img_shape)
            except (FileNotFoundError, IOError):
                continue
        return result
