"""The Analysis tab's left side: tools, the probe canvas and the frame navigator.

The canvas carries two overlays on its viewport, both transparent to the
mouse: the colorbar of the field drawn under the probes, and a banner that
says what the canvas is waiting for.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui import icons
from al_dic.gui.panels.analysis import text
from al_dic.gui.panels.probe_canvas import ProbeCanvas
from al_dic.gui.panels.strain_canvas import FieldImage
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.colorbar_overlay import ColorbarOverlay
from al_dic.gui.widgets.strain_navigator import StrainNavigator

# Placement tools in toolbar order, with their icons.
TOOLS = (
    ("point", icons.icon_probe_point),
    ("line", icons.icon_probe_line),
    ("area_rect", icons.icon_probe_rect),
    ("area_circle", icons.icon_probe_circle),
    ("area_polygon", icons.icon_probe_polygon),
    ("extensometer", icons.icon_extensometer),
    ("crack_gauge", icons.icon_crack_gauge),
)


class AnalysisCanvasPanel(QWidget):
    """Toolbar, probe canvas with its overlays, and the frame navigator."""

    # A tool button was pressed: (tool, checked).
    tool_toggled = Signal(str, bool)
    # The "Show field" box changed.
    show_field_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self._build_toolbar())

        self.canvas = ProbeCanvas()
        self._fit_btn.clicked.connect(self.canvas.fit_to_view)
        self._zoom_100_btn.clicked.connect(self.canvas.zoom_to_100)
        self._zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self._zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        column.addWidget(self.canvas, 1)

        viewport = self.canvas.viewport()
        self.colorbar = ColorbarOverlay(viewport)
        self.banner = QLabel(viewport)
        self.banner.setTextFormat(Qt.TextFormat.RichText)
        self.banner.setWordWrap(True)
        self.banner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.banner.setStyleSheet(
            f"background: rgba(20, 25, 41, 215); color: {COLORS.TEXT_PRIMARY};"
            " border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        self.banner.hide()
        viewport.installEventFilter(self)

        self.nav = StrainNavigator()
        column.addWidget(self.nav)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("analysisToolBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        # Icons, named by tooltip: seven text buttons do not fit beside the
        # chart in German. The armed tool's name and instructions show on
        # the canvas while it is armed.
        self.tool_buttons: dict[str, QPushButton] = {}
        for tool, icon in TOOLS:
            if tool == "extensometer":
                row.addWidget(_separator())
            button = _icon_button(icon())
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, t=tool: self.tool_toggled.emit(t, checked))
            row.addWidget(button)
            self.tool_buttons[tool] = button
        row.addWidget(_separator())
        self._fit_btn = _icon_button(icons.icon_maximize())
        self._zoom_100_btn = QPushButton()
        self._zoom_in_btn = _icon_button(icons.icon_zoom_in())
        self._zoom_out_btn = _icon_button(icons.icon_zoom_out())
        for button in (self._fit_btn, self._zoom_100_btn,
                       self._zoom_in_btn, self._zoom_out_btn):
            row.addWidget(button)
        row.addStretch()
        self.show_field_box = QCheckBox()
        self.show_field_box.setChecked(True)
        self.show_field_box.toggled.connect(self.show_field_toggled)
        row.addWidget(self.show_field_box)
        return bar

    # -- text -------------------------------------------------------------

    def retranslate_ui(self, has_selection: bool) -> None:
        for tool, button in self.tool_buttons.items():
            # Name and instructions are two sentences, one per line.
            button.setToolTip(f"{text.tool_name(tool)}\n{text.tool_instruction(tool)}")
            if button.icon().isNull():          # no SVG support: say it
                button.setText(text.tool_name(tool))
        for button, label in (
            (self._fit_btn, QCoreApplication.translate(
                "AnalysisTab", "Fit", "Zoom button: fit the image to the view")),
            (self._zoom_in_btn, "+"), (self._zoom_out_btn, "–"),
        ):
            if button.icon().isNull():
                button.setText(label)
        self._fit_btn.setToolTip(
            QCoreApplication.translate("AnalysisTab", "Fit image to viewport"))
        self._zoom_100_btn.setText(QCoreApplication.translate(
            "AnalysisTab", "100%", "Zoom button: one image pixel per screen pixel"))
        self._zoom_100_btn.setToolTip(
            QCoreApplication.translate("AnalysisTab", "Zoom to 100% (1:1)"))
        self._zoom_in_btn.setToolTip(QCoreApplication.translate("AnalysisTab", "Zoom in"))
        self._zoom_out_btn.setToolTip(QCoreApplication.translate("AnalysisTab", "Zoom out"))
        self.show_field_box.setText(QCoreApplication.translate(
            "AnalysisTab", "Show field", "Analysis canvas: colour the image by the field"))
        self.show_field_box.setToolTip(QCoreApplication.translate(
            "AnalysisTab",
            "Colour the reference image with the plotted field at the current "
            "frame. For a gauge reading, the Strain Field tab's field is shown."))
        self.update_banner(has_selection)

    # -- tools and banner ---------------------------------------------------

    def sync_tools(self) -> None:
        """Check the button of the canvas's armed tool, and only that one."""
        active = self.canvas.tool
        for tool, button in self.tool_buttons.items():
            button.setChecked(tool == active)

    def update_banner(self, has_selection: bool) -> None:
        html = text.banner_html(self.canvas.tool, has_selection)
        self.banner.setText(html)
        self.banner.setVisible(bool(html))
        self._place_banner()

    def _place_banner(self) -> None:
        if self.banner.isHidden():
            return
        viewport = self.canvas.viewport()
        # Clear of the colorbar on the right.
        width = max(160, min(460, viewport.width() - 130))
        self.banner.setGeometry(8, 8, width, self.banner.heightForWidth(width))
        self.banner.raise_()

    # -- the field ----------------------------------------------------------

    def show_field(self, image: FieldImage, scale: float, unit: str) -> None:
        """Draw *image* under the probes, its colorbar in the chart's unit.

        The colorbar is only a legend, so its numbers can be rescaled (to %
        or µε) without touching the picture.
        """
        self.canvas.show_field(image)
        viewport = self.canvas.viewport()
        self.colorbar.setGeometry(0, 0, viewport.width(), viewport.height())
        label = f"{image.label} ({unit})" if scale != 1.0 else image.label
        self.colorbar.update_params(
            image.cmap, image.vmin * scale, image.vmax * scale, label)
        self.colorbar.setVisible(True)

    def clear_field(self) -> None:
        self.canvas.clear_overlay()
        self.colorbar.setVisible(False)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.canvas.viewport() and event.type() == QEvent.Type.Resize:
            viewport = self.canvas.viewport()
            self.colorbar.setGeometry(0, 0, viewport.width(), viewport.height())
            self._place_banner()
        return super().eventFilter(obj, event)


def _icon_button(icon: QIcon) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
    if not icon.isNull():
        button.setFixedWidth(32)        # an icon alone has nothing to translate
    return button


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setStyleSheet(f"color: {COLORS.BORDER};")
    return line


__all__ = ["AnalysisCanvasPanel", "TOOLS"]
