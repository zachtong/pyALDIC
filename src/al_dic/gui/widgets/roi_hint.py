"""Dynamic hint above the Region of Interest toolbar.

Shows which frames need a Region of Interest based on the current
workflow type (tracking mode + reference-update policy). Updates live
whenever the workflow parameters change so users never have to guess
whether they should draw a region on frame 1 only or on several
reference frames.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS


_HINT_STYLE = (
    f"QLabel {{ "
    f"color: {COLORS.TEXT_SECONDARY}; "
    f"background: {COLORS.BG_INPUT}; "
    f"border: 1px solid {COLORS.BORDER}; "
    f"border-left: 3px solid {COLORS.ACCENT}; "
    f"border-radius: 3px; "
    f"padding: 6px 8px; "
    f"font-size: 11px; "
    f"}}"
)


class ROIHint(QLabel):
    """Info line showing which frames need a Region of Interest.

    Listens to ``AppState.params_changed`` and rewrites itself whenever
    ``tracking_mode``, ``inc_ref_mode``, ``inc_ref_interval``, or
    ``inc_custom_refs`` change. Also listens to ``images_changed`` so
    the frame count in the hint stays current.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self.setStyleSheet(_HINT_STYLE)
        self.setWordWrap(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self._state.params_changed.connect(self._refresh)
        self._state.images_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        """Recompute the hint text based on current state."""
        s = self._state
        n = len(s.image_files)

        # No images yet: skip mode-specific advice, just tell the user
        # they need to load images first.
        if n == 0:
            self.setText(
                "Load images first, then draw a Region of Interest on "
                "frame 1."
            )
            return

        if s.tracking_mode == "accumulative":
            self.setText(
                "<b>Accumulative mode</b> \u2014 only frame 1 needs a "
                "Region of Interest. All later frames are compared "
                "against it directly."
            )
            return

        # Incremental: the required ref frames depend on inc_ref_mode
        mode = s.inc_ref_mode
        if mode == "every_frame":
            self.setText(
                "<b>Incremental, every frame</b> \u2014 frame 1 needs a "
                "Region of Interest. It is automatically warped forward "
                "to each later frame (no per-frame drawing required)."
            )
            return

        if mode == "every_n":
            N = max(2, s.inc_ref_interval)
            # Reference frames are 0, N, 2N, ... up to n-1 (0-based)
            refs = list(range(0, n, N))
            # Convert to 1-based for display
            refs_display = [str(r + 1) for r in refs]
            if len(refs_display) > 8:
                preview = ", ".join(refs_display[:8]) + ", \u2026"
            else:
                preview = ", ".join(refs_display)
            self.setText(
                f"<b>Incremental, every {N} frames</b> \u2014 draw a "
                f"Region of Interest on frames: <b>{preview}</b> "
                f"({len(refs_display)} reference frames total)."
            )
            return

        if mode == "custom":
            refs = sorted(set(s.inc_custom_refs))
            # Always include frame 0 as a reference
            if 0 not in refs:
                refs = [0] + refs
            refs_display = [str(r + 1) for r in refs]
            if not refs_display or refs_display == ["1"]:
                self.setText(
                    "<b>Incremental, custom</b> \u2014 no custom reference "
                    "frames set. Frame 1 will be the only reference; add "
                    "more indices in the Reference Frames field."
                )
                return
            if len(refs_display) > 8:
                preview = ", ".join(refs_display[:8]) + ", \u2026"
            else:
                preview = ", ".join(refs_display)
            self.setText(
                f"<b>Incremental, custom</b> \u2014 draw a Region of "
                f"Interest on frames: <b>{preview}</b> "
                f"({len(refs_display)} reference frames total)."
            )
            return

        # Fallback — unrecognized mode
        self.setText(
            "Draw a Region of Interest on frame 1."
        )
