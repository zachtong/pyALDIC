"""One-off retrofit: wrap hardcoded UI strings in self.tr(...).

Pattern matches common Qt constructors/setters whose first string
argument is user-visible. Skips strings already inside tr()/tr_args(),
empty strings, CSS-looking strings, and stylesheet fragments.

Run once to sweep unretrofitted widgets, then review the diff before
committing. Cleanup helper; not a reusable dev tool.
"""

from __future__ import annotations

import re
from pathlib import Path

GUI_ROOT = Path(__file__).resolve().parents[1] / "src" / "al_dic" / "gui"

# Target files (high-visibility widgets that still have hardcoded strings).
TARGETS = [
    "strain_window.py",
    "panels/canvas_area.py",
    "panels/strain_canvas.py",
    "widgets/frame_navigator.py",
    "widgets/strain_navigator.py",
    "widgets/image_list.py",
    "widgets/strain_param_panel.py",
    "widgets/strain_field_selector.py",
    "widgets/console_log.py",
    "widgets/colorbar_overlay.py",
    "widgets/physical_units_widget.py",
    "widgets/strain_viz_panel.py",
    "widgets/color_range.py",
    "widgets/velocity_settings.py",
    "widgets/mesh_appearance_widget.py",
    "widgets/sticky_headers_overlay.py",
    "widgets/field_selector.py",
    "widgets/canvas_config_overlay.py",
    "widgets/advanced_tuning_widget.py",
]

# Patterns that capture (prefix, quoted_string, suffix) so we can
# reconstruct with self.tr(...) wrapping around the captured text.
PATTERNS = [
    # Constructor forms where text is the first positional arg.
    re.compile(r'(\bQ(?:Label|PushButton|CheckBox|RadioButton|GroupBox|ToolButton)\()"([^"\n]*?)"(\s*\))'),
    # setText / setToolTip / setWindowTitle / setPlaceholderText / setStatusTip
    re.compile(r'(\.set(?:Text|ToolTip|WindowTitle|PlaceholderText|StatusTip|Title)\()"([^"\n]*?)"(\s*\))'),
    # QAction("text", self)
    re.compile(r'(\bQAction\()"([^"\n]*?)"(\s*,)'),
]

SKIP_PATTERNS = [
    re.compile(r"^\s*$"),           # empty / whitespace
    re.compile(r"^[a-z_]+$"),       # snake_case identifiers only (not "Edit"/"Add")
    re.compile(r"^\*\.\w+"),        # file glob like *.png
    re.compile(r"^\.[\w]+$"),       # extension like .json
    re.compile(r"^#[0-9A-Fa-f]{3,8}$"),  # hex color
    re.compile(r"^\d+(\.\d+)?(\s*[a-z]+)?$"),  # pure number maybe with unit
    re.compile(r"^[^\w\s]*$"),      # punctuation only (arrows etc)
]


def should_skip(text: str) -> bool:
    if any(p.match(text) for p in SKIP_PATTERNS):
        return True
    # Already-translated sentinels some code may emit
    if text.startswith("self.tr(") or text.startswith("tr_args("):
        return True
    return False


def process_file(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    orig = src
    n_changed = 0
    for pat in PATTERNS:
        def _repl(m: re.Match) -> str:
            nonlocal n_changed
            prefix, body, suffix = m.group(1), m.group(2), m.group(3)
            if should_skip(body):
                return m.group(0)
            n_changed += 1
            return f'{prefix}self.tr("{body}"){suffix}'
        src = pat.sub(_repl, src)

    if src != orig:
        path.write_text(src, encoding="utf-8")
        print(f"  {n_changed:3d} wraps -> {path.relative_to(GUI_ROOT.parent.parent.parent)}")
    return n_changed


def main() -> None:
    total = 0
    for rel in TARGETS:
        p = GUI_ROOT / rel
        if not p.is_file():
            print(f"  SKIP (not found): {rel}")
            continue
        total += process_file(p)
    print(f"\nTotal wraps applied: {total}")


if __name__ == "__main__":
    main()
