"""Wrap simple log_message.emit("literal", "level") calls in self.tr().

Only rewrites the easy case where the first argument is a single
double-quoted literal. f-strings and concatenation remain untouched
and must be converted manually to the %1/%2 placeholder form.
"""

from __future__ import annotations

import re
from pathlib import Path

GUI = Path(__file__).resolve().parents[1] / "src" / "al_dic" / "gui"

# .log_message.emit("literal", "level")
# Pattern groups: 1=prefix up to opening paren, 2=string body (no escapes),
# 3=comma + level quoted literal
PATTERN = re.compile(
    r'(\.log_message\.emit\()'
    r'"([^"\n]+?)"'
    r'(\s*,\s*"(?:error|warn|warning|info|success|debug)"\s*\))'
)


def main() -> None:
    total = 0
    for f in GUI.rglob("*.py"):
        text = f.read_text(encoding="utf-8")

        def _rep(m: re.Match) -> str:
            nonlocal total
            total += 1
            return f'{m.group(1)}self.tr("{m.group(2)}"){m.group(3)}'

        new_text = PATTERN.sub(_rep, text)
        if new_text != text:
            f.write_text(new_text, encoding="utf-8")
            print(f"  {f.relative_to(GUI.parent.parent.parent)}")
    print(f"\nWrapped {total} literal log emits")


if __name__ == "__main__":
    main()
