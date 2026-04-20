"""i18n workflow helper.

Subcommands:
    extract   — scan src/al_dic/gui/ for tr() strings and update every
                language's .ts source file (uses pyside6-lupdate).
    compile   — convert .ts sources into binary .qm runtime catalogs
                (uses pyside6-lrelease).
    stats     — count how many translation entries are unfinished per
                language.
    add-lang  — create a new empty .ts file for a language code
                (e.g. `add-lang ko` for Korean).

Designed to be idempotent: safe to run many times. Meant to be driven
directly or via CI ("extract then git-diff the .ts files — any drift
means a new tr() string was added without syncing").

Adding a new target language:
    1. `python tools/i18n.py add-lang <code>` (e.g. zh_TW, ja, ko, de).
    2. Add a column to docs/i18n/glossary.md.
    3. Fill `<translation type="unfinished">` entries in the .ts file
       — typically with Claude Code + the glossary as context.
    4. `python tools/i18n.py compile`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCAN_DIR = PROJECT_ROOT / "src" / "al_dic" / "gui"
TS_DIR = PROJECT_ROOT / "src" / "al_dic" / "i18n" / "source"
QM_DIR = PROJECT_ROOT / "src" / "al_dic" / "i18n" / "compiled"

# Languages pyALDIC targets. Must match SUPPORTED_LANGUAGES in
# src/al_dic/i18n/__init__.py. Order = tier: first shipped with v0.x,
# later ones accepted by community contribution.
LANGUAGES: tuple[str, ...] = (
    # Tier 1 — core Asian + source group
    "zh_CN",    # Simplified Chinese (primary non-English)
    "zh_TW",    # Traditional Chinese
    "ja",       # Japanese
    "ko",       # Korean
    # Tier 2 — European engineering markets
    "de",       # German
    "fr",       # French
    "es",       # Spanish
)


# ---------- Tool discovery -------------------------------------------------

def _find_tool(name: str) -> str:
    """Locate pyside6-lupdate / pyside6-lrelease on PATH."""
    path = shutil.which(name)
    if path:
        return path
    # Fallback: PySide6 ships binaries in its package dir on some platforms.
    try:
        import PySide6  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError(
            f"Cannot locate {name!r}: PySide6 is not installed."
        ) from None
    candidate = Path(PySide6.__file__).parent / name
    if candidate.is_file() or candidate.with_suffix(".exe").is_file():
        return str(candidate)
    raise RuntimeError(
        f"Cannot locate {name!r} on PATH or in PySide6 package dir."
    )


# ---------- Helpers --------------------------------------------------------

def _ensure_dirs() -> None:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    QM_DIR.mkdir(parents=True, exist_ok=True)


def _ts_path(lang: str) -> Path:
    return TS_DIR / f"al_dic_{lang}.ts"


def _qm_path(lang: str) -> Path:
    return QM_DIR / f"al_dic_{lang}.qm"


def _unfinished_count(ts_file: Path) -> int:
    if not ts_file.is_file():
        return 0
    text = ts_file.read_text(encoding="utf-8")
    return len(re.findall(r'<translation type="unfinished"', text))


def _total_entries(ts_file: Path) -> int:
    if not ts_file.is_file():
        return 0
    text = ts_file.read_text(encoding="utf-8")
    return len(re.findall(r"<message", text))


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


# ---------- Subcommands ----------------------------------------------------

def cmd_extract(_args: argparse.Namespace) -> None:
    """Scan gui/ and refresh every language's .ts file.

    pyside6-lupdate does NOT walk directories recursively for Python
    sources (it expects Qt .pro files for that). We enumerate every
    .py under gui/ ourselves and pass the explicit file list.
    """
    _ensure_dirs()
    lupdate = _find_tool("pyside6-lupdate")

    if not SOURCE_SCAN_DIR.is_dir():
        raise SystemExit(f"GUI source dir not found: {SOURCE_SCAN_DIR}")

    py_files = sorted(str(p) for p in SOURCE_SCAN_DIR.rglob("*.py"))
    if not py_files:
        raise SystemExit(f"No .py files found under {SOURCE_SCAN_DIR}")

    print(f"[extract] scanning {len(py_files)} .py files under "
          f"{SOURCE_SCAN_DIR.relative_to(PROJECT_ROOT)}")
    for lang in LANGUAGES:
        ts = _ts_path(lang)
        print(f"[extract] {lang} -> {ts.relative_to(PROJECT_ROOT)}")
        _run([
            lupdate,
            *py_files,
            "-ts", str(ts),
            "-no-obsolete",
            "-source-language", "en_US",
            "-target-language", lang,
        ])


def cmd_compile(_args: argparse.Namespace) -> None:
    """Compile every .ts into its .qm binary catalog."""
    _ensure_dirs()
    lrelease = _find_tool("pyside6-lrelease")

    for lang in LANGUAGES:
        ts = _ts_path(lang)
        qm = _qm_path(lang)
        if not ts.is_file():
            print(f"[compile] skip {lang} — no source .ts yet")
            continue
        print(f"[compile] {lang} -> {qm.relative_to(PROJECT_ROOT)}")
        _run([lrelease, str(ts), "-qm", str(qm)])


def cmd_stats(_args: argparse.Namespace) -> None:
    """Report unfinished vs total entries for each language."""
    print(f"{'Language':<10} {'Total':>8} {'Unfinished':>12} {'Complete':>10}")
    print("-" * 44)
    for lang in LANGUAGES:
        ts = _ts_path(lang)
        if not ts.is_file():
            print(f"{lang:<10} (no .ts file yet - run 'extract' first)")
            continue
        total = _total_entries(ts)
        unfin = _unfinished_count(ts)
        if total == 0:
            print(f"{lang:<10} (empty - no tr() strings found in gui/)")
            continue
        complete_pct = 100 * (total - unfin) / total
        print(f"{lang:<10} {total:>8} {unfin:>12} {complete_pct:>9.1f}%")


def cmd_add_lang(args: argparse.Namespace) -> None:
    """Register a new language code and create an empty .ts for it."""
    code: str = args.code
    if code in LANGUAGES:
        print(f"[add-lang] {code} already registered.")
        return
    print(
        f"[add-lang] To register {code!r}, add it to the LANGUAGES tuple in "
        f"tools/i18n.py (near the top) and rerun `extract`. This script does "
        f"not self-modify to keep the registration explicit."
    )


# ---------- Entry point ----------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract", help="scan gui/ and update .ts files").set_defaults(
        func=cmd_extract)
    sub.add_parser("compile", help="build .qm from .ts").set_defaults(
        func=cmd_compile)
    sub.add_parser("stats", help="report translation completion").set_defaults(
        func=cmd_stats)
    p_add = sub.add_parser("add-lang", help="advise how to add a new language")
    p_add.add_argument("code", help='e.g. "de", "ko", "zh_TW"')
    p_add.set_defaults(func=cmd_add_lang)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
