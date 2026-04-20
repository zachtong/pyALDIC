"""Audit German layout — find widgets with setFixedWidth/FixedSize
that might truncate long German labels, and report the worst-case
label-to-fixed-width ratios at the de locale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
app.setOrganizationName("pyALDIC-test")

from al_dic.i18n import LanguageManager

mgr = LanguageManager(app)
mgr.load("de")


# 1) Grep for setFixedWidth/setFixedSize in gui/ to find constrained widgets
GUI = Path(__file__).resolve().parents[1] / "src" / "al_dic" / "gui"

print("=== Widgets with explicit pixel constraints ===")
pat_fw = re.compile(r"setFixedWidth\((\d+)\)")
pat_fh = re.compile(r"setFixedHeight\((\d+)\)")
pat_fs = re.compile(r"setFixedSize\((\d+),\s*(\d+)\)")
pat_mw = re.compile(r"setMinimumWidth\((\d+)\)")

constraints: list[tuple[str, int, str]] = []  # (file, lineno, excerpt)
for f in GUI.rglob("*.py"):
    lines = f.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if pat_fw.search(line) or pat_fs.search(line):
            rel = f.relative_to(GUI.parent.parent.parent)
            constraints.append((str(rel), i, line.strip()))

print(f"Found {len(constraints)} fixed-width/size call sites.")
for rel, ln, excerpt in constraints[:40]:
    print(f"  {rel}:{ln}  {excerpt[:90]}")

# 2) Measure common labels vs fixed widths
print()
print("=== Worst-case German label widths ===")
samples = {
    "Tracking Mode":          120,   # param_panel label
    "Solver":                 120,
    "Reference Update":       108,
    "Subset Size":            120,
    "Subset Step":            120,
    "Search Range":           120,
    "Refinement Level":       120,
    "Interval":               108,
    "Reference Frames":       108,
    "Colormap":                64,   # right_sidebar
    "Opacity":                 64,
    "Run DIC Analysis":       None,  # no fixed width (btn-primary)
    "Cancel":                 None,
    "Export Results":         None,
    "Open Strain Window":     None,
    "ADMM Iterations":        120,
    "Method":                 None,  # strain_param_panel
    "VSG size":               None,
    "Strain field smoothing": None,
    "Strain type":            None,
    "Browse…":                 90,   # export dialog
    "Open Folder":             90,
}

from al_dic.gui.theme import build_stylesheet
app.setStyleSheet(build_stylesheet())

# Use a QLabel to measure
from PySide6.QtWidgets import QLabel
probe = QLabel()
font = probe.font()
fm = QFontMetrics(font)

overflow: list[tuple[str, str, int, int]] = []
for en, fixed_px in samples.items():
    de_text = QCoreApplication.translate("ParamPanel", en)
    if de_text == en:
        de_text = QCoreApplication.translate("RightSidebar", en)
    if de_text == en:
        de_text = QCoreApplication.translate("StrainParamPanel", en)
    if de_text == en:
        de_text = QCoreApplication.translate("ExportDialog", en)
    width_px = fm.horizontalAdvance(de_text)
    status = ""
    if fixed_px is not None and width_px > fixed_px:
        status = f"  OVERFLOW by {width_px - fixed_px}px"
        overflow.append((en, de_text, fixed_px, width_px))
    fx = f"{fixed_px}" if fixed_px else "flex"
    print(f"  {en:28s} -> {de_text:38s} fixed={fx:>5s}  actual={width_px:>4d}px{status}")

print()
print(f"=== Summary: {len(overflow)} label(s) will be truncated in de ===")
for en, de_text, fixed_px, width_px in overflow:
    print(f"  {en!r} -> {de_text!r} ({width_px}px > fixed {fixed_px}px)")
