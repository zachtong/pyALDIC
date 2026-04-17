"""Smoke test: session menu actions resolve their imports.

Regression guard for the 838534f bug where ``Path`` was used in
``_on_save_session`` / ``_on_open_session`` but not imported at module
scope, crashing with NameError the first time the user clicked Save
Session.

These tests do NOT exercise the full QFileDialog flow (that needs a
display server); they just ensure the MainWindow entry points can be
called with the dialog bypassed, which proves all module-level names
resolve.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState


@pytest.fixture(autouse=True)
def reset_state():
    AppState._instance = None
    yield
    AppState._instance = None


def test_save_session_returns_silently_when_user_cancels(monkeypatch, tmp_path):
    """User cancels the Save dialog (empty string) -> no crash."""
    from al_dic.gui.app import MainWindow
    from PySide6.QtWidgets import QFileDialog

    # Mock the dialog to return empty (simulates user Cancel)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: ("", "")),
    )
    win = MainWindow()
    win._on_save_session()   # must not raise


def test_save_session_writes_file(monkeypatch, tmp_path):
    """Full happy path: user chooses a path -> file exists."""
    from al_dic.gui.app import MainWindow
    from PySide6.QtWidgets import QFileDialog

    out = tmp_path / "my.aldic.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(out), "")),
    )
    win = MainWindow()
    win._on_save_session()
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("{")


def test_open_session_returns_silently_when_user_cancels(monkeypatch):
    from al_dic.gui.app import MainWindow
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: ("", "")),
    )
    win = MainWindow()
    win._on_open_session()   # must not raise
