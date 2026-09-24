"""Where the frozen app keeps its logs: each platform's own place.

It used to be ``%LOCALAPPDATA%`` or else the home directory, which on macOS
put a bare ``pyALDIC`` folder in the user's home. packaging/rthook_pyaldic.py
mirrors this choice for matplotlib's cache and must be kept in step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from al_dic.gui.app import user_data_dir


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
    for var in ("LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_windows_uses_localappdata(home, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(home / "Local"))
    assert user_data_dir() == home / "Local" / "pyALDIC"


def test_macos_uses_application_support(home, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert user_data_dir() == home / "Library" / "Application Support" / "pyALDIC"


@pytest.mark.parametrize("xdg", [None, "xdg-data"])
def test_linux_follows_xdg(home, monkeypatch, xdg):
    monkeypatch.setattr(sys, "platform", "linux")
    if xdg:
        monkeypatch.setenv("XDG_DATA_HOME", str(home / xdg))
        assert user_data_dir() == home / xdg / "pyALDIC"
    else:
        assert user_data_dir() == home / ".local" / "share" / "pyALDIC"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_never_a_bare_folder_in_home(home, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    assert user_data_dir().parent != Path(home)
