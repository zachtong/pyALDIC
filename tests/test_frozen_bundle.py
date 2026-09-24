"""Drive a built PyInstaller bundle from outside and assert it really works.

Opt in by pointing ``PYALDIC_FROZEN_EXE`` at the built executable::

    python tools/build_exe.py --no-zip
    set PYALDIC_FROZEN_EXE=dist-exe\\pyALDIC\\pyALDIC-console.exe
    pytest tests/test_frozen_bundle.py -v

Skipped otherwise, so an ordinary ``pytest`` run on a fresh clone is unaffected.

The bundle must be driven as a separate process: importing ``al_dic`` from the
test process would exercise the development install, which is exactly the thing
these tests are not about. The heavy lifting lives in
``al_dic.gui.self_test``, which ships inside the bundle; this file is the
harness that runs it under hostile conditions and reads the verdict.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.frozen

EXE_ENV = "PYALDIC_FROZEN_EXE"
TIMEOUT = 900


@pytest.fixture(scope="session")
def frozen_exe() -> Path:
    raw = os.environ.get(EXE_ENV)
    if not raw:
        pytest.skip(f"{EXE_ENV} is not set")
    # Absolute, because the bundle is run with cwd= a temp directory: POSIX
    # resolves a relative executable path against the child's new working
    # directory (Windows against the parent's), so a relative path that
    # passes is_file() here is "not found" there.
    exe = Path(raw).resolve()
    if not exe.is_file():
        pytest.fail(f"{EXE_ENV} points at {exe}, which does not exist")
    return exe


def _run_self_test(exe: Path, workdir: Path) -> tuple[int, dict, str]:
    """Run the bundle's self-test in *workdir* and return (code, report, output)."""
    report = workdir / "报告 report.json"  # non-ASCII and a space, deliberately
    proc = subprocess.run(
        [str(exe), "--self-test", str(report)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(workdir), timeout=TIMEOUT,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    data = {}
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
    return proc.returncode, data, output


def _assert_all_ok(code: int, data: dict, output: str) -> None:
    assert data, f"the bundle wrote no report; it died early.\n{output[-4000:]}"
    failed = {k: v["detail"] for k, v in data.items() if not v["ok"]}
    assert not failed, "checks failed:\n" + "\n".join(
        f"  {k}: {detail}" for k, detail in failed.items()
    )
    assert code == 0, f"self-test exited {code} with no failed check\n{output[-2000:]}"


def test_bundle_self_test_passes(frozen_exe, tmp_path):
    """Every packaged feature works, run from an ordinary temp directory."""
    code, data, output = _run_self_test(frozen_exe, tmp_path)
    _assert_all_ok(code, data, output)


def test_bundle_works_from_non_ascii_cwd(frozen_exe, tmp_path):
    """The working directory is whatever launched the exe, not the app folder.

    Explorer, a shortcut's "Start in", or a double-clicked .aldic can each hand
    the process a directory with a space and non-ASCII characters in it.
    """
    workdir = tmp_path / "试样 1" / "结果"
    workdir.mkdir(parents=True)
    code, data, output = _run_self_test(frozen_exe, workdir)
    _assert_all_ok(code, data, output)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bundle")
def test_bundle_survives_unusable_jit_cache(frozen_exe, tmp_path):
    """A blocked Numba cache must degrade, not kill the process.

    ``@njit(cache=True)`` establishes its cache locator at decoration time and
    raises RuntimeError -- not ImportError -- when it cannot. In a frozen build
    the only locator left is the user-wide one, so an unwritable
    ``%LOCALAPPDATA%\\Numba`` used to take ``import al_dic`` down with it, and
    a windowed build vanished with no window and nothing logged.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        pytest.skip("LOCALAPPDATA is not set")
    cache_root = Path(local) / "Numba"
    backup = Path(local) / "Numba.pytest-backup"
    if backup.exists():
        pytest.skip(f"stale backup at {backup}")

    moved = False
    try:
        if cache_root.exists():
            shutil.move(str(cache_root), str(backup))
            moved = True
        # A regular file where the directory must go: os.makedirs then raises
        # NotADirectoryError, the same OSError a locked-down profile gives.
        cache_root.write_text("not a directory", encoding="utf-8")

        code, data, output = _run_self_test(frozen_exe, tmp_path)
        _assert_all_ok(code, data, output)
        assert "cache=False" in data["numba"]["detail"], (
            "the cache was blocked but the bundle still reported it as usable: "
            f"{data['numba']['detail']}"
        )
    finally:
        if cache_root.is_file():
            cache_root.unlink()
        if moved and backup.exists():
            shutil.move(str(backup), str(cache_root))
