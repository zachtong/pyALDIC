"""Build the desktop bundle for this platform, then package it for release.

    python tools/build_exe.py                 # build + package
    python tools/build_exe.py --installer     # Windows: also the installer
    python tools/build_exe.py --no-package    # build only (--no-zip works too)
    python tools/build_exe.py --clean         # discard previous build cache

Windows: ``dist-exe/pyALDIC/`` -> ``pyALDIC-Windows-Portable.zip`` and, with
``--installer`` (needs Inno Setup 6), ``pyALDIC-Windows-Setup.exe``.
macOS (Apple silicon): ``dist-exe/pyALDIC.app`` -> ``pyALDIC-macOS.dmg``.

The release names carry no version on purpose: README and course handouts link
to ``releases/latest/download/<name>``, which only resolves while the name stays
the same from one release to the next.

Output goes to ``dist-exe/``, not ``dist/``: that is the directory
``publish.yml`` runs ``twine check`` and ``gh release upload`` against, and a
500 MB bundle landing there would be uploaded to PyPI as if it were a wheel.

The build log is the most valuable output of a first build on new library
versions -- PyInstaller reports a missing hidden import as a WARNING and
carries on, producing a bundle that starts and then fails somewhere specific.
This script surfaces those lines rather than leaving them in the scrollback.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "pyaldic.spec"
ISS = ROOT / "packaging" / "pyaldic.iss"
DIST = ROOT / "dist-exe"
WORK = ROOT / "build-exe"
BUNDLE = DIST / "pyALDIC"
APP = DIST / "pyALDIC.app"
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

ZIP_NAME = "pyALDIC-Windows-Portable.zip"
INSTALLER_NAME = "pyALDIC-Windows-Setup.exe"
DMG_NAME = "pyALDIC-macOS.dmg"

# Put on the .dmg beside the app: the first launch of a downloaded copy stops
# at Gatekeeper, and this is where a student looks when it does.
MAC_FIRST_LAUNCH = """\
pyALDIC is free, open-source software that is not notarized by Apple, so the
first time you open it macOS stops it with a message such as
"Apple could not verify 'pyALDIC' is free of malware".

To open it -- once per installation:

  1. Drag pyALDIC into the Applications folder and double-click it there.
     When the message appears, click Done.
  2. Open System Settings > Privacy & Security, scroll down to the line about
     pyALDIC, click Open Anyway, and confirm with your password.
  3. pyALDIC opens. From then on it opens like any other app.

Installation help: https://github.com/zachtong/pyALDIC#installation
"""

# Lines worth reading back out of several thousand lines of build log.
INTERESTING = re.compile(
    r"^\d+ WARNING: (Hidden import .* not found|Library not found|"
    r"Cannot find |lib not found)|^\d+ ERROR:",
)


def _version() -> str:
    text = (ROOT / "src" / "al_dic" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if match is None:
        raise SystemExit("could not read __version__ from src/al_dic/__init__.py")
    return match.group(1)


def _check_environment() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed in this interpreter.\n"
            f"  {sys.executable} -m pip install -r packaging/requirements-build.txt"
        )
    if "conda" in sys.prefix.lower() or "anaconda" in sys.prefix.lower():
        print(
            f"note: building from a conda environment ({sys.prefix}).\n"
            "      This environment's DLL directories are put first on PATH "
            "for the build,\n"
            "      and the spec fails the build if anything is still resolved "
            "from outside\n"
            "      it. A clean venv remains the reference build environment.\n"
        )


def _dll_search_env() -> dict:
    """A copy of os.environ with this interpreter's own DLL directories first.

    PyInstaller resolves a binary dependency by searching PATH, so the build
    machine's PATH decides what goes into the bundle. Running an environment's
    python.exe without activating that environment is enough to poison it:
    Anaconda's base ``Library\bin`` stays on PATH while the environment's does
    not, so every conda-provided DLL resolves to the wrong build. That is not a
    theoretical concern -- it silently produced a bundle whose Qt could not
    load at all, and another whose colorbars vanished from every export.
    Windows only: macOS resolves libraries through their install names.
    """
    env = dict(os.environ)
    if not IS_WINDOWS:
        return env
    prefix = Path(sys.prefix)
    candidates = [
        prefix / "Library" / "bin",
        prefix / "Library" / "mingw-w64" / "bin",
        prefix / "Library" / "usr" / "bin",
        prefix / "DLLs",
        prefix,
    ]
    front = [str(p) for p in candidates if p.is_dir()]
    env["PATH"] = os.pathsep.join(front + [env.get("PATH", "")])
    return env


def _build(clean: bool) -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        str(SPEC),
    ]
    if clean:
        cmd.insert(3, "--clean")
    print("$", " ".join(cmd), "\n")

    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          env=_dll_search_env())
    log = (proc.stdout or "") + (proc.stderr or "")
    (WORK / "build.log").parent.mkdir(parents=True, exist_ok=True)
    (WORK / "build.log").write_text(log, encoding="utf-8")

    flagged = [line for line in log.splitlines() if INTERESTING.match(line)]
    if flagged:
        print("--- missing imports / libraries reported by PyInstaller ---")
        for line in flagged:
            print(" ", line)
        print()
    if proc.returncode != 0:
        print(log[-6000:], file=sys.stderr)
        raise SystemExit(f"PyInstaller failed with exit code {proc.returncode}")
    print(f"build log: {WORK / 'build.log'}")


def _report_size() -> None:
    root = APP if IS_MAC else BUNDLE
    files = [f for f in root.rglob("*") if f.is_file() and not f.is_symlink()]
    total = sum(f.stat().st_size for f in files)
    print(f"\nbundle: {root}")
    print(f"  {len(files)} files, {total / 1024 ** 2:.0f} MB uncompressed")
    for f in sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:8]:
        print(f"    {f.stat().st_size / 1024 ** 2:7.1f} MB  {f.relative_to(root)}")


def _zip() -> Path:
    out = DIST / ZIP_NAME
    if out.exists():
        out.unlink()
    print(f"\nzipping -> {out.name} ...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(BUNDLE.rglob("*")):
            if f.is_file():
                zf.write(f, Path("pyALDIC") / f.relative_to(BUNDLE))
    print(f"  {out.stat().st_size / 1024 ** 2:.0f} MB")
    return out


def _find_iscc() -> str | None:
    """Inno Setup 6's command-line compiler, wherever it was installed."""
    found = shutil.which("iscc")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base and (Path(base) / "Inno Setup 6" / "ISCC.exe").is_file():
            return str(Path(base) / "Inno Setup 6" / "ISCC.exe")
    return None


def _installer() -> Path:
    iscc = _find_iscc()
    if iscc is None:
        raise SystemExit(
            "Inno Setup 6 is not installed (ISCC.exe not found).\n"
            "  choco install innosetup   or   https://jrsoftware.org/isdl.php"
        )
    out = DIST / INSTALLER_NAME
    if out.exists():
        out.unlink()
    cmd = [iscc, "/Qp", f"/DAppVersion={_version()}", f"/DSourceDir={BUNDLE}",
           f"/DOutputDir={DIST}", str(ISS)]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not out.is_file():
        raise SystemExit(f"Inno Setup finished but {out} is missing")
    print(f"  {out.name}: {out.stat().st_size / 1024 ** 2:.0f} MB")
    return out


def _dmg() -> Path:
    """A compressed disk image: the app, an Applications link, first-launch help.

    ``ditto`` copies the bundle: it keeps the framework symlinks and the
    signature that a plain copy can break.
    """
    out = DIST / DMG_NAME
    stage = WORK / "dmg"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    subprocess.run(["ditto", str(APP), str(stage / "pyALDIC.app")], check=True)
    (stage / "Applications").symlink_to("/Applications")
    (stage / "If pyALDIC will not open.txt").write_text(
        MAC_FIRST_LAUNCH, encoding="utf-8")
    if out.exists():
        out.unlink()
    print(f"\nimaging -> {out.name} ...")
    subprocess.run(["hdiutil", "create", "-volname", "pyALDIC", "-srcfolder",
                    str(stage), "-ov", "-format", "UDZO", str(out)], check=True)
    subprocess.run(["hdiutil", "verify", str(out)], check=True)
    print(f"  {out.stat().st_size / 1024 ** 2:.0f} MB")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-package", "--no-zip", dest="no_package",
                        action="store_true",
                        help="build only: no zip, installer or disk image")
    parser.add_argument("--installer", action="store_true",
                        help="Windows: also build the Inno Setup installer")
    parser.add_argument("--clean", action="store_true",
                        help="discard PyInstaller's cached analysis first")
    args = parser.parse_args()
    if not (IS_WINDOWS or IS_MAC):
        raise SystemExit("desktop bundles are built on Windows and macOS only")

    _check_environment()
    if args.clean and DIST.exists():
        shutil.rmtree(DIST)
    _build(args.clean)
    _report_size()
    if not args.no_package:
        if IS_MAC:
            _dmg()
        else:
            _zip()
            if args.installer:
                _installer()
    print(f"\nRun it:  {APP / 'Contents' / 'MacOS' / 'pyALDIC' if IS_MAC else BUNDLE / 'pyALDIC.exe'}")


if __name__ == "__main__":
    main()
