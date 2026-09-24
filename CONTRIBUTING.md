# Contributing to pyALDIC

Thank you for your interest in contributing to pyALDIC! This document provides guidelines for contributing to the project.

## Reporting Bugs

Please open an issue on [GitHub Issues](https://github.com/zachtong/pyALDIC/issues) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Python version, OS, and package versions (`pip list`)
- Sample images if relevant (or a minimal synthetic example)

## Suggesting Features

Open an issue with the label `enhancement` and describe:

- The problem your feature would solve
- How you envision it working
- Whether you're willing to help implement it

## Development Setup

```bash
# Clone the repository
git clone https://github.com/zachtong/pyALDIC.git
cd pyALDIC

# Install in development mode
pip install -e ".[dev]"

# Run the test suite
pytest
```

## Building the Desktop Bundles

CI builds and verifies them on every tag -- the Windows installer and
portable zip on `windows-latest`, the macOS disk image on `macos-14`
(Apple silicon) -- so you rarely need to. When you do, on Windows:

```bash
# A CLEAN venv, not conda -- see the warning below
python -m venv .venv-build
.venv-build\Scripts\activate
pip install -r packaging/requirements-build.txt
pip install -e . --no-deps

python tools/i18n.py compile     # the .qm catalogs are build products
python tools/build_exe.py        # -> dist-exe/pyALDIC/ and the portable zip
python tools/build_exe.py --installer   # ... and the installer (Inno Setup 6)
```

On a Mac the same `python tools/build_exe.py` produces `dist-exe/pyALDIC.app`
and `pyALDIC-macOS.dmg`; point `PYALDIC_FROZEN_EXE` at
`dist-exe/pyALDIC.app/Contents/MacOS/pyALDIC` to verify it.

Then verify it — and do verify it, because almost everything that goes wrong
here goes wrong quietly:

```bash
set PYALDIC_FROZEN_EXE=dist-exe\pyALDIC\pyALDIC-console.exe
pytest tests/test_frozen_bundle.py -v
```

The application guards eleven optional pieces behind `try/except ImportError`
or `Path.is_file()`. A bundle that has lost QtSvg, every icon, all seven
translation catalogs, the spin-box arrows and Numba acceleration still opens a
window that looks correct, so "it launched" proves very little. The checks live
in `src/al_dic/gui/self_test.py` and ship inside the bundle; add one there
whenever you add something that can fail silently.

**Why a clean environment matters.** PyInstaller resolves each binary
dependency by searching `PATH`, so any DLL your machine happens to have can be
baked into the bundle and shadow the one a user's machine would load. This is
not hypothetical. Anaconda's `icuuc.dll` exports versioned ICU symbols
(`ucnv_open_73`) where the Qt in the PySide6 wheel expects the unversioned ones
that ship in Windows' own ICU — collecting it made every PySide6 import fail
with "The specified procedure could not be found", and the application never
got as far as a window. A second build lost every colorbar to Anaconda's
`libexpat.dll` in the same way. The spec now refuses to build when it finds a
binary sourced from outside the build environment; if you hit that, fix the
environment rather than setting `PYALDIC_ALLOW_AMBIENT`.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure all tests pass: `pytest`
4. Keep your changes focused — one feature or fix per PR
5. Write a clear PR description explaining what and why

## Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings to public functions and classes
- Keep functions focused and under 50 lines where practical

## Testing

- Write tests for all new features and bug fixes
- Place tests in the appropriate `tests/` subdirectory
- Run the full suite before submitting: `pytest`
- Run parallel execution for speed: `pytest -n auto`

## Questions?

Open an issue or reach out to the maintainers. We're happy to help!
