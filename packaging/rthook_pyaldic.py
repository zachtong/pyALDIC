"""PyInstaller runtime hook: environment defaults for the frozen build.

Runs before ``al_dic`` -- or matplotlib, or numba -- is imported.

Deliberately absent: ``NUMBA_CACHE_DIR``. Under ``sys.frozen`` Numba's
``UserProvidedCacheLocator`` delegates to ``_SourceFileBackedLocatorMixin``,
which requires the function's source ``.py`` to exist on disk; inside a bundle
it never does, so the locator declines and the variable is ignored. The cache
always resolves to ``UserWideCacheLocator``. What matters instead is surviving
the case where *that* directory is unwritable, which is handled in
``al_dic._numba_compat``.
"""

import os
import sys
import tempfile

if getattr(sys, "frozen", False):
    # The same per-user base as al_dic.gui.app.user_data_dir(), which cannot
    # be imported this early: a platform's own place, never a bare folder in
    # the home directory.
    _home = os.path.expanduser("~") or tempfile.gettempdir()
    if sys.platform == "win32":
        _base = os.environ.get("LOCALAPPDATA") or _home
    elif sys.platform == "darwin":
        _base = os.path.join(_home, "Library", "Application Support")
    else:
        _base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            _home, ".local", "share")

    # Pin matplotlib's config/cache directory. Without a writable one it falls
    # back to a fresh temp dir per launch, which re-runs the full system font
    # scan every time the application starts.
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(_base, "pyALDIC", "mpl")
    )

    # Immunise against ambient matplotlib configuration on the target machine:
    # matplotlib_fname() probes ./matplotlibrc first, and a frozen app's
    # working directory is whatever launched it -- Explorer, a shortcut's
    # "Start in", or the folder of a double-clicked .aldic.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.pop("MATPLOTLIBRC", None)
