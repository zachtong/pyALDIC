"""Prove a built bundle actually works, not merely that it starts.

    pyALDIC.exe --self-test <report.json>

A launch-only check is close to worthless here. The application guards eleven
optional pieces behind ``try/except ImportError`` or ``Path.is_file()``, so a
bundle that has lost QtSvg, every icon, all seven translation catalogs, the
spin-box arrows and Numba acceleration still opens a window and looks fine.
Each check below targets one of those silent fallbacks.

Results go to a JSON file rather than stdout so the windowed and console
executables behave identically, and the process exit code is non-zero if any
check fails.

This module is deliberately importable without a display: every Qt check
creates its own ``QApplication`` under whatever platform plugin is available.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# A directory name that is non-ASCII *and* contains a space, because both have
# independently broken image export in the past.
NON_ASCII_DIR = "试样 1"

SHIPPED_LANGS = ["en", "zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es"]

# One string per context that must come back translated in every non-source
# locale. Kept in step with tests/test_smoke_i18n_languages.py.
TRANSLATION_PROBE = ("RightSidebar", "Run DIC Analysis")


class CheckFailed(Exception):
    """A self-test check did not pass. The message is the report detail."""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_packaged_data() -> str:
    """Every data file the app resolves via ``Path(__file__).parent``."""
    import al_dic.gui.icons as icons_mod
    import al_dic.gui.theme as theme_mod
    import al_dic.i18n as i18n_mod

    qm_dir = Path(i18n_mod.__file__).parent / "compiled"
    qm = sorted(p.name for p in qm_dir.glob("*.qm")) if qm_dir.is_dir() else []
    if len(qm) != 7:
        raise CheckFailed(
            f"expected 7 .qm catalogs in {qm_dir}, found {len(qm)}: {qm}. "
            "The spec's datas destination must be 'al_dic/i18n/compiled'."
        )

    arrows_dir = Path(theme_mod.__file__).parent / "arrows"
    arrows = sorted(p.name for p in arrows_dir.glob("*.svg")) if arrows_dir.is_dir() else []
    if len(arrows) != 4:
        raise CheckFailed(
            f"expected 4 spin-box arrow SVGs in {arrows_dir}, found {arrows}"
        )

    icon_dir = Path(icons_mod.__file__).parent / "assets" / "icon"
    missing = [
        n for n in ("pyALDIC.ico", "pyALDIC-256.png", "pyALDIC.svg")
        if not (icon_dir / n).is_file()
    ]
    if missing:
        raise CheckFailed(f"missing icon assets in {icon_dir}: {missing}")

    return f"7 catalogs, 4 arrows, 3 icons"


def check_qt_and_icons() -> str:
    """QtSvg is present and the icon functions return real pixmaps."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(["pyALDIC-self-test"])
    from al_dic.gui import icons

    if not icons._HAS_SVG:
        raise CheckFailed(
            "PySide6.QtSvg missing: every toolbar icon degrades to an empty "
            "QIcon without raising"
        )
    app_icon = icons.icon_app()
    if app_icon.isNull() or not app_icon.availableSizes():
        raise CheckFailed("icon_app() produced a null QIcon")
    play = icons.icon_play()
    if play.isNull():
        raise CheckFailed("icon_play() produced a null QIcon")

    return f"QtSvg live, app icon sizes {[s.width() for s in app_icon.availableSizes()][:4]}"


def check_all_locales() -> str:
    """Every shipped locale loads *and* actually changes the text.

    ``LanguageManager.load`` returns True after silently falling back to
    English, so loading is not evidence of anything on its own.
    """
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["pyALDIC-self-test"])
    from al_dic.i18n import LanguageManager

    context, source = TRANSLATION_PROBE
    manager = LanguageManager(app)
    untranslated = []
    for lang in SHIPPED_LANGS:
        if not manager.load(lang):
            raise CheckFailed(f"LanguageManager.load({lang!r}) returned False")
        text = QCoreApplication.translate(context, source)
        if lang != "en" and text == source:
            untranslated.append(lang)
    manager.load("en")

    if untranslated:
        raise CheckFailed(
            f"locales loaded but did not translate {source!r}: {untranslated}. "
            "The .qm catalogs are missing or stale in the bundle."
        )
    return f"{len(SHIPPED_LANGS)} locales load and translate"


def check_numba() -> str:
    """Numba is present, caching is usable, and the kernels are real."""
    from al_dic._numba_compat import HAS_NUMBA, JIT_CACHE
    from al_dic.solver import numba_kernels as nk

    if not HAS_NUMBA:
        raise CheckFailed("numba is not importable; the solver falls back to Python")

    dispatcher = type(nk.icgn_6dof_parallel).__module__
    if not dispatcher.startswith("numba"):
        raise CheckFailed(
            f"icgn_6dof_parallel is {dispatcher}, i.e. the pass-through stub, "
            "not a compiled Dispatcher"
        )

    import numpy as np
    import numba

    # Force a parallel=True kernel to compile so the threading layer resolves.
    t0 = time.perf_counter()
    nk._cubic_weight(0.25)
    img = np.ascontiguousarray(np.random.default_rng(0).random((64, 64)))
    nk._bicubic_interp(img, 12.5, 12.5, 64, 64)
    cold = time.perf_counter() - t0

    threads = numba.get_num_threads()
    # numba.threading_layer() only resolves once a parallel=True kernel has
    # run, which happens for real in check_mini_dic -- so it is reported there.
    note = f"cache={JIT_CACHE}, threads={threads}, cold JIT {cold:.1f}s"
    if not JIT_CACHE:
        # Not fatal: correctness is unaffected, every launch just recompiles.
        note += " (WARNING: JIT cache unusable, every launch pays full compile)"
    return note


def check_mini_dic() -> str:
    """A real correlation end to end: llvmlite, threading and scipy together."""
    import numpy as np
    from scipy.ndimage import gaussian_filter, shift as ndshift

    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange
    from al_dic.core.pipeline import run_aldic

    size, true_u, true_v = 192, 1.5, -0.75
    rng = np.random.default_rng(42)
    ref = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0, mode="nearest")
    ref -= ref.min()
    ref = 20.0 + 215.0 * (ref / ref.max())
    # A rigid translation needs no fixed-point warp: the deformation gradient
    # is the identity, so the Eulerian pull warp is already the Lagrangian one.
    # scipy's shift=(sv, su) gives out[y, x] = ref[y - sv, x - su], i.e. a
    # feature moves by (+su, +sv) -- the same convention as the test suite's
    # apply_displacement, warped(y, x) = ref(y - v, x - u).
    deformed = ndshift(ref, (true_v, true_u), order=3, mode="nearest")

    para = dicpara_default(
        winsize=32,
        winstepsize=16,
        gridxy_roi_range=GridxyROIRange(gridx=(0, size - 1), gridy=(0, size - 1)),
        admm_max_iter=2,
        admm_tol=1e-2,
        icgn_max_iter=30,
        tol=1e-2,
        show_plots=False,
    )
    masks = [np.ones((size, size), np.uint8), np.ones((size, size), np.uint8)]
    result = run_aldic(para, [ref, deformed], masks, compute_strain=False)

    if not result.result_disp:
        raise CheckFailed("run_aldic returned no displacement result")
    frame = result.result_disp[0]
    disp = frame.U_accum if frame.U_accum is not None else frame.U
    disp = np.asarray(disp, float)
    if not np.isfinite(disp).all():
        raise CheckFailed(
            f"{int((~np.isfinite(disp)).sum())} of {disp.size} displacement "
            "components are NaN or inf"
        )
    u_err = float(np.abs(disp[0::2].mean() - true_u))
    v_err = float(np.abs(disp[1::2].mean() - true_v))
    if max(u_err, v_err) > 0.10:
        raise CheckFailed(
            f"recovered shift is wrong: mean |u error| {u_err:.3f} px, "
            f"|v error| {v_err:.3f} px against a 0.10 px tolerance"
        )
    # A full pipeline run has now exercised the parallel kernels, so the
    # threading layer is finally resolvable. A silent demotion to 'workqueue'
    # is what a missing vcomp140.dll looks like.
    import numba

    try:
        layer = numba.threading_layer()
    except Exception:
        layer = "unresolved"
    return (
        f"{disp.size // 2} nodes, |u err| {u_err:.3f} px, "
        f"|v err| {v_err:.3f} px, threading layer {layer}"
    )


def check_image_io_non_ascii() -> str:
    """Image encode/decode survives a non-ASCII path with a space in it."""
    import cv2
    import numpy as np

    root = Path(tempfile.mkdtemp()) / NON_ASCII_DIR
    root.mkdir(parents=True)
    img = (np.random.default_rng(1).random((48, 64)) * 255).astype(np.uint8)

    written = []
    for ext in (".png", ".jpg"):
        out = root / f"结果{ext}"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise CheckFailed(f"cv2.imencode failed for {ext}")
        buf.tofile(str(out))
        if not out.is_file() or out.stat().st_size == 0:
            raise CheckFailed(f"nothing was written to {out}")
        back = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_GRAYSCALE)
        if back is None or back.shape != img.shape:
            raise CheckFailed(f"could not read {out} back")
        written.append(f"{ext} {out.stat().st_size}B")
    return ", ".join(written)


def check_video_and_gif() -> str:
    """MP4 needs OpenCV's FFmpeg DLL; GIF needs the imageio Pillow plugin."""
    import cv2
    import numpy as np

    root = Path(tempfile.mkdtemp()) / NON_ASCII_DIR
    root.mkdir(parents=True)
    frames = [
        (np.full((48, 64, 3), v, np.uint8)) for v in (40, 120, 200)
    ]

    mp4 = root / "动画.mp4"
    writer = cv2.VideoWriter(
        str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        raise CheckFailed(
            f"cv2.VideoWriter could not open {mp4}: opencv_videoio_ffmpeg*.dll "
            "is probably missing from the bundle"
        )
    for f in frames:
        writer.write(f)
    writer.release()
    if not mp4.is_file() or mp4.stat().st_size == 0:
        raise CheckFailed("MP4 encoder opened but wrote nothing")

    import imageio

    gif = root / "动画.gif"
    with imageio.get_writer(str(gif), format="GIF", mode="I", duration=0.1) as w:
        for f in frames:
            w.append_data(f[:, :, ::-1])
    if not gif.is_file() or gif.stat().st_size == 0:
        raise CheckFailed("GIF writer produced nothing; PIL plugin missing?")

    return f"mp4 {mp4.stat().st_size}B, gif {gif.stat().st_size}B"


def check_colorbar() -> str:
    """matplotlib renders: catches a missing mpl-data or Agg backend.

    ``attach_colorbar`` swallows every exception and returns the input image
    untouched, so a broken matplotlib shows up as exports that quietly have no
    colorbar rather than as an error.
    """
    import numpy as np

    from al_dic.export.colorbar import ColorbarStyle, _render_bar, attach_colorbar

    # Render through the private entry point first. attach_colorbar catches
    # everything and hands back the input, so going straight at the renderer is
    # the only way to learn what actually went wrong.
    try:
        _render_bar(80, 40, "vertical", "jet", 0.0, 1.0, "test", 9.0,
                    "black", 100, "sans-serif")
    except Exception as exc:
        raise CheckFailed(
            f"matplotlib colorbar rendering raised {type(exc).__name__}: {exc}"
        ) from exc

    img = np.full((80, 120, 3), 90, np.uint8)
    style = ColorbarStyle(position="right")
    out = np.asarray(attach_colorbar(img, style, "jet", 0.0, 1.0, "test", 100))
    if out.shape == img.shape and np.array_equal(out, img):
        raise CheckFailed(
            "the renderer works but attach_colorbar returned the image "
            "unchanged, so it failed somewhere after _render_bar"
        )
    return f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}"


def check_hashing() -> str:
    """hashlib still produces correct digests without OpenSSL's accelerator.

    The bundle excludes ``_hashlib``, so ``hashlib`` falls back to the standard
    library's built-in ``_sha256`` / ``_sha1`` implementations. Session save
    and load depend on this: ``io.session_serialize._array_key`` hashes array
    contents with sha256 to deduplicate them, and a wrong digest there would
    corrupt a saved session rather than raise.
    """
    import hashlib

    import numpy as np

    from al_dic.io.session_serialize import _array_key

    # Compare the two constructors rather than a hardcoded constant: they
    # reach the implementation by different routes, so agreement between them
    # is the evidence that the fallback is wired up correctly.
    direct = hashlib.sha256(b"pyALDIC").hexdigest()
    by_name = hashlib.new("sha256", b"pyALDIC").hexdigest()
    if direct != by_name:
        raise CheckFailed(
            f"hashlib.sha256 and hashlib.new('sha256') disagree: "
            f"{direct} vs {by_name}"
        )
    if len(direct) != 64 or len(hashlib.sha1(b"pyALDIC").hexdigest()) != 40:
        raise CheckFailed(f"digest lengths are wrong: sha256={direct}")

    # And the call site that a wrong digest would silently corrupt.
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    key_a, key_b = _array_key(arr), _array_key(arr.copy())
    if key_a != key_b:
        raise CheckFailed("session array hashing is not deterministic")
    if key_a == _array_key(arr + 1.0):
        raise CheckFailed("session array hashing collides on different data")

    return f"sha256/sha1 available, session array key {key_a[:20]}..."


def check_embedded_chart() -> str:
    """The analysis tab's matplotlib canvas imports and renders.

    Charts are the one place the application needs matplotlib's *Qt* backend
    rather than headless Agg. A bundle that dropped it would open, show every
    field correctly, and fail only when someone opened the analysis tab -- so
    this is checked here rather than trusted to the spec.
    """
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(["pyALDIC-self-test"])

    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from types import SimpleNamespace

    import numpy as np

    from al_dic.analysis.engine import AnalysisEngine
    from al_dic.analysis.probes import LineGeom, Probe
    from al_dic.core.data_structures import DICMesh, FrameResult, PipelineResult
    from al_dic.gui.widgets.mpl_chart import MplChart

    chart = MplChart()
    if not isinstance(chart._canvas, FigureCanvasQTAgg):
        raise CheckFailed("the chart is not backed by matplotlib's Qt canvas")

    # Read a real curve through the engine, so the bundle's triangulation and
    # sampling path runs too -- not only the drawing of a hand-made series.
    xs = np.arange(0.0, 40.0, 4.0)
    gx, gy = np.meshgrid(xs, xs)
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    frames = []
    for f in (1, 2, 3):
        U = np.empty(2 * len(nodes))
        U[0::2], U[1::2] = 0.01 * f * nodes[:, 0], 0.0
        frames.append(FrameResult(U=U, U_accum=U))
    result = PipelineResult(
        dic_para=SimpleNamespace(winstepsize=4, img_size=(40, 40)),
        dic_mesh=DICMesh(coordinates_fem=nodes,
                         elements_fem=np.zeros((0, 4), np.int64)),
        result_disp=frames, result_def_grad=[], result_strain=[],
        result_fe_mesh_each_frame=[],
    )
    gauge = Probe(id=1, kind="line", geometry=LineGeom(4.0, 20.0, 32.0, 20.0),
                  label="E1", color="#ef4444")
    ts = AnalysisEngine(result).series(gauge, None, "strain")
    if not np.allclose(ts.values, [0.0, 0.01, 0.02, 0.03], atol=1e-9):
        raise CheckFailed(f"the extensometer read {ts.values.tolist()}")

    chart.plot_series([("E1", "#ef4444", ts)], y_label="strain")
    chart.figure.canvas.draw()
    width, height = chart.figure.canvas.get_width_height()
    if width <= 0 or height <= 0:
        raise CheckFailed(f"the chart rendered at {width}x{height}")
    return f"engine + QtAgg canvas render at {width}x{height}"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("packaged_data", check_packaged_data),
    ("qt_and_icons", check_qt_and_icons),
    ("all_locales", check_all_locales),
    ("numba", check_numba),
    ("hashing", check_hashing),
    ("image_io_non_ascii", check_image_io_non_ascii),
    ("video_and_gif", check_video_and_gif),
    ("colorbar", check_colorbar),
    ("embedded_chart", check_embedded_chart),
    ("mini_dic", check_mini_dic),  # slowest; last so failures surface sooner
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_self_test(report_path: str | Path) -> int:
    """Run every check, write the JSON report, return a process exit code."""
    results: dict[str, Any] = {}
    for name, fn in CHECKS:
        started = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except CheckFailed as exc:
            detail, ok = str(exc), False
        except Exception:
            detail, ok = traceback.format_exc(), False
        results[name] = {
            "ok": ok,
            "detail": detail,
            "seconds": round(time.perf_counter() - started, 2),
        }
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if all(r["ok"] for r in results.values()) else 1


def main_self_test(argv: list[str]) -> int:
    """``--self-test <report.json>`` entry point."""
    try:
        report = argv[argv.index("--self-test") + 1]
    except (ValueError, IndexError):
        print("usage: pyALDIC --self-test <report.json>", file=sys.stderr)
        return 2
    return run_self_test(report)
