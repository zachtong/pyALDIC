"""Every message the pipeline controller shows comes out in the GUI language.

The run's error dialogs (``fatal_error``: a run that cannot start, a run
that failed, the Starting Points errors) and its console lines were English
in every language. Each scenario here drives the real code path --
``PipelineController.start()`` with a failing precondition,
``_on_finished`` with a failing result, ``PipelineWorker.run()`` with
``run_aldic`` raising -- and captures what the user would see.

A message that names a control names it as the GUI shows it: the label is
the widget's own translation, so a German user looking for
"ANFANGSSCHÄTZUNG" finds it. Exception text from the core stays as it is
(it is the console detail); the sentences around it are translated.
"""

from __future__ import annotations

import contextlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from al_dic.gui.app_state import AppState, SeedRecord
from al_dic.gui.controllers import pipeline_controller as pc_module
from al_dic.gui.controllers.pipeline_controller import (
    PipelineController,
    PipelineWorker,
)
from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    PartialResultError,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedQualityError,
    SeedWarpFailure,
)

LANGS = ("zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es")
TS_DIR = Path(__file__).resolve().parents[2] / "src" / "al_dic" / "i18n" / "source"

#: Exception text from the core: shown as it is, never translated.
RAW = "raw core detail 9031"
#: Frame a PartialResultError says the run stopped at (1-based).
STOP_FRAME = 7
#: A Region of Interest 9 x 9 px with a Subset Step of 16 (needs 32 x 32).
SMALL_ROI = (9, 16, 32)

# GUI labels the messages name: (translation context, source) of the widget
# that shows them.
IMAGES = ("LeftSidebar", "IMAGES")
ROI_SECTION = ("LeftSidebar", "REGION OF INTEREST")
INITIAL_GUESS = ("LeftSidebar", "INITIAL GUESS")
PARAMETERS = ("LeftSidebar", "PARAMETERS")
PLACE = ("InitGuessWidget", "Place Starting Points")
AUTO_PLACE = ("InitGuessWidget", "Auto-place")
CLEAR_POINTS = ("InitGuessWidget", "Clear")
FFT = ("InitGuessWidget", "FFT (cross-correlation)")
SUBSET_SIZE = ("ParamPanel", "Subset Size")
SUBSET_STEP = ("ParamPanel", "Subset Step")
BATCH_IMPORT = ("ROIToolbar", "Batch Import")
ALL_LABELS = (IMAGES, ROI_SECTION, INITIAL_GUESS, PARAMETERS, PLACE,
              AUTO_PLACE, CLEAR_POINTS, FFT, SUBSET_SIZE, SUBSET_STEP,
              BATCH_IMPORT)


@dataclass(frozen=True)
class Shown:
    """A dialog: what ``fatal_error`` carried."""

    title: str
    text: str


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _fresh_state(qapp):
    AppState._instance = None
    yield
    AppState._instance = None


@contextlib.contextmanager
def _language(lang: str):
    from al_dic.i18n import LanguageManager

    mgr = LanguageManager(QApplication.instance())
    assert mgr.load(lang)
    try:
        yield
    finally:
        mgr.load("en")


# --- Scenarios -----------------------------------------------------------------


def _state(**fields) -> AppState:
    """A fresh AppState ready to run but for what ``fields`` set."""
    AppState._instance = None
    st = AppState.instance()
    st.image_files = [f"/fake/img{i}.tif" for i in range(3)]
    st.subset_size = 40
    st.subset_step = 16
    st.search_range = 10
    st.tracking_mode = "accumulative"
    st.init_guess_mode = "previous"
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 8:56] = True
    st.per_frame_rois = {0: mask}
    for name, value in fields.items():
        setattr(st, name, value)
    return st


def _start(st: AppState) -> Shown:
    shown: list[Shown] = []
    st.fatal_error.connect(lambda t, m: shown.append(Shown(t, m)))
    PipelineController(st, None).start()
    (one,) = shown
    return one


def _small_roi() -> dict:
    side, _step, _need = SMALL_ROI
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:10 + side + 1, 10:10 + side + 1] = True
    return {"per_frame_rois": {0: mask}}


def _start_failure(monkeypatch) -> Shown:
    def fail(**_kwargs):
        raise ValueError(RAW)

    monkeypatch.setattr(pc_module, "dicpara_default", fail)
    return _start(_state())


def _results_failure(monkeypatch) -> Shown:
    st = _state()
    shown: list[Shown] = []
    st.fatal_error.connect(lambda t, m: shown.append(Shown(t, m)))

    def fail(_result):
        raise ValueError(RAW)

    monkeypatch.setattr(st, "set_results", fail)
    PipelineController(st, None)._on_finished(SimpleNamespace(result_disp=[0]))
    (one,) = shown
    return one


class _Provider:
    shape = (8, 8)

    def __len__(self) -> int:
        return 2


def _worker(monkeypatch, exc: Exception) -> tuple[Shown, list[str]]:
    """The dialog and the error console lines of a run that raised ``exc``."""
    def fail(**_kwargs):
        raise exc

    monkeypatch.setattr(pc_module, "run_aldic", fail)
    para = SimpleNamespace(winsize=32, winstepsize=16,
                           size_of_fft_search_region=10,
                           reference_mode="accumulative")
    worker = PipelineWorker(para, _Provider(), [np.ones((8, 8))])
    shown: list[Shown] = []
    errors: list[str] = []
    worker.fatal_error.connect(lambda t, m: shown.append(Shown(t, m)))
    worker.log.connect(lambda m, lvl: errors.append(m) if lvl == "error" else None)
    worker.run()
    (one,) = shown
    return one, errors


def _seed_failure(exc: Exception):
    return lambda mp: _worker(mp, exc)[0]


#: name -> (scenario, labels the text names, numbers it shows, raw shown).
SCENARIOS = {
    "too_few_images": (lambda mp: _start(_state(image_files=[])),
                       (IMAGES,), (), False),
    "no_roi": (lambda mp: _start(_state(per_frame_rois={})),
               (ROI_SECTION,), (), False),
    "empty_roi": (lambda mp: _start(_state(
        per_frame_rois={0: np.zeros((64, 64), dtype=bool)})),
        (ROI_SECTION,), (), False),
    "roi_too_small": (lambda mp: _start(_state(**_small_roi())),
                      (SUBSET_STEP,), SMALL_ROI, False),
    "seeds_required": (lambda mp: _start(_state(
        init_guess_mode="seed_propagation", seeds=[])),
        (PLACE, AUTO_PLACE, INITIAL_GUESS), (), False),
    "invalid_seeds": (lambda mp: _start(_state(
        init_guess_mode="seed_propagation",
        seeds=[SeedRecord(node_idx=0, region_id=-1)])),
        (CLEAR_POINTS, INITIAL_GUESS), (), False),
    "failed_to_start": (_start_failure, (), (), True),
    "failed_to_process": (_results_failure, (), (), True),
    "runtime_error": (lambda mp: _worker(mp, RuntimeError(RAW))[0],
                      (), (), True),
    "unexpected_error": (lambda mp: _worker(mp, ValueError(RAW))[0],
                         (), (), True),
    "seed_ncc": (_seed_failure(SeedNCCBelowThreshold(RAW)),
                 (FFT, INITIAL_GUESS), (), True),
    "seed_icgn": (_seed_failure(SeedICGNDiverged(RAW)),
                  (SUBSET_SIZE, PARAMETERS, FFT, INITIAL_GUESS), (), True),
    "seed_quality": (_seed_failure(SeedQualityError(RAW)),
                     (FFT, INITIAL_GUESS), (), True),
    "seed_warp": (_seed_failure(SeedWarpFailure(RAW)),
                  (BATCH_IMPORT,), (), True),
    "seed_missing": (_seed_failure(MissingSeedForRegion(RAW)),
                     (PLACE, INITIAL_GUESS), (), True),
    "seed_partial": (_seed_failure(PartialResultError(
        RAW, partial_result=None, stopped_at_frame=STOP_FRAME)),
        (FFT, INITIAL_GUESS), (STOP_FRAME,), True),
    "seed_other": (_seed_failure(SeedPropagationError(RAW)), (), (), True),
}


def _shown(monkeypatch, name: str, lang: str) -> Shown:
    scenario = SCENARIOS[name][0]
    with _language(lang):
        return scenario(monkeypatch)


def _label(label: tuple[str, str]) -> str:
    """The label as its widget shows it in the loaded language."""
    return QCoreApplication.translate(*label)


def _sentences(text: str) -> list[str]:
    """The English lines of ``text`` worth comparing: neither the raw core
    detail nor a bare label."""
    return [
        line.strip() for line in text.replace(RAW, "").splitlines()
        if len(line.split()) >= 3
    ]


# --- Tests ---------------------------------------------------------------------


@pytest.mark.parametrize("lang", LANGS)
def test_dialogs_are_translated(monkeypatch, lang):
    bad = []
    for name, (_run, _labels, _numbers, raw) in SCENARIOS.items():
        english = _shown(monkeypatch, name, "en")
        local = _shown(monkeypatch, name, lang)
        if local.title == english.title:
            bad.append(f"{name}: title {local.title!r}")
        bad += [f"{name}: English {line!r}" for line in _sentences(english.text)
                if line in local.text]
        if re.search(r"%(\d|n)", local.title + local.text):
            bad.append(f"{name}: placeholder left in {local.text!r}")
        if raw and RAW not in local.text:
            bad.append(f"{name}: core detail missing from {local.text!r}")
    assert not bad, f"[{lang}]\n" + "\n".join(bad)


@pytest.mark.parametrize("lang", ("en", *LANGS))
def test_dialogs_name_controls_as_the_gui_shows_them(monkeypatch, lang):
    """Section names, buttons and fields appear exactly as their widgets
    label them in ``lang``; counts and frame numbers are filled in."""
    bad = []
    for name, (_run, labels, numbers, _raw) in SCENARIOS.items():
        local = _shown(monkeypatch, name, lang)
        with _language(lang):
            expected = [_label(label) for label in labels]
        bad += [f"{name}: no {label!r} in {local.text!r}"
                for label in expected if label not in local.text]
        bad += [f"{name}: no {n} in {local.text!r}" for n in numbers
                if not re.search(rf"(?<!\d){n}(?!\d)", local.text)]
    assert not bad, f"[{lang}]\n" + "\n".join(bad)


def test_named_controls_are_still_shown_by_their_widgets():
    """Each label is a message its widget extracts: renaming the control
    without updating the messages fails here."""
    tree = ET.parse(TS_DIR / "al_dic_de.ts")
    catalog = {
        (context.findtext("name"), message.findtext("source"))
        for context in tree.getroot().iter("context")
        for message in context.iter("message")
    }
    missing = [label for label in ALL_LABELS if label not in catalog]
    assert not missing, missing


def _outside_translation(text: str) -> str:
    """``text`` without the core detail and without every ⟦…⟧ the
    pseudo-locale wrapped (innermost first: labels sit inside sentences)."""
    text = text.replace(RAW, "")
    while True:
        stripped = re.sub("⟦[^⟦⟧]*⟧", "", text)
        if stripped == text:
            return text
        text = stripped


def test_every_piece_goes_through_the_translator(monkeypatch):
    bad = []
    for name in SCENARIOS:
        shown = _shown(monkeypatch, name, "pseudo")
        for piece in (shown.title, shown.text):
            rest = _outside_translation(piece)
            if re.search(r"[A-Za-z]", rest):
                bad.append(f"{name}: not translated: {rest!r} in {piece!r}")
    assert not bad, "\n".join(bad)


# --- Console lines ---------------------------------------------------------------


class _IdleWorker:
    """Stands in for PipelineWorker: records nothing, never runs."""

    class _Signal:
        def connect(self, *_args) -> None:
            pass

    def __init__(self, *_args, **_kwargs) -> None:
        self.progress = self.log = self.finished_result = self._Signal()
        self.fatal_error = self._Signal()

    def start(self) -> None:
        pass


def _start_lines(monkeypatch, lang: str) -> list[str]:
    """Console lines of a run that starts (accumulative, refined mesh)."""
    monkeypatch.setattr(pc_module, "PipelineWorker", _IdleWorker)
    with _language(lang):
        st = _state(refine_inner=True)
        lines: list[str] = []
        st.log_message.connect(lambda m, _lvl: lines.append(m))
        PipelineController(st, None).start()
    return lines


def _canceled_lines(monkeypatch, lang: str) -> list[str]:
    """Warnings of an incremental run whose missing-region question was
    answered No."""
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    with _language(lang):
        st = _state(tracking_mode="incremental", inc_ref_mode="every_frame")
        lines: list[str] = []
        st.log_message.connect(
            lambda m, lvl: lines.append(m) if lvl == "warn" else None)
        PipelineController(st, None).start()
    return lines


def _error_lines(monkeypatch, lang: str) -> list[str]:
    with _language(lang):
        return _worker(monkeypatch, ValueError(RAW))[1][:1]


@pytest.mark.parametrize("lang", LANGS)
def test_console_lines_are_translated(monkeypatch, lang):
    """What start() logs on the way to the worker and the line a canceled
    run leaves; key=value dumps inside them stay literal."""
    bad = []
    for lines_of in (_start_lines, _canceled_lines):
        english, local = lines_of(monkeypatch, "en"), lines_of(monkeypatch, lang)
        assert len(local) == len(english) > 0, (lines_of.__name__, local)
        bad += [f"{lines_of.__name__}: {line!r}"
                for line, source in zip(local, english) if line == source]
    assert not bad, f"[{lang}] untranslated:\n" + "\n".join(bad)


def test_console_lines_go_through_the_translator(monkeypatch):
    """Pseudo-locale: every console line, the worker's error line too
    (Spanish "Error: %1" reads like the English), is a translated
    sentence."""
    lines = [line
             for lines_of in (_start_lines, _canceled_lines, _error_lines)
             for line in lines_of(monkeypatch, "pseudo")]
    bad = [line for line in lines
           if not (line.startswith("⟦") and line.endswith("⟧"))]
    assert lines and not bad, bad


def test_canceled_run_line_matches_the_dialog(monkeypatch):
    """The cancel line says what the dialog asked: frame 1's region is
    reused, not "inherited"."""
    lines = _canceled_lines(monkeypatch, "en")
    assert lines and "frame 1's region" in lines[0], lines
    assert "inherit" not in lines[0].lower(), lines
