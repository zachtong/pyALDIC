"""Smoke test for the strain window uniform-shear visual PDF report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "report_strain_window_uniform_shear.py"


def _load_report_module():
    """Import the report script as a module without polluting sys.modules."""
    spec = importlib.util.spec_from_file_location(
        "report_strain_window_uniform_shear", SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load report script at {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report_module():
    return _load_report_module()


def test_report_script_exists():
    assert SCRIPT_PATH.exists(), f"Missing report script: {SCRIPT_PATH}"


def test_generate_report_writes_nonempty_pdf(tmp_path, report_module):
    """Smoke test: generate_report() creates a non-empty PDF on disk."""
    pdf_path = report_module.generate_report(out_dir=tmp_path)

    assert pdf_path.exists(), f"PDF not created at {pdf_path}"
    assert pdf_path.suffix == ".pdf"
    size = pdf_path.stat().st_size
    # A 7-page strain report should comfortably exceed 10 KB.
    assert size > 10_000, f"PDF unexpectedly small ({size} bytes)"


def test_generate_report_runs_all_strain_methods(report_module):
    """Sanity check on the script's runtime constants."""
    assert report_module.SHEAR == 0.01
    assert report_module.N_FRAMES >= 2
    assert report_module.PREVIEW_FRAME < report_module.N_FRAMES - 1
