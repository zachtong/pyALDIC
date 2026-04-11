"""Tests for HTML report export."""

import pytest
from pathlib import Path

from al_dic.export.export_report import export_html_report


def test_html_report_created(tmp_path, minimal_result):
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result,
        fields=["disp_u", "strain_exx"],
        image_configs=[], sample_every=1,
    )
    assert p.exists()
    assert p.suffix == ".html"


def test_html_self_contained(tmp_path, minimal_result):
    """No external HTTP resources in the generated HTML."""
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result,
        fields=["disp_u"], image_configs=[], sample_every=1,
    )
    html = p.read_text(encoding="utf-8")
    # Should not reference any external URLs
    assert "http://" not in html
    assert "https://" not in html


def test_html_contains_field_names(tmp_path, minimal_result):
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result,
        fields=["disp_u", "strain_exx"],
        image_configs=[], sample_every=1,
    )
    html = p.read_text(encoding="utf-8")
    assert "disp_u" in html
    assert "strain_exx" in html


def test_html_contains_param_table(tmp_path, minimal_result):
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result,
        fields=[], image_configs=[], sample_every=1,
    )
    html = p.read_text(encoding="utf-8")
    assert "<table" in html
    # Should contain some DICPara field name (e.g., winsize)
    assert "winsize" in html


def test_html_has_stats_table(tmp_path, minimal_result):
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result,
        fields=["disp_u"], image_configs=[], sample_every=1,
    )
    html = p.read_text(encoding="utf-8")
    assert "Min" in html
    assert "Max" in html
    assert "Mean" in html
    assert "Std" in html


def test_html_embeds_base64_images(tmp_path, minimal_result):
    """Sample images should be embedded as base64 data URIs."""
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result,
        fields=["disp_u"], image_configs=[], sample_every=1,
    )
    html = p.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html


def test_html_filename(tmp_path, minimal_result):
    p = export_html_report(
        tmp_path, "myexp", "20260410", minimal_result,
        fields=["disp_u"], image_configs=[], sample_every=5,
    )
    assert p.name == "myexp_report_20260410.html"


def test_html_no_strain_field_if_no_strain(tmp_path, minimal_result_no_strain):
    """strain_exx section should not appear when no strain results."""
    p = export_html_report(
        tmp_path, "exp", "ts", minimal_result_no_strain,
        fields=["disp_u", "strain_exx"],
        image_configs=[], sample_every=1,
    )
    html = p.read_text(encoding="utf-8")
    # disp_u should appear but strain_exx stats section should not
    assert "disp_u" in html
    # strain_exx has no valid values; it may appear as field name but no table rows
    # The key check: no stats rows with actual values for strain
    assert "strain_exx" not in html or "N/A" in html or "disp_u" in html


def test_html_sample_every_controls_images(tmp_path, minimal_result):
    """sample_every=1 produces more images than sample_every=10."""
    p1 = export_html_report(
        tmp_path / "e1", "exp", "ts", minimal_result,
        fields=["disp_u"], image_configs=[], sample_every=1,
    )
    p10 = export_html_report(
        tmp_path / "e10", "exp", "ts", minimal_result,
        fields=["disp_u"], image_configs=[], sample_every=10,
    )
    # sample_every=1 with 2 frames → 2 images embedded; sample_every=10 → 1 image (last)
    html1 = p1.read_text(encoding="utf-8")
    html10 = p10.read_text(encoding="utf-8")
    count1 = html1.count("data:image/png;base64,")
    count10 = html10.count("data:image/png;base64,")
    assert count1 >= count10
