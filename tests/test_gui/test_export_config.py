"""Tests for export dialog pure-data configuration types."""

from pathlib import Path

from al_dic.gui.dialogs.export_dialog import (
    VizExportHint,
    FieldImageConfig,
    ExportConfig,
)


class TestVizExportHint:
    def test_defaults(self):
        h = VizExportHint()
        assert h.colormap == "jet"
        assert h.auto_range is True
        assert h.vmin == 0.0
        assert h.vmax == 1.0
        assert h.show_deformed is False
        assert h.overlay_alpha == 0.7


class TestFieldImageConfig:
    def test_creation(self):
        cfg = FieldImageConfig(
            field_name="disp_u",
            enabled=True,
            colormap="RdBu_r",
            auto_range=False,
            vmin=-0.1,
            vmax=0.1,
        )
        assert cfg.field_name == "disp_u"
        assert cfg.bg_alpha == 0.7  # default


class TestExportConfig:
    def test_defaults(self):
        cfg = ExportConfig(
            dest_dir=Path("/tmp/export"),
            prefix="test",
            timestamp="20260411_120000",
        )
        assert cfg.export_npz is True
        assert cfg.export_mat is True
        assert cfg.export_csv is True
        assert cfg.data_fields == []
        assert cfg.export_images is False
        assert cfg.image_format == "png"
        assert cfg.anim_format == "mp4"
        assert cfg.export_report is False

    def test_custom_values(self):
        cfg = ExportConfig(
            dest_dir=Path("/tmp/custom"),
            prefix="exp",
            timestamp="20260411",
            export_npz=False,
            image_dpi=300,
            anim_fps=30,
            bg_mode="current_frame",
        )
        assert cfg.export_npz is False
        assert cfg.image_dpi == 300
        assert cfg.anim_fps == 30
        assert cfg.bg_mode == "current_frame"
