"""Batch-process several DIC samples from one config file, without the GUI.

Point it at a JSON or YAML config listing your samples and run:

    python examples/scripting/batch_process.py my_batch.json
    python examples/scripting/batch_process.py my_batch.yaml   # needs pyyaml

Paths inside the config are resolved against your working directory, not
against the config file, so run the command from wherever those paths make
sense. The bundled example expects the repository root.

Each sample is loaded, correlated, strain-computed and exported on its own; a
sample that fails is logged and the batch moves on to the next one. See
``batch_config.example.yaml`` next to this file for every supported key.

Nothing here is private API: it is the same ``run_aldic`` entry point the GUI
calls, so results are identical to clicking Run in the interface.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange, PipelineResult
from al_dic.core.pipeline import run_aldic
from al_dic.export.export_csv import export_csv
from al_dic.export.export_mat import export_mat
from al_dic.export.export_npz import export_npz
from al_dic.export.export_params import export_params
from al_dic.export.export_utils import make_timestamp
from al_dic.io.io_utils import load_images, load_masks, read_mask_as_bool

# Canonical field names accepted by the exporters.
DISP_FIELDS = {"disp_u", "disp_v", "disp_magnitude"}
STRAIN_FIELDS = {
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
}
ALL_FIELDS = DISP_FIELDS | STRAIN_FIELDS
DEFAULT_FIELDS = ["disp_u", "disp_v", "strain_exx", "strain_eyy", "strain_exy"]
KNOWN_EXPORTS = {"npz", "mat", "csv", "params", "images"}


# ---------------------------------------------------------------- config


def load_config(path: Path) -> dict[str, Any]:
    """Read a batch config from .json or .yaml/.yml."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ModuleNotFoundError:
            raise SystemExit(
                f"{path.name} is YAML but PyYAML is not installed.\n"
                f"Either run:  pip install pyyaml\n"
                f"or convert the config to JSON (same keys)."
            ) from None
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)

    if not isinstance(cfg, dict):
        raise ValueError(f"{path.name}: top level must be a mapping")
    if not cfg.get("samples"):
        raise ValueError(f"{path.name}: no 'samples' listed")
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail fast on typos, before spending hours correlating."""
    bad = set(cfg.get("export", [])) - KNOWN_EXPORTS
    if bad:
        raise ValueError(
            f"Unknown export type(s): {sorted(bad)}. "
            f"Valid: {sorted(KNOWN_EXPORTS)}"
        )
    for key in ("fields", "image_fields"):
        bad = set(cfg.get(key, [])) - ALL_FIELDS
        if bad:
            raise ValueError(
                f"Unknown field(s) in '{key}': {sorted(bad)}. "
                f"Valid: {sorted(ALL_FIELDS)}"
            )
    for i, s in enumerate(cfg["samples"]):
        if not s.get("images"):
            raise ValueError(f"samples[{i}]: 'images' folder is required")


# ------------------------------------------------------------------ ROI


def resolve_masks(
    sample: dict[str, Any],
    images: list[np.ndarray],
) -> list[np.ndarray]:
    """Build the per-frame mask list for one sample.

    ``roi``      -- one mask image, applied to every frame.
    ``roi_dir``  -- a folder of masks, one per frame (growing cracks etc.).
    neither      -- the whole image is the region of interest.
    """
    shape = images[0].shape
    roi_dir = sample.get("roi_dir")
    roi_file = sample.get("roi")

    if roi_dir:
        masks = load_masks(roi_dir, pattern=sample.get("roi_pattern", "*.tif"))
        if len(masks) != len(images):
            raise ValueError(
                f"roi_dir has {len(masks)} masks but there are {len(images)} "
                f"images -- per-frame ROIs must match the image count"
            )
        return [m.astype(np.float64) for m in masks]

    if roi_file:
        roi = read_mask_as_bool(roi_file, target_shape=shape)
        return [roi.astype(np.float64)] * len(images)

    return [np.ones(shape, dtype=np.float64)] * len(images)


def roi_range_from(mask: np.ndarray) -> GridxyROIRange:
    """Bounding box of a mask, as the pixel box to correlate.

    ``gridxy_roi_range`` has no usable default -- in the GUI it comes from the
    ROI you draw, so a script must set it explicitly.
    """
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        raise ValueError("ROI mask is empty -- nothing to correlate")
    return GridxyROIRange(
        gridx=(int(xs.min()), int(xs.max())),
        gridy=(int(ys.min()), int(ys.max())),
    )


# -------------------------------------------------------------- exporting


def export_results(
    result: PipelineResult,
    dest: Path,
    name: str,
    cfg: dict[str, Any],
    image_files: list[str],
    roi_mask: np.ndarray,
) -> dict[str, int]:
    """Write the requested outputs; returns {kind: file count}."""
    wanted = cfg.get("export", ["npz", "params"])
    fields = cfg.get("fields", DEFAULT_FIELDS)
    ts = make_timestamp()
    written: dict[str, int] = {}

    if "npz" in wanted:
        out = export_npz(dest, name, ts, result, fields,
                         per_frame=cfg.get("npz_per_frame", False))
        written["npz"] = len(out) if isinstance(out, list) else 1
    if "mat" in wanted:
        export_mat(dest, name, ts, result, fields)
        written["mat"] = 1
    if "csv" in wanted:
        written["csv"] = len(export_csv(dest, name, ts, result, fields))
    if "params" in wanted:
        export_params(dest, name, ts, result)
        written["params"] = 1

    if "images" in wanted:
        # Imported lazily: this is the one exporter whose config object lives
        # in the GUI package, so batches that export data only never touch Qt.
        from al_dic.export.export_png import export_png
        from al_dic.gui.dialogs.export_dialog import FieldImageConfig

        configs = [
            FieldImageConfig(f, True, cfg.get("colormap", "jet"), True, 0.0, 1.0,
                             cfg.get("bg_alpha", 0.7))
            for f in cfg.get("image_fields", ["disp_u", "disp_v"])
        ]
        paths = export_png(
            dest_dir=dest, prefix=name, timestamp=ts, results=result,
            configs=configs, image_files=image_files,
            bg_mode=cfg.get("bg_mode", "ref_frame"), roi_mask=roi_mask,
            dpi=cfg.get("dpi", 150), show_deformed=cfg.get("show_deformed", False),
            hidden_bg_color=cfg.get("hidden_bg_color", "white"),
            frame_start=0, frame_end=len(result.result_disp) - 1,
            include_colorbar=cfg.get("colorbar", True),
            image_format=cfg.get("image_format", "png"),
        )
        written["images"] = len(paths)

    return written


# ------------------------------------------------------------ one sample


def process_sample(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the whole pipeline for one sample and export it."""
    name = sample.get("name") or Path(sample["images"]).name
    dest = Path(cfg.get("output_dir", "results")) / name
    t0 = time.perf_counter()

    images = load_images(sample["images"], pattern=sample.get("pattern", "*.tif"))
    if len(images) < 2:
        raise ValueError(f"need >= 2 images, found {len(images)}")
    masks = resolve_masks(sample, images)

    # Global params, then per-sample overrides on top.
    params = {**cfg.get("params", {}), **sample.get("params", {})}
    para = dicpara_default(
        img_size=images[0].shape,
        use_masks=True,
        img_ref_mask=masks[0],
        gridxy_roi_range=roi_range_from(masks[0]),
        **params,
    )

    last_pct = -10
    def progress(frac: float, msg: str) -> None:
        nonlocal last_pct
        pct = int(frac * 100)
        if pct >= last_pct + 20:
            last_pct = pct
            print(f"      {pct:3d}%  {msg[:56]}", flush=True)

    result = run_aldic(
        para, images, masks,
        progress_fn=progress,
        compute_strain=cfg.get("compute_strain", True),
    )

    image_files = sorted(
        str(p) for p in Path(sample["images"]).glob(sample.get("pattern", "*.tif"))
    )
    written = export_results(
        result, dest, name, cfg, image_files, masks[0] > 0.5
    )

    return {
        "name": name,
        "frames": sum(r is not None for r in result.result_disp),
        "strain_frames": sum(r is not None for r in result.result_strain),
        "nodes": len(result.result_disp[-1].U) // 2 if result.result_disp[-1] else 0,
        "written": written,
        "seconds": time.perf_counter() - t0,
        "output": str(dest),
    }


def already_done(sample: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """True if this sample already has exported results (for resume)."""
    name = sample.get("name") or Path(sample["images"]).name
    dest = Path(cfg.get("output_dir", "results")) / name
    return dest.is_dir() and any(dest.glob("*_results_*"))


# ------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        print("usage: python batch_process.py <config.json|config.yaml>")
        return 2

    cfg = load_config(Path(argv[1]))
    validate_config(cfg)

    samples = cfg["samples"]
    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "batch_log.txt"
    log_lines: list[str] = []
    ok = skipped = 0

    print(f"pyALDIC batch — {len(samples)} sample(s) -> {out_root}")

    for i, sample in enumerate(samples, 1):
        name = sample.get("name") or Path(sample["images"]).name
        head = f"[{i}/{len(samples)}] {name}"

        if cfg.get("resume", True) and already_done(sample, cfg):
            print(f"{head}  (results exist, skipped)")
            log_lines.append(f"{name}: SKIPPED (already done)")
            skipped += 1
            continue

        print(head)
        try:
            r = process_sample(sample, cfg)
            ok += 1
            summary = ", ".join(f"{k}={v}" for k, v in r["written"].items())
            print(f"      OK  {r['frames']} frame(s), {r['nodes']} nodes, "
                  f"{r['seconds']:.1f}s  [{summary}]")
            log_lines.append(
                f"{name}: OK frames={r['frames']} strain={r['strain_frames']} "
                f"nodes={r['nodes']} {r['seconds']:.1f}s {summary}"
            )
        except Exception as exc:                    # keep the batch running
            print(f"      FAILED: {type(exc).__name__}: {exc}")
            log_lines.append(f"{name}: FAILED {type(exc).__name__}: {exc}")
            log_lines.append(traceback.format_exc())

    failed = len(samples) - ok - skipped
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nDone — {ok} ok, {skipped} skipped, {failed} failed")
    print(f"Log: {log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
