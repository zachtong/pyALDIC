"""Export pipeline results as animated GIF or MP4.

Rendering pipeline:
    1. For each frame, call render_field_frame() (reused from export_png).
    2. Collect rendered BGR uint8 images.
    3. GIF:  convert BGR→RGB, pass frames to imageio.mimwrite.
    4. MP4:  write frames via cv2.VideoWriter (fourcc mp4v; falls back to avi).

Directory structure:
    dest_dir/
      {prefix}_animation_{timestamp}/
        {field_name}.mp4   (or .gif)
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir, frame_tag


def export_animation(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    configs: list,
    image_files: list[str],
    bg_mode: str,
    roi_mask: NDArray | None,
    fmt: str,
    fps: int,
    show_deformed: bool,
    frame_start: int,
    frame_end: int,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    per_frame_rois: dict[int, NDArray] | None = None,
    include_colorbar: bool = False,
    use_physical_units: bool = False,
    pixel_size: float = 1.0,
    pixel_unit: str = "mm",
    render_max_dim: int = 1536,
    frame_step: int = 1,
    output_max_dim: int = 0,
    colorbar_style: ColorbarStyle | None = None,
    margin_ratio: float = 0.0,
    margin_color: str = "white",
    fill_trimmed_edges: bool = False,
    hidden_bg_color: str = "black",
) -> list[Path]:
    """Export one animation file per enabled field.

    Args:
        dest_dir: Parent output directory.
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        configs: List of FieldImageConfig from export_dialog, one per field.
        image_files: Ordered list of source image file paths.
        bg_mode: "ref_frame" or "current_frame".
        roi_mask: Optional (H, W) bool mask for field trimming
                  (reference coords).
        fmt: Animation format: "mp4" or "gif".
        fps: Frames per second.
        show_deformed: Shift node positions by accumulated displacement.
        frame_start: First frame index (inclusive).
        frame_end: Last frame index (inclusive; -1 = last frame).
        stop_event: If set, stop after the current frame.
        progress_callback: Called with (frames_done, total_frames, field_name).
        per_frame_rois: Optional mapping of image-file index → (H, W) bool
                     mask.  Overrides inverse-warped mask in deformed mode.
                     Keys: 0 = reference, 1..N = deformed frames.
        include_colorbar: Append a matplotlib-rendered colorbar strip to the
                     right of each frame.
        use_physical_units: Scale displacement values by *pixel_size*.
        pixel_size:  Physical size of one pixel (e.g. mm/px).
        pixel_unit:  Unit string shown on colorbar labels.
        fill_trimmed_edges: When True, re-interpolate the edge-trimmed strain
                     band from reliable interior nodes instead of blanking it
                     (matches the strain window's toggle).  Default False.

    Returns:
        List of Paths to written animation files (one per enabled field).
    """
    from al_dic.export.export_png import (
        render_field_frame, _extract_field_values, _load_frame_image,
        _compute_warped_mask, scale_field_values, colorbar_label,
        attach_colorbar, ColorbarStyle, add_margin, _DISPLACEMENT_FIELDS,
        output_shape_for, hidden_bg_fill,
    )

    cb_style = colorbar_style if colorbar_style is not None else ColorbarStyle()

    anim_dir = dest_dir / f"{prefix}_animation_{timestamp}"
    ensure_dir(anim_dir)

    n_frames = len(results.result_disp)
    if frame_end < 0 or frame_end >= n_frames:
        frame_end = n_frames - 1

    enabled_configs = [c for c in configs if c.enabled]
    if not enabled_configs:
        return []

    coords = results.dic_mesh.coordinates_fem
    img_shape = results.dic_para.img_size  # (H, W)
    if img_shape == (0, 0):
        img_shape = (256, 256)

    # Output resolution cap (file size + encode time).
    out_shape = output_shape_for(img_shape, output_max_dim)

    # Frame decimation: keep every Nth frame (faster + smaller, choppier). The
    # given fps is the pre-decimation timeline, so scale the playback fps down
    # by the same factor to keep the real-time duration unchanged.
    frame_step = max(1, int(frame_step))
    out_fps = max(1, round(fps / frame_step))

    frame_indices = list(range(frame_start, frame_end + 1, frame_step))
    total_frames = len(frame_indices)
    fmt = fmt.lower()
    paths: list[Path] = []

    # Reuse the field triangulation across fields/frames. Non-deformed:
    # render_coords == coords is constant, so one Delaunay per valid mask
    # serves the whole animation; deformed: a fresh cache per frame.
    shared_tri_cache: dict = {}

    # "ref_frame" mode uses image 0 as background for every frame and field;
    # decode it once instead of re-reading the file inside both loops.
    ref_bg = (
        _load_frame_image(image_files, 0, "ref_frame")
        if bg_mode == "ref_frame" else None
    )

    # Neither MP4 nor GIF carries a usable alpha channel here, so a request
    # for transparency becomes the nearest thing that does render.
    if hidden_bg_color == "transparent":
        hidden_bg_color = "white"
    blank_level = hidden_bg_fill(hidden_bg_color) if bg_mode == "none" else 0

    for cfg in enabled_configs:
        frames_done = 0
        # Stream frames straight to the encoder instead of collecting them:
        # hundreds of 4K frames would otherwise pin tens of GB of RAM.
        writer: _StreamingAnimWriter | None = None

        for t in frame_indices:
            if stop_event is not None and stop_event.is_set():
                break

            fr = results.result_disp[t]

            # Deformed node positions
            if show_deformed and fr.U_accum is not None:
                u, v = split_uv(fr.U_accum)
                deformed_coords: NDArray | None = coords + np.column_stack([u, v])
            elif show_deformed and fr.U is not None:
                u, v = split_uv(fr.U)
                deformed_coords = coords + np.column_stack([u, v])
            else:
                deformed_coords = None

            frame_tri_cache = (
                {} if deformed_coords is not None else shared_tri_cache
            )

            # result_disp[t] corresponds to image_files[t + 1] (deformed frame);
            # reuse the pre-decoded reference image in "ref_frame" mode.
            bg_image = ref_bg if ref_bg is not None else _load_frame_image(
                image_files, t + 1, bg_mode)

            # Resolve deformed mask (same logic as export_png)
            deformed_mask: NDArray | None = None
            if show_deformed and deformed_coords is not None:
                img_idx = t + 1
                if per_frame_rois is not None:
                    pfr = per_frame_rois.get(img_idx)
                    if pfr is not None:
                        deformed_mask = pfr
                if deformed_mask is None and roi_mask is not None:
                    deformed_mask = _compute_warped_mask(
                        coords, deformed_coords, roi_mask, img_shape
                    )

            raw_values = _extract_field_values(cfg.field_name, t, results, fr)
            if raw_values is None:
                # Blank frame (at output size) keeps timing consistent.
                # It has to match the chosen fill, or a field that drops out
                # for one frame flashes black in the middle of a white figure.
                img = np.full((*out_shape, 3), blank_level, dtype=np.uint8)
            else:
                # Physical-unit scaling
                values = (scale_field_values(raw_values, cfg.field_name, pixel_size)
                          if use_physical_units else raw_values)

                # Pre-compute range when colorbar is requested, or
                # when physical-unit scaling changes the value range.
                is_scaled = use_physical_units and cfg.field_name in _DISPLACEMENT_FIELDS
                need_precompute = include_colorbar or is_scaled
                if need_precompute:
                    finite = values[np.isfinite(values)]
                    if cfg.auto_range:
                        actual_vmin = float(finite.min()) if len(finite) > 0 else 0.0
                        actual_vmax = float(finite.max()) if len(finite) > 0 else 1.0
                    else:
                        actual_vmin, actual_vmax = cfg.vmin, cfg.vmax
                        if use_physical_units and cfg.field_name in _DISPLACEMENT_FIELDS:
                            actual_vmin *= pixel_size
                            actual_vmax *= pixel_size
                    render_cfg = replace(cfg, auto_range=False,
                                         vmin=actual_vmin, vmax=actual_vmax)
                else:
                    render_cfg = cfg

                img = render_field_frame(
                    coords=coords,
                    values=values,
                    image_shape=img_shape,
                    bg_image=bg_image,
                    field_cfg=render_cfg,
                    roi_mask=roi_mask,
                    deformed_coords=deformed_coords,
                    deformed_mask=deformed_mask,
                    tri_cache=frame_tri_cache,
                    render_max_dim=render_max_dim,
                    output_shape=out_shape,
                    blank_invalid_nodes=not fill_trimmed_edges,
                    hidden_bg_color=hidden_bg_color,
                )

                # Append the styled colorbar
                if include_colorbar:
                    cb_lbl = colorbar_label(
                        cfg.field_name, use_physical_units, pixel_unit)
                    img = attach_colorbar(
                        img, cb_style, cfg.colormap,
                        actual_vmin, actual_vmax, cb_lbl)
                img = add_margin(img, margin_ratio, margin_color)

            # Lazily open the encoder once the first frame's size is known
            # (that size includes the colorbar strip when present).
            if writer is None:
                w = _StreamingAnimWriter(
                    fmt, anim_dir, cfg.field_name, out_fps, img.shape[:2])
                if not w.ok:
                    w.close()
                    # Breaking left `paths` empty and raised nothing, which the
                    # dialog rendered as a green "Exported 0 animation(s)".
                    raise IOError(
                        f"Could not open a video encoder for {anim_dir}: the "
                        "FFmpeg backend may be missing, or the destination is "
                        "not writable"
                    )
                writer = w
            writer.append(img)

            frames_done += 1
            if progress_callback is not None:
                progress_callback(frames_done, total_frames, cfg.field_name)

        if writer is not None:
            writer.close()
            paths.append(writer.out)

    return paths


class _StreamingAnimWriter:
    """Append BGR frames to a GIF/MP4/AVI encoder one at a time.

    Streaming avoids holding every rendered frame in RAM (hundreds of 4K
    frames = tens of GB).  The output frame size is fixed by the first frame;
    later frames are resized to match (as the old batch writers did).  For
    MP4 the mp4v codec is tried first, falling back to XVID/.avi.
    """

    def __init__(self, fmt: str, anim_dir: Path, field_name: str,
                 fps: int, frame_hw: tuple[int, int]) -> None:
        self.fmt = fmt
        self.h, self.w = frame_hw
        self.fps = fps
        self.ok = True
        if fmt == "gif":
            import imageio
            self.out = anim_dir / f"{field_name}.gif"
            self._w = imageio.get_writer(
                str(self.out), format="GIF", mode="I",
                duration=1.0 / max(fps, 1), loop=0)
            return
        # MP4 (mp4v) with AVI (XVID) fallback
        self.out = anim_dir / f"{field_name}.mp4"
        self._w = cv2.VideoWriter(
            str(self.out), cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps), (self.w, self.h))
        if not self._w.isOpened():
            self.out = anim_dir / f"{field_name}.avi"
            self._w = cv2.VideoWriter(
                str(self.out), cv2.VideoWriter_fourcc(*"XVID"),
                float(fps), (self.w, self.h))
            self.ok = self._w.isOpened()

    def append(self, frame: NDArray) -> None:
        if frame.shape[:2] != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h))
        if self.fmt == "gif":
            self._w.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            self._w.write(frame)

    def close(self) -> None:
        if self.fmt == "gif":
            self._w.close()
        else:
            self._w.release()
