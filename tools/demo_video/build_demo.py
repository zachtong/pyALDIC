"""Assemble the pyALDIC demo video (~80s, 1920x1080 @ 30fps).

v3 revisions:
  - Longer pacing (<=90s total, room to read each step).
  - End-of-step dwell so the last frame lingers before fade-out.
  - Step 2: bullet annotations on the right side of the zoom panel
    (tracking mode / solver / initialization / subset size-step).
  - Step 3 branches drop the "(A)/(B)" suffix.
  - Step 3 branch B retitled "Batch-import ROIs" and slowed down.
  - Step 4 retitled "Refine where it matters" with a center zoom-in
    animation at the very end so the refined mesh is clearly visible.
  - Step 5 speed bumped to 13x (was 10x); no "(10x playback)" label.
  - Step 6 card retitled "Displacement field".
  - Step 7 card retitled "Strain field".
  - Outro drops "pip install al-dic" and adds the app icon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIF_DIR = PROJECT_ROOT / "assets" / "videos"
OUT_MP4 = PROJECT_ROOT / "assets" / "videos" / "pyALDIC_demo.mp4"

# ---------- Canvas + palette -----------------------------------------------
W, H = 1920, 1080
FPS = 30

BG_RGB = (10, 10, 15)
TEXT_RGB = (240, 240, 245)
MUTED_RGB = (140, 150, 170)
ACCENT_RGB = (149, 108, 255)

# ---------- Timing ----------------------------------------------------------
TITLE_S = 3.5
CARD_S = 1.8
OUTRO_S = 4.0
FADE_S = 0.4
DWELL_S_DEFAULT = 1.2           # hold on last frame before fade-out
TOP_BAR_H = 56
TOP_BAR_ACCENT_H = 2

# ---------- Zoom config ----------------------------------------------------
FROST_BLUR_RADIUS = 8
FROST_DIM_ALPHA = 40
ZOOM_SCALE = 2.0
STEP2_CROP = (0, 600, 500, 1200)
STEP5_CROP = (1700, 80, 2150, 520)
RED_RGB = (255, 64, 64)

# Step 4 center zoom-in animation
S4_ZOOM_ANIM_S = 1.5
S4_ZOOM_FINAL = 2.0

# ---------- Fonts ----------------------------------------------------------
_FONTS_REG = ["C:/Windows/Fonts/segoeui.ttf",
              "C:/Windows/Fonts/arial.ttf", "arial.ttf"]
_FONTS_BOLD = ["C:/Windows/Fonts/segoeuib.ttf",
               "C:/Windows/Fonts/arialbd.ttf", "arialbd.ttf"]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for fp in (_FONTS_BOLD if bold else _FONTS_REG):
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------- Step specs -----------------------------------------------------
@dataclass(frozen=True)
class StepSpec:
    number: int
    number_label: str
    title: str
    clip_stem: str
    target_clip_s: float
    zoom_crop: tuple[int, int, int, int] | None = None
    speedup: float = 1.0
    dwell_s: float = DWELL_S_DEFAULT
    annotate: str | None = None           # "step2" | "step4_center_zoom"


STEPS: tuple[StepSpec, ...] = (
    StepSpec(1, "Step 1", "Import your image sequence",
             "step1_import_images",
             target_clip_s=6.0),
    StepSpec(2, "Step 2", "Pick your workflow",
             "step2_select_your_workflow",
             target_clip_s=7.0,
             zoom_crop=STEP2_CROP,
             annotate="step2"),
    StepSpec(3, "Step 3", "Draw the ROI manually",
             "step3.1_manually_draw_ROI",
             target_clip_s=6.0),
    StepSpec(3, "Step 3", "Batch-import ROIs",
             "step3.2_batch_import_ROIs_for_increnmental_tracking",
             target_clip_s=9.0),
    StepSpec(4, "Step 4", "Refine where it matters",
             "step4_mesh_refinement",
             target_clip_s=6.0,
             annotate="step4_center_zoom"),
    StepSpec(5, "Step 5", "Run DIC analysis",
             "step5_computing",
             target_clip_s=3.0,
             zoom_crop=STEP5_CROP,
             speedup=13.0),
    StepSpec(6, "Step 6", "Displacement field",
             "step6_displacement_display",
             target_clip_s=4.0),
    StepSpec(7, "Step 7", "Strain field",
             "step7_strain_von_Mises_strain",
             target_clip_s=4.0),
)


# ---------- Primitives -----------------------------------------------------

def _bg() -> Image.Image:
    return Image.new("RGB", (W, H), BG_RGB)


def _draw_center(draw: ImageDraw.ImageDraw,
                 cy: int, text: str,
                 font: ImageFont.FreeTypeFont,
                 fill: tuple[int, int, int]) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((W - tw) // 2 - bbox[0], cy - th // 2 - bbox[1]),
              text, font=font, fill=fill)


def _img_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _load_app_icon(px: int) -> Image.Image | None:
    asset_dir = PROJECT_ROOT / "src" / "al_dic" / "gui" / "assets" / "icon"
    for name in ("pyALDIC-256.png", "pyALDIC.png"):
        path = asset_dir / name
        if path.is_file():
            im = Image.open(path).convert("RGBA")
            im.thumbnail((px, px), Image.LANCZOS)
            return im
    return None


# ---------- Title card -----------------------------------------------------

def _title_frame() -> Image.Image:
    img = _bg()
    draw = ImageDraw.Draw(img)

    icon = _load_app_icon(260)
    title_font = _font(150, bold=True)
    sub_font = _font(48, bold=False)

    block_h = (260 if icon else 0) + 40 + 160 + 20 + 60
    top = (H - block_h) // 2

    if icon is not None:
        img.paste(icon, ((W - icon.width) // 2, top), icon)
        top += icon.height + 40

    _draw_center(draw, top + 80, "pyALDIC", title_font, ACCENT_RGB)
    top += 180

    _draw_center(draw, top + 30,
                 "Augmented Lagrangian Digital Image Correlation in Python",
                 sub_font, TEXT_RGB)
    return img


# ---------- Step intro card ------------------------------------------------

def _step_card(label: str, title: str) -> Image.Image:
    img = _bg()
    draw = ImageDraw.Draw(img)
    step_font = _font(120, bold=True)
    title_font = _font(60, bold=False)
    cy = H // 2
    _draw_center(draw, cy - 40, label, step_font, ACCENT_RGB)
    _draw_center(draw, cy + 80, title, title_font, TEXT_RGB)
    return img


# ---------- Outro ----------------------------------------------------------

def _outro_frame() -> Image.Image:
    img = _bg()
    draw = ImageDraw.Draw(img)

    icon = _load_app_icon(180)
    brand_font = _font(120, bold=True)
    url_font = _font(52, bold=False)
    name_font = _font(40, bold=False)
    small_font = _font(32, bold=False)

    # Layout block: icon + "pyALDIC" grouped top; URL middle; info below.
    top = 170
    if icon is not None:
        img.paste(icon, ((W - icon.width) // 2, top), icon)
        top += icon.height + 30

    _draw_center(draw, top + 60, "pyALDIC", brand_font, ACCENT_RGB)
    top += 160
    _draw_center(draw, top, "github.com/zachtong/pyALDIC",
                 url_font, TEXT_RGB)

    top += 140
    _draw_center(draw, top, "Zixiang Tong", name_font, TEXT_RGB)
    top += 50
    _draw_center(draw, top, "zachtong@utexas.edu", small_font, MUTED_RGB)
    top += 80
    _draw_center(draw, top, "Dr. Jin Yang Group   -   UT Austin",
                 small_font, MUTED_RGB)
    return img


# ---------- Top bar overlay ------------------------------------------------

def _apply_top_bar(img: Image.Image, label_text: str) -> Image.Image:
    bar = Image.new("RGBA", (W, TOP_BAR_H + TOP_BAR_ACCENT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(bar)
    d.rectangle((0, 0, W, TOP_BAR_H), fill=(0, 0, 0, int(0.55 * 255)))
    d.rectangle((0, TOP_BAR_H, W, TOP_BAR_H + TOP_BAR_ACCENT_H),
                fill=(*ACCENT_RGB, int(0.85 * 255)))

    font = _font(26, bold=True)
    bbox = d.textbbox((0, 0), label_text, font=font)
    th = bbox[3] - bbox[1]
    d.text((24, (TOP_BAR_H - th) // 2 - bbox[1]),
           label_text, font=font, fill=(255, 255, 255, 240))

    out = img.convert("RGBA").copy()
    out.alpha_composite(bar, (0, 0))
    return out.convert("RGB")


# ---------- Zoom + frost (source res) --------------------------------------

def _clamp(crop: tuple[int, int, int, int],
           w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _apply_zoom_frost(frame_rgb: np.ndarray,
                      crop: tuple[int, int, int, int]) -> np.ndarray:
    original = Image.fromarray(frame_rgb, mode="RGB")
    w, h = original.size
    x0, y0, x1, y1 = _clamp(crop, w, h)

    frosted = original.filter(
        ImageFilter.GaussianBlur(FROST_BLUR_RADIUS)).convert("RGBA")
    frosted.alpha_composite(
        Image.new("RGBA", (w, h), (0, 0, 0, FROST_DIM_ALPHA)))
    frosted = frosted.convert("RGB")
    frosted.paste(original.crop((x0, y0, x1, y1)), (x0, y0))

    draw = ImageDraw.Draw(frosted, "RGBA")
    draw.rectangle((x0, y0, x1, y1),
                   outline=(*RED_RGB, 255), width=6)

    region = original.crop((x0, y0, x1, y1))
    zw = int(round((x1 - x0) * ZOOM_SCALE))
    zh = int(round((y1 - y0) * ZOOM_SCALE))
    zoomed = region.resize((zw, zh), Image.LANCZOS)
    cx = (w - zw) // 2
    cy = (h - zh) // 2
    frosted.paste(zoomed, (cx, cy))
    draw = ImageDraw.Draw(frosted, "RGBA")
    draw.rectangle((cx, cy, cx + zw, cy + zh),
                   outline=(*RED_RGB, 255), width=10)
    return np.asarray(frosted)


# ---------- Step 2 annotations (bullets + labels on right of zoom) ---------

# (y_on_source, text)  — y is the vertical center *inside the zoom box*
# on the 2124x1482 source frame. Zoom box is at y=[141, 1341].
_STEP2_LABELS: tuple[tuple[int, str], ...] = (
    (310, "Tracking mode"),
    (405, "Solver"),
    (485, "Initialization"),
    (1200, "Subset size  /  step"),
)


def _apply_step2_annotations(frame_rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame_rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(44, bold=True)

    # Zoom box right edge is at x = (2124 - 1000) // 2 + 1000 = 1562.
    zoom_right = 1562
    dot_x = zoom_right + 28
    text_x = zoom_right + 80

    for y, text in _STEP2_LABELS:
        # Small accent dot + short connector line.
        r = 10
        draw.ellipse((dot_x - r, y - r, dot_x + r, y + r),
                     fill=(*ACCENT_RGB, 255))
        draw.line(((dot_x + r + 4, y), (text_x - 12, y)),
                  fill=(*ACCENT_RGB, 210), width=3)
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
        draw.text((text_x, y - th // 2 - bbox[1]),
                  text, font=font, fill=(240, 240, 245, 255))
    return np.asarray(img.convert("RGB"))


# ---------- Step 4 center zoom-in animation --------------------------------

def _center_zoom_frame(last_rgb: np.ndarray, t: float) -> np.ndarray:
    """Progressive zoom towards image center. t in [0, 1]."""
    img = Image.fromarray(last_rgb, mode="RGB")
    sw, sh = img.size
    factor = 1.0 + (S4_ZOOM_FINAL - 1.0) * t
    crop_w = int(sw / factor)
    crop_h = int(sh / factor)
    x0 = (sw - crop_w) // 2
    y0 = (sh - crop_h) // 2
    zoomed = img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
        (sw, sh), Image.LANCZOS)
    return np.asarray(zoomed)


# ---------- Fit + read + resample ------------------------------------------

def _fit_to_canvas(frame_rgb: np.ndarray) -> np.ndarray:
    sh, sw = frame_rgb.shape[:2]
    scale = min(W / sw, H / sh)
    nw = int(round(sw * scale))
    nh = int(round(sh * scale))
    pil = Image.fromarray(frame_rgb, mode="RGB").resize(
        (nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), BG_RGB)
    canvas.paste(pil, ((W - nw) // 2, (H - nh) // 2))
    return np.asarray(canvas)


def _read_clip_rgb(path: Path) -> list[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    sw, sh = meta["size"]
    frames: list[np.ndarray] = []
    for raw in reader:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(sh, sw, 3)
        frames.append(arr.copy())
    return frames


def _resample_indices(n_src: int, n_out: int) -> list[int]:
    if n_src == 0:
        return [0] * n_out
    return np.linspace(0, n_src - 1, num=n_out).round().astype(int).tolist()


def _clip_frames(spec: StepSpec) -> list[np.ndarray]:
    src = _read_clip_rgb(GIF_DIR / f"{spec.clip_stem}.mp4")
    n_src = len(src)
    src_duration = n_src / 15.0
    eff_target = spec.target_clip_s
    if spec.speedup > 1.0:
        eff_target = min(spec.target_clip_s, src_duration / spec.speedup)
    n_out = max(1, int(round(eff_target * FPS)))

    idx = _resample_indices(n_src, n_out)
    label_text = f"{spec.number_label}    -    {spec.title}"

    out: list[np.ndarray] = []
    last_source = None
    for i in idx:
        f = src[i]
        if spec.zoom_crop is not None:
            f = _apply_zoom_frost(f, spec.zoom_crop)
        if spec.annotate == "step2":
            f = _apply_step2_annotations(f)
        last_source = f
        fitted = Image.fromarray(_fit_to_canvas(f), mode="RGB")
        fitted = _apply_top_bar(fitted, label_text)
        out.append(np.asarray(fitted))

    # Step 4 center zoom-in on the LAST source frame.
    if spec.annotate == "step4_center_zoom" and last_source is not None:
        n_anim = int(round(S4_ZOOM_ANIM_S * FPS))
        for k in range(1, n_anim + 1):
            t = k / n_anim
            z = _center_zoom_frame(last_source, t)
            fitted = Image.fromarray(_fit_to_canvas(z), mode="RGB")
            fitted = _apply_top_bar(fitted, label_text)
            out.append(np.asarray(fitted))

    return out


def _append_dwell(frames: list[np.ndarray], dwell_s: float) -> list[np.ndarray]:
    if not frames or dwell_s <= 0:
        return frames
    n = int(round(dwell_s * FPS))
    return frames + [frames[-1]] * n


# ---------- Static frames + fades ------------------------------------------

def _still_frames(img: Image.Image, seconds: float) -> list[np.ndarray]:
    arr = _img_np(img)
    return [arr] * int(round(seconds * FPS))


def _apply_fades(frames: list[np.ndarray],
                 fade_in_s: float = FADE_S,
                 fade_out_s: float = FADE_S) -> list[np.ndarray]:
    n = len(frames)
    fi = min(int(round(fade_in_s * FPS)), n // 2)
    fo = min(int(round(fade_out_s * FPS)), n - fi)
    out: list[np.ndarray] = []
    for idx, f in enumerate(frames):
        if idx < fi:
            alpha = (idx + 1) / max(fi, 1)
        elif idx >= n - fo:
            alpha = (n - idx) / max(fo, 1)
        else:
            out.append(f)
            continue
        out.append((f.astype(np.float32) * alpha).astype(np.uint8))
    return out


# ---------- Main -----------------------------------------------------------

def main() -> None:
    print("[1/3] Rendering static cards...")
    title = _title_frame()
    outro = _outro_frame()

    sequence: list[np.ndarray] = []
    sequence.extend(_apply_fades(_still_frames(title, TITLE_S)))

    for i, spec in enumerate(STEPS, 1):
        print(f"[2/3] Step {i}/{len(STEPS)}  {spec.clip_stem}"
              f"  target_clip={spec.target_clip_s}s")
        card = _step_card(spec.number_label, spec.title)
        sequence.extend(_apply_fades(_still_frames(card, CARD_S)))
        clip = _clip_frames(spec)
        clip = _append_dwell(clip, spec.dwell_s)
        sequence.extend(_apply_fades(clip))

    sequence.extend(_apply_fades(_still_frames(outro, OUTRO_S)))

    total_s = len(sequence) / FPS
    print(f"[3/3] Writing {OUT_MP4.name}  "
          f"frames={len(sequence)}  duration={total_s:.2f}s")

    writer = imageio_ffmpeg.write_frames(
        str(OUT_MP4),
        (W, H),
        fps=FPS,
        codec="libx264",
        quality=7,
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)
    for f in sequence:
        writer.send(f)
    writer.close()
    print(f"done -> {OUT_MP4}")


if __name__ == "__main__":
    main()
