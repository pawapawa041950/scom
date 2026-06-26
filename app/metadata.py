"""Embed generation metadata into saved images (all formats).

Uses the widely-recognized Automatic1111 ``parameters`` convention so other
tools can read it back:
  * PNG  -> a ``parameters`` tEXt chunk (+ a ``scom`` JSON chunk, and the
    ComfyUI API graph under ``prompt`` so the file can be dropped into ComfyUI)
  * JPEG -> EXIF UserComment
  * WEBP -> EXIF UserComment
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

from .config import APP_SIGNATURE

try:
    from PIL import Image, PngImagePlugin
    import piexif
    import piexif.helper
    AVAILABLE = True
except Exception:  # pragma: no cover - Pillow/piexif missing
    AVAILABLE = False


def build_parameters(meta: dict) -> str:
    """Render an Automatic1111-style parameters string."""
    prompt = str(meta.get("prompt", "")).strip()
    negative = str(meta.get("negative", "")).strip()
    lines = [prompt]
    if negative:
        lines.append(f"Negative prompt: {negative}")
    fields = [
        f"Steps: {meta.get('steps')}",
        f"Sampler: {meta.get('sampler')}",
        f"Schedule type: {meta.get('scheduler')}",
        f"CFG scale: {meta.get('cfg')}",
        f"Seed: {meta.get('seed')}",
        f"Size: {meta.get('width')}x{meta.get('height')}",
        f"Model: {meta.get('model')}",
        f"VAE: {meta.get('vae')}",
        f"Text encoder: {meta.get('text_encoder')}",
        f"CLIP type: {meta.get('clip_type')}",
        f"Batch size: {meta.get('batch')}",
        f"Weight dtype: {meta.get('dtype')}",
        # Marks the generating app (A1111/Civitai-style Version token).
        f"Version: {APP_SIGNATURE}",
    ]
    lines.append(", ".join(str(x) for x in fields))
    return "\n".join(lines)


def _exif_bytes(params_text: str) -> bytes:
    uc = piexif.helper.UserComment.dump(params_text, encoding="unicode")
    return piexif.dump({
        "0th": {piexif.ImageIFD.Software: APP_SIGNATURE.encode("ascii")},
        "Exif": {piexif.ExifIFD.UserComment: uc},
    })


def save_with_metadata(png_bytes: bytes, path: Path, fmt: str, quality: int,
                       params_text: str, extra: Optional[dict] = None,
                       comfy_prompt: Optional[dict] = None) -> None:
    """Save ``png_bytes`` (decoded) to ``path`` as ``fmt`` with embedded metadata.

    ``quality`` is the PNG compress level (0-9) for PNG, or the 1-100 quality
    for JPEG/WEBP.
    """
    img = Image.open(io.BytesIO(png_bytes))
    if fmt == "png":
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", APP_SIGNATURE)
        info.add_text("parameters", params_text)
        if extra:
            info.add_text("scom", json.dumps(extra, ensure_ascii=False))
        if comfy_prompt:
            info.add_text("prompt", json.dumps(comfy_prompt))
        img.save(str(path), "PNG", compress_level=int(quality), pnginfo=info)
    elif fmt == "jpg":
        img.convert("RGB").save(
            str(path), "JPEG", quality=int(quality), exif=_exif_bytes(params_text)
        )
    elif fmt == "webp":
        img.save(
            str(path), "WEBP", quality=int(quality),
            exif=_exif_bytes(params_text), lossless=(int(quality) >= 100),
        )
    else:  # pragma: no cover - unknown format
        img.save(str(path))
