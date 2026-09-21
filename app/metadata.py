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
    ]
    if meta.get("hires_upscale"):
        # webui の Hires fix と同じフィールド名（各種ビューアが解釈する）。
        fields.append(f"Denoising strength: {meta.get('denoising_strength')}")
        fields.append(f"Hires upscale: {meta.get('hires_upscale')}")
        fields.append(f"Hires steps: {meta.get('hires_steps')}")
        fields.append(f"Hires upscaler: {meta.get('hires_method')}")
    if meta.get("loras"):
        fields.append(f"Lora: {meta.get('loras')}")
    if meta.get("lora_hashes"):
        # webui と同じクォート付き形式（civitai 等がリソース照合に使う）。
        fields.append(f"Lora hashes: \"{meta.get('lora_hashes')}\"")
    if meta.get("prompt_llm"):
        # LLM 整形: 使ったモデルと人間が書いた元のプロンプト（カンマや改行を
        # 含むので JSON 文字列としてクォートする）。
        fields.append(f"Prompt LLM: {meta.get('prompt_llm')}")
        fields.append("Original prompt: "
                      + json.dumps(str(meta.get("prompt_original", "")),
                                   ensure_ascii=False))
    # Marks the generating app (A1111/Civitai-style Version token).
    fields.append(f"Version: {APP_SIGNATURE}")
    lines.append(", ".join(str(x) for x in fields))
    return "\n".join(lines)


def _exif_bytes(params_text: str) -> bytes:
    uc = piexif.helper.UserComment.dump(params_text, encoding="unicode")
    return piexif.dump({
        "0th": {piexif.ImageIFD.Software: APP_SIGNATURE.encode("ascii")},
        "Exif": {piexif.ExifIFD.UserComment: uc},
    })


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """JPEG 用: アルファ付き画像（Qwen-Image 2.1 は RGBA 出力）は白地に
    合成してから RGB にする。単に convert("RGB") すると透明部分の下の色
    （黒やゴミ）がそのまま出てしまう。PNG/WebP はアルファのまま保存。"""
    if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def save_with_metadata(png_bytes: bytes, path: Path, fmt: str, quality: int,
                       params_text: str, extra: Optional[dict] = None,
                       comfy_prompt: Optional[dict] = None,
                       embed: bool = True) -> None:
    """Save ``png_bytes`` (decoded) to ``path`` as ``fmt`` with embedded metadata.

    ``quality`` is the PNG compress level (0-9) for PNG, or the 1-100 quality
    for JPEG/WEBP. ``embed=False`` はメタ情報（parameters/EXIF/Software 等）を
    一切書き込まないプレーン保存。
    """
    img = Image.open(io.BytesIO(png_bytes))
    if fmt == "png":
        info = None
        if embed:
            info = PngImagePlugin.PngInfo()
            info.add_text("Software", APP_SIGNATURE)
            info.add_text("parameters", params_text)
            if extra:
                info.add_text("scom", json.dumps(extra, ensure_ascii=False))
            if comfy_prompt:
                info.add_text("prompt", json.dumps(comfy_prompt))
        img.save(str(path), "PNG", compress_level=int(quality), pnginfo=info)
    elif fmt == "jpg":
        kw = {"exif": _exif_bytes(params_text)} if embed else {}
        _flatten_alpha(img).save(str(path), "JPEG", quality=int(quality), **kw)
    elif fmt == "webp":
        kw = {"exif": _exif_bytes(params_text)} if embed else {}
        img.save(
            str(path), "WEBP", quality=int(quality),
            lossless=(int(quality) >= 100), **kw
        )
    else:  # pragma: no cover - unknown format
        img.save(str(path))
