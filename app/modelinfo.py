"""Identify a model file's family (anima / krea2) from its safetensors header.

Files stay where ComfyUI already keeps them (``models/<kind>/``). To decide
which preset a file belongs to we read only the *safetensors header* — the
tensor names and shapes, a few KB–MB at the start of the file — never the
weights. This makes detection independent of the filename, so a model the user
downloaded under any name is still sorted correctly.

Signatures (confirmed against the real published files):
  * text encoder: a vision tower (``visual``/``vision`` keys) or hidden dim
    2560 -> krea2 (Qwen3-VL 4B); hidden dim 1024 -> anima (Qwen3 0.6B).
  * diffusion:    top-level keys ``txtfusion``/``tmlp``/``tproj`` -> krea2
    (~12B DiT); top-level ``net`` -> anima. Param count is the fallback.
  * vae:          the Qwen-Image VAE is shared by both families.

Results are cached per (path, size, mtime) so repeated preset switches don't
re-read headers.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

ANIMA = "anima"
KREA2 = "krea2"
SHARED = "shared"   # belongs to both families (e.g. the common Qwen-Image VAE)
UNKNOWN = "unknown"

# Refuse to allocate for an absurd/garbage header length.
_HEADER_MAX = 100_000_000

# path -> (size, mtime, family)
_cache: dict[str, tuple[int, float, str]] = {}


def _read_header(path: Path) -> Optional[dict]:
    """Return the parsed safetensors header dict, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            raw_n = f.read(8)
            if len(raw_n) < 8:
                return None
            n = struct.unpack("<Q", raw_n)[0]
            if n <= 0 or n > _HEADER_MAX:
                return None
            data = f.read(n)
        if len(data) < n:
            return None
        return json.loads(data)
    except (OSError, ValueError):
        return None


def _param_count(h: dict, keys: list[str]) -> int:
    total = 0
    for k in keys:
        shape = h[k].get("shape") or []
        p = 1
        for d in shape:
            p *= d
        total += p
    return total


def _classify_header(kind: str, h: dict) -> str:
    keys = [k for k in h if k != "__metadata__"]
    if not keys:
        return UNKNOWN

    if kind == "vae":
        return SHARED  # both families use the same Qwen-Image VAE

    if kind == "text_encoders":
        if any("visual" in k or "vision" in k for k in keys):
            return KREA2
        hidden = None
        for k in keys:
            if k.endswith("embed_tokens.weight"):
                shape = h[k].get("shape") or []
                if len(shape) == 2:
                    hidden = shape[1]
                break
        if hidden == 2560:
            return KREA2
        if hidden == 1024:
            return ANIMA
        p = _param_count(h, keys)
        if p >= 2_500_000_000:
            return KREA2
        if p <= 1_500_000_000:
            return ANIMA
        return UNKNOWN

    if kind == "diffusion_models":
        prefixes = {k.split(".")[0] for k in keys}
        if {"txtfusion", "tmlp", "tproj"} & prefixes:
            return KREA2
        if "net" in prefixes:
            return ANIMA
        p = _param_count(h, keys)
        if p >= 8_000_000_000:
            return KREA2
        if p <= 4_000_000_000:
            return ANIMA
        return UNKNOWN

    return UNKNOWN


def _filename_family(kind: str, name: str) -> str:
    """Last-resort guess from the filename (non-safetensors / unreadable file)."""
    n = name.lower()
    if kind == "vae":
        return SHARED if "qwen_image" in n else UNKNOWN
    if "krea2" in n or "krea-2" in n:
        return KREA2
    if kind == "text_encoders" and "qwen3vl" in n:
        return KREA2
    if "anima" in n:
        return ANIMA
    if kind == "text_encoders" and ("qwen_3" in n or "qwen3" in n):
        return ANIMA
    return UNKNOWN


def family(kind: str, path: Path) -> str:
    """Return 'anima' | 'krea2' | 'shared' | 'unknown' for the model file.

    Reads the safetensors header (cached by size+mtime); falls back to the
    filename when the file isn't safetensors or the header can't be parsed.
    """
    try:
        st = path.stat()
    except OSError:
        return UNKNOWN
    key = str(path)
    cached = _cache.get(key)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime:
        return cached[2]

    fam = UNKNOWN
    if path.suffix.lower() == ".safetensors":
        h = _read_header(path)
        if h is not None:
            fam = _classify_header(kind, h)
    if fam == UNKNOWN:
        fam = _filename_family(kind, path.name)

    _cache[key] = (st.st_size, st.st_mtime, fam)
    return fam
