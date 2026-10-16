"""Tiny preview VAEs (TAESD family) for ComfyUI's live latent previews.

ComfyUI (``--preview-method taesd``) decodes the in-progress latents with a
tiny autoencoder it finds in ``models/vae_approx`` by name prefix, e.g.
``taeqi2_1_decoder.*`` for Qwen-Image 2.1 and ``taef2_decoder.*`` for FLUX.3
(FLUX.2 latents). When the file for a model family is
missing it falls back to the coarse latent2rgb preview, so these files are
optional.

The Qwen-Image 2.1 one is published as a single file holding encoder and
decoder (``madebyollin/taeqi2_1``: ``taeqi2_1.safetensors``, keys
``decoder.layers.N.*``). ComfyUI loads a decoder-only file into its
``DecoderF16`` whose layer 0 is an extra clamp, so ``decoder.layers.N`` maps
to ``N+1`` (verified: all 67 tensors line up by name and shape). The rewrite
below only renames keys and copies the raw tensor bytes, so it needs neither
torch nor safetensors (the GUI app bundles neither).

``madebyollin/taef2`` (FLUX.2 / FLUX.3) ships the same combined layout and
ComfyUI's ``DecoderFlux2`` has the same leading clamp, so the same +1 rename
applies (verified: all 79 tensors line up).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

VAE_APPROX = "vae_approx"   # config.MODEL_DIRS のキー（models/vae_approx）

TAEQI21_COMBINED = "taeqi2_1.safetensors"
TAEQI21_DECODER = "taeqi2_1_decoder.safetensors"
TAEF2_COMBINED = "taef2.safetensors"
TAEF2_DECODER = "taef2_decoder.safetensors"

# (combined file, decoder file ComfyUI looks for, label for the log)
_COMBINED = (
    (TAEQI21_COMBINED, TAEQI21_DECODER, "Qwen-Image 2.1"),
    (TAEF2_COMBINED, TAEF2_DECODER, "FLUX.3"),
)


def _read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def extract_taeqi21_decoder(src: Path, dst: Path) -> int:
    """Write the decoder half of ``src`` to ``dst`` in ComfyUI's layout.
    Returns the number of tensors written."""
    header, data_start = _read_header(src)
    items = []
    for key, info in header.items():
        if key == "__metadata__" or not key.startswith("decoder.layers."):
            continue
        parts = key.split(".")[2:]          # "N", "conv", "0", "weight" ...
        parts[0] = str(int(parts[0]) + 1)   # ComfyUI: layer 0 is Clamp()
        items.append((".".join(parts), info))
    if not items:
        raise ValueError(f"{src.name} にデコーダの重みがありません")
    items.sort(key=lambda kv: kv[1]["data_offsets"][0])
    out_header: dict = {"__metadata__": {"format": "pt",
                                         "source": src.name}}
    offset = 0
    for name, info in items:
        begin, end = info["data_offsets"]
        out_header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + (end - begin)]}
        offset += end - begin
    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)        # keep the data section 8-aligned
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for _name, info in items:
            begin, end = info["data_offsets"]
            fin.seek(data_start + begin)
            fout.write(fin.read(end - begin))
    tmp.replace(dst)
    return len(items)


def ensure_preview_decoders(models_root: Path, log=None) -> None:
    """Create the decoder-only files ComfyUI's previews need from the
    downloaded combined files (cheap no-op once done)."""
    folder = Path(models_root) / VAE_APPROX
    for combined, decoder, label in _COMBINED:
        src, dst = folder / combined, folder / decoder
        if not src.is_file():
            continue
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        try:
            n = extract_taeqi21_decoder(src, dst)
            if log:
                log(f"{label} のプレビュー用デコーダを準備しました（{n} テンソル）")
        except (OSError, ValueError, KeyError) as e:
            if log:
                log(f"{label} のプレビュー用デコーダの準備に失敗しました: {e}")
