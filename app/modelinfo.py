"""Identify a model file's family (anima / krea2 / qwen21 / flux3 / sdxl) from its
safetensors header.

Files stay where ComfyUI already keeps them (``models/<kind>/``). To decide
which preset a file belongs to we read only the *safetensors header* — the
tensor names and shapes, a few KB–MB at the start of the file — never the
weights. This makes detection independent of the filename, so a model the user
downloaded under any name is still sorted correctly.

Signatures (confirmed against the real published files):
  * text encoder: a vision tower (``visual``/``vision`` keys) -> Qwen3-VL:
    merger/hidden dim 2560 -> krea2 (4B), 4096 -> qwen21 (8B);
    hidden dim 1024 -> anima (Qwen3 0.6B). FLUX.3 uses the same Qwen3-VL 4B
    as krea2, so its TE is reported as krea2 (families.py lets flux3 use it).
  * diffusion:    top-level keys ``txtfusion``/``tmlp``/``tproj`` -> krea2
    (~12B DiT); ``txt_in.text_norm`` -> qwen21 (Qwen-Image 2.1, 7B);
    top-level ``net`` -> anima; ``content_mode_blocks`` +
    ``single_stream_modulations`` -> flux3. Param count is the fallback.
  * vae:          the Qwen-Image VAE is shared by anima/krea2; Qwen-Image 2.1
    has its own 64ch/16x RGBA VAE (Wan2.2 layout, temporal kernel 1);
    the FLUX.2 VAE (32ch, ``bn`` latent BatchNorm) -> flux3; kl-f8 (4ch) -> sdxl.

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
QWEN21 = "qwen21"   # Qwen-Image 2.1（7B DiT / Qwen3-VL 8B TE / 専用 64ch VAE）
FLUX3 = "flux3"     # FLUX.3（DiT / Qwen3-VL 4B TE = krea2 と同じ / FLUX.2 VAE）
SDXL = "sdxl"
SHARED = "shared"   # anima/krea2 共通（Qwen-Image VAE）。sdxl には含まれない。
OTHER = "other"     # アーキテクチャは特定できたが、どのプリセットでも使えない
                    # （SD1.x/SD2.x・Flux・Qwen-Image 20B などの LoRA）
UNKNOWN = "unknown"

# Refuse to allocate for an absurd/garbage header length.
_HEADER_MAX = 100_000_000

# path -> (size, mtime, family)
# 起動をまたいで userdata/model_families.json に永続化する。LoRA が数百本
# あるとヘッダ読み（ファイル先頭の数百KB〜数MB × 本数）だけで初回は
# 数秒〜十数秒かかるため、判定済みの結果は二度と読み直さない。
_cache: dict[str, tuple[int, float, str]] = {}
_cache_loaded = False
_CACHE_NAME = "model_families.json"
# 判定規則を変えたら上げ、_stale() に「結果が変わり得る判定」を足す
# （それだけ読み直す。全部捨てると LoRA 数百本の読み直しになるうえ、
# 判定待ちの間は XYZ の LoRA 候補が絞り込まれない）。
# 2: FLUX.3 対応（FLUX.2 VAE が sdxl、FLUX.3 本体が unknown、FLUX.3 の
#    LoRA が other と判定されていた）。
_CACHE_VERSION = 2
_VERSION_KEY = "__version__"


def _cache_path() -> Optional[Path]:
    try:
        from . import config
        return config.AppPaths().user_data / _CACHE_NAME
    except Exception:  # noqa: BLE001 - キャッシュ無しでも動作は変わらない
        return None


def _load_cache() -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    p = _cache_path()
    if p is None:
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    version = data.get(_VERSION_KEY)
    for k, v in data.items():
        if (isinstance(v, list) and len(v) == 3
                and isinstance(v[2], str)):
            if version != _CACHE_VERSION and _stale(str(k), v[2], version):
                continue
            try:
                _cache[str(k)] = (int(v[0]), float(v[1]), v[2])
            except (TypeError, ValueError):
                continue


def _stale(path: str, fam: str, version) -> bool:
    """古い版で判定した結果のうち、今の規則だと変わり得るもの。"""
    if not isinstance(version, int) or version < 2:
        parts = {s.lower() for s in Path(path).parts}
        if fam in (UNKNOWN, OTHER):
            return True
        if "vae" in parts and fam == SDXL:
            return True
    return False


def _save_cache() -> None:
    p = _cache_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        data: dict = {_VERSION_KEY: _CACHE_VERSION}
        data.update({k: list(v) for k, v in _cache.items()})
        tmp.write_text(json.dumps(data, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass


def cached_family(path: Path) -> Optional[str]:
    """判定済み（size+mtime 一致）ならその系統、未判定なら None。ヘッダは
    読まないので GUI スレッドから何百回呼んでも即座に返る。"""
    _load_cache()
    try:
        st = path.stat()
    except OSError:
        return None
    cached = _cache.get(str(path))
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime:
        return cached[2]
    return None


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


def _lora_in_features(h: dict, keys: list[str], *substrs: str) -> Optional[int]:
    """in_features of the first LoRA down-projection matching all substrings.

    kohya は ``<module>.lora_down.weight``、diffusers/PEFT は
    ``<module>.lora_A.weight``。どちらも形状 [rank, in_features] なので、
    接続先モジュールの入力次元がヘッダだけで分かる。キーは '.'→'_' 正規化
    して照合する（kohya はモジュール名の '.' を '_' に潰すため）。
    """
    for k in keys:
        norm = k.replace(".", "_")
        if not (norm.endswith("_lora_down_weight")
                or norm.endswith("_lora_A_weight")):
            continue
        if all(s in norm for s in substrs):
            shape = h[k].get("shape") or []
            if len(shape) == 2:
                return shape[1]
    return None


def _classify_lora(h: dict, keys: list[str]) -> str:
    """LoRA ファイルの対応系統をキー名+形状から判定する。

    LoRA のキーには学習対象モジュールのパスがそのまま残る（kohya:
    lora_unet_<path>_..., diffusers: transformer.<path>.lora_A.weight,
    comfy: diffusion_model.<path>）ので、ベースモデルのアーキテクチャが
    特定できる。判定できたが対応プリセットが無いものは OTHER（フィルタで
    非表示）、構造を認識できないものは UNKNOWN（常に表示 = 安全側）。
    """
    norm_all = " ".join(k.replace(".", "_") for k in keys)

    # krea2: txtfusion（ネイティブ名）/ text_fusion（diffusers 名）は固有。
    if "txtfusion" in norm_all or "text_fusion" in norm_all:
        return KREA2

    # anima: blocks.N.cross_attn.q_proj（hidden 2048）と llm_adapter が固有。
    if "cross_attn_q_proj" in norm_all or "llm_adapter" in norm_all:
        d = _lora_in_features(h, keys, "cross_attn_q_proj")
        if d is not None and d != 2048:
            return OTHER  # cross_attn.q_proj を持つ別アーキテクチャ
        return ANIMA

    # SD-UNet（kohya の LDM 名 input_blocks / diffusers の down_blocks）。
    # attn2（cross-attention）の入力次元が context dim: 2048=SDXL,
    # 768=SD1.x, 1024=SD2.x。
    if (any(m in norm_all for m in ("input_blocks", "output_blocks",
                                    "middle_block"))
            or ("down_blocks" in norm_all and "attentions" in norm_all)):
        d = (_lora_in_features(h, keys, "attn2_to_k")
             or _lora_in_features(h, keys, "attn2_to_v"))
        if d == 2048:
            return SDXL
        if d is not None:
            return OTHER  # SD1.x / SD2.x
        # UNet 側に手掛かりなし（attn2 を学習していない LoRA）。
        # SDXL だけが text encoder を2つ持つ。
        return SDXL if "lora_te2" in norm_all else UNKNOWN

    if "transformer_blocks" in norm_all:
        # Qwen-Image (20B) / Flux(diffusers名): joint-attention の
        # add_*_proj や img_mod/txt_mod。anima とは名前も次元も別物。
        if any(m in norm_all for m in ("add_q_proj", "add_k_proj",
                                       "img_mod", "txt_mod")):
            return OTHER
        # Qwen-Image 2.1: 単一ストリームで attn.to_q + img_mlp（gate_up /
        # proj / out）だけを持つ。hidden 4096（20B は 3072）で確認する。
        if "img_mlp" in norm_all or "attn_to_q" in norm_all:
            d = (_lora_in_features(h, keys, "attn_to_q")
                 or _lora_in_features(h, keys, "img_mlp_gate_up")
                 or _lora_in_features(h, keys, "img_mlp_proj"))
            if d == 4096:
                return QWEN21
            if d is not None:
                return OTHER
            return UNKNOWN
        # krea2 (diffusers名) 固有の gate 付き attention/FF。
        if "to_gate" in norm_all or "ff_gate" in norm_all:
            return KREA2
        return UNKNOWN

    # FLUX.3: モード別ブロック（content_mode_blocks / txt_mode_blocks）と
    # attn_out / mlp_in / mlp_out。FLUX.1/2 の single_blocks は linear1/2。
    if any(m in norm_all for m in ("content_mode_blocks", "txt_mode_blocks",
                                   "single_stream_modulations")):
        return FLUX3
    for k in keys:
        norm = k.replace(".", "_")
        if "single_blocks_" in norm and any(
                m in norm for m in ("_attn_out_", "_mlp_in_", "_mlp_out_")):
            return FLUX3

    if "double_blocks" in norm_all or "single_blocks" in norm_all:
        return OTHER  # Flux / HunyuanVideo 系（kohya 名）

    return UNKNOWN


# Qwen3.5 系（線形 attention 層を持つ）= 文章生成用 LLM（Qwen-Image 2.1 の
# プロンプト整形モデル PE もこれ）。置き場は models/llm だが、間違えて
# text_encoders に置かれても TE 候補には出さない。
_QWEN35_KEY = "model.language_model.layers.0.linear_attn.A_log"


def _is_text_llm_header(keys) -> bool:
    return _QWEN35_KEY in keys


def _classify_header(kind: str, h: dict) -> str:
    keys = [k for k in h if k != "__metadata__"]
    if not keys:
        return UNKNOWN

    if kind == "loras":
        return _classify_lora(h, keys)

    if kind == "vae":
        # FLUX.2 VAE: 32ch の潜在 + 潜在を正規化する BatchNorm（bn.*）。
        conv_in = (h.get("decoder.conv_in.weight") or {}).get("shape") or []
        if "bn.running_mean" in h or (len(conv_in) == 4 and conv_in[1] == 32):
            return FLUX3
        # kl-f8 (SD/SDXL) VAE: 4ch の潜在（quant_conv / decoder.conv_in）。
        if "post_quant_conv.weight" in h or (len(conv_in) == 4
                                             and conv_in[1] == 4):
            return SDXL
        if len(conv_in) == 4:
            return OTHER  # FLUX.1 / SD3（16ch）などの 2D VAE
        # Qwen-Image 2.1 VAE: Wan2.2 配置（upsamples 二段）で出力ヘッドが
        # 時間カーネル 1 の RGBA（ComfyUI sd.py と同じ判定）。16 倍圧縮・
        # 64ch なので anima/krea2 の Qwen-Image VAE とは互換が無い。
        head = (h.get("decoder.head.2.weight") or {}).get("shape") or []
        if ("decoder.upsamples.0.upsamples.0.residual.2.weight" in h
                and len(head) == 5 and head[2] == 1):
            return QWEN21
        return SHARED  # Qwen-Image VAE (anima/krea2 共通) ほか

    if kind == "text_encoders":
        if _is_text_llm_header(keys):
            return OTHER   # プロンプト整形用 LLM（TE の候補には出さない）
        if any("visual" in k or "vision" in k for k in keys):
            # Qwen3-VL: 4B (2560) = krea2、8B (4096) = Qwen-Image 2.1。
            # ComfyUI と同じく merger の出力次元で見る（無ければ embed）。
            dim = None
            for k in keys:
                if k.endswith("visual.merger.linear_fc2.weight"):
                    shape = h[k].get("shape") or []
                    dim = shape[0] if len(shape) == 2 else None
                    break
            if dim is None:
                for k in keys:
                    if k.endswith("embed_tokens.weight"):
                        shape = h[k].get("shape") or []
                        dim = shape[1] if len(shape) == 2 else None
                        break
            return QWEN21 if dim == 4096 else KREA2
        # CLIP (SDXL の clip_l/clip_g)。transformers 形式は text_model.*、
        # チェックポイントから取り出した clip_g は open_clip 形式
        # (transformer.resblocks.*)。
        if any(k.startswith("text_model.")
               or "transformer.resblocks." in k for k in keys):
            return SDXL
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
        # Qwen-Image 2.1（7B, 単一ストリーム）: txt_in.text_norm が固有
        # （20B の Qwen-Image は txt_norm がトップレベル）。
        if "txt_in.text_norm.weight" in h:
            return QWEN21
        # FLUX.3（ComfyUI model_detection と同じ鍵）。
        if {"content_mode_blocks", "single_stream_modulations"} <= prefixes:
            return FLUX3
        if "net" in prefixes:
            return ANIMA
        # SD/SDXL U-Net（単体ファイル / フルチェックポイントの両形式）。
        # パラメタ数フォールバックより先に判定しないと SDXL (~2.6-3.5B) が
        # anima に誤分類される。
        if any(k.startswith(("input_blocks.",
                             "model.diffusion_model.input_blocks."))
               for k in keys):
            return SDXL
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
    q21 = any(s in n for s in ("qwen_image_2.1", "qwen_image_2_1",
                               "qwen-image-2.1", "qwen_image2.1"))
    flux = "flux3" in n or "flux.3" in n or "flux-3" in n
    if kind == "vae":
        if "flux2" in n or "flux.2" in n or "flux-2" in n or flux:
            return FLUX3
        if "sdxl" in n or "sd_xl" in n:
            return SDXL
        if q21:
            return QWEN21
        return SHARED if "qwen_image" in n else UNKNOWN
    if "sdxl" in n or "sd_xl" in n:
        return SDXL
    if q21:
        return QWEN21
    if flux and kind != "text_encoders":
        return FLUX3
    if kind == "text_encoders" and ("clip_l" in n or "clip_g" in n):
        return SDXL
    if "krea2" in n or "krea-2" in n:
        return KREA2
    if kind == "text_encoders" and "qwen3vl_8b" in n:
        return QWEN21
    if kind == "text_encoders" and "qwen3vl" in n:
        return KREA2
    if "anima" in n:
        return ANIMA
    if kind == "text_encoders" and ("qwen_3" in n or "qwen3" in n):
        return ANIMA
    return UNKNOWN


def sdxl_te_kind(path: Path) -> Optional[str]:
    """SDXL 系 TE の "clip_l" / "clip_g" 判別（それ以外は None）。

    token embedding の隠れ次元で判定する: 768 -> clip_l、1280 -> clip_g
    （open_clip 形式のスタンドアロン clip_g は ``token_embedding.weight``）。
    ヘッダが読めない場合はファイル名で推測する。
    """
    h = _read_header(path) if path.suffix.lower() == ".safetensors" else None
    if h:
        v = (h.get("text_model.embeddings.token_embedding.weight")
             or h.get("token_embedding.weight"))
        shape = (v or {}).get("shape") or []
        if len(shape) == 2:
            if shape[1] == 768:
                return "clip_l"
            if shape[1] == 1280:
                return "clip_g"
    n = path.name.lower()
    if "clip_l" in n:
        return "clip_l"
    if "clip_g" in n:
        return "clip_g"
    return None


# path -> (size, mtime, has_vae, has_clip)
_builtin_cache: dict[str, tuple[int, float, bool, bool]] = {}


def builtin_components(path: Path) -> tuple[bool, bool]:
    """(has_builtin_vae, has_builtin_clip) for a diffusion file.

    Full SD/SDXL checkpoints bundle the VAE (``first_stage_model.*`` keys) and
    the CLIP text encoders (``conditioner.*`` for SDXL / ``cond_stage_model.*``
    for SD1.x). UNet-only diffusion files (anima/krea2 の Qwen-Image、split
    SDXL の UNet 単体) carry neither. Header-only, cached by size+mtime."""
    try:
        st = path.stat()
    except OSError:
        return (False, False)
    key = str(path)
    c = _builtin_cache.get(key)
    if c and c[0] == st.st_size and c[1] == st.st_mtime:
        return (c[2], c[3])
    has_vae = has_clip = False
    if path.suffix.lower() == ".safetensors":
        h = _read_header(path)
        if h:
            has_vae = any(k.startswith("first_stage_model.") for k in h)
            has_clip = any(k.startswith(("conditioner.", "cond_stage_model."))
                           for k in h)
    _builtin_cache[key] = (st.st_size, st.st_mtime, has_vae, has_clip)
    return (has_vae, has_clip)


def is_checkpoint(path: Path) -> bool:
    """True if ``path`` is a full checkpoint bundling both VAE and CLIP, so it
    can be loaded via CheckpointLoaderSimple without separate VAE/TE files."""
    has_vae, has_clip = builtin_components(path)
    return has_vae and has_clip


def family(kind: str, path: Path) -> str:
    """Return 'anima' | 'krea2' | 'qwen21' | 'flux3' | 'sdxl' | 'shared' | 'other' |
    'unknown' for the model file.

    Reads the safetensors header (cached by size+mtime); falls back to the
    filename when the file isn't safetensors or the header can't be parsed.
    """
    _load_cache()
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
    _save_cache()
    return fam
