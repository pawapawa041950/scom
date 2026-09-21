"""anima model manifest + downloader.

Files live under the Hugging Face repo ``circlestone-labs/Anima`` in
``split_files/{diffusion_models,vae,text_encoders}``. The manifest is written to
``userdata/models.json`` on first run so users can edit URLs / add variants
without touching code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import config
from .downloader import download

HF_BASE = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"
# Krea-2 (Comfy-Org). Files live directly under diffusion_models/ vae/
# text_encoders/ (no split_files prefix). Multiple safetensors quantizations
# are offered so users pick the size/quality that fits their VRAM.
KREA_BASE = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main"
# Qwen-Image 2.1 (Comfy-Org): 生成モデル一式（DiT / Qwen3-VL 8B TE / 専用
# VAE）と、プロンプト整形 LLM (PE)。ライセンスは Qwen Research License
# （非商用）。ComfyUI v0.37.0 以上が必要。
QWEN21_BASE = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
PE_BASE = QWEN21_BASE


@dataclass
class ModelFile:
    kind: str         # one of config.MODEL_DIRS
    filename: str
    url: str
    size: int = 0     # bytes, for progress + verification (0 = unknown)
    required: bool = True


_FIELDS = ("kind", "filename", "url", "size", "required")


def _krea(kind: str, filename: str, size: int) -> "ModelFile":
    return ModelFile(kind, filename, f"{KREA_BASE}/{kind}/{filename}",
                     size, required=False)


def _qwen21(kind: str, filename: str, size: int) -> "ModelFile":
    return ModelFile(kind, filename, f"{QWEN21_BASE}/{kind}/{filename}",
                     size, required=False)


# Defaults mirror the HF split_files layout. The base diffusion model + VAE +
# text encoder are required; the preview variants are optional extras.
DEFAULT_MODELS: list[ModelFile] = [
    ModelFile("diffusion_models", "anima-base-v1.0.safetensors",
              f"{HF_BASE}/diffusion_models/anima-base-v1.0.safetensors",
              4_182_218_328, required=True),
    ModelFile("vae", "qwen_image_vae.safetensors",
              f"{HF_BASE}/vae/qwen_image_vae.safetensors",
              253_806_246, required=True),
    ModelFile("text_encoders", "qwen_3_06b_base.safetensors",
              f"{HF_BASE}/text_encoders/qwen_3_06b_base.safetensors",
              1_192_135_096, required=True),
    ModelFile("diffusion_models", "anima-preview.safetensors",
              f"{HF_BASE}/diffusion_models/anima-preview.safetensors",
              4_182_218_360, required=False),
    ModelFile("diffusion_models", "anima-preview2.safetensors",
              f"{HF_BASE}/diffusion_models/anima-preview2.safetensors",
              4_182_218_360, required=False),
    ModelFile("diffusion_models", "anima-preview3-base.safetensors",
              f"{HF_BASE}/diffusion_models/anima-preview3-base.safetensors",
              4_182_218_360, required=False),

    # --- Krea-2 (all optional; the shared qwen_image_vae above is reused) -----
    # Turbo: 8-step distilled. nvfp4 is the smallest; bf16 the highest quality.
    _krea("diffusion_models", "krea2_turbo_nvfp4.safetensors", 7_673_668_288),
    _krea("diffusion_models", "krea2_turbo_fp8_scaled.safetensors", 13_141_730_784),
    _krea("diffusion_models", "krea2_turbo_int8_convrot.safetensors", 13_492_686_496),
    _krea("diffusion_models", "krea2_turbo_mxfp8.safetensors", 13_532_318_080),
    _krea("diffusion_models", "krea2_turbo_bf16.safetensors", 26_283_332_608),
    # Raw/Base: full-step (~52) model, higher fidelity, slower.
    _krea("diffusion_models", "krea2_raw_fp8_scaled.safetensors", 13_141_730_784),
    _krea("diffusion_models", "krea2_raw_bf16.safetensors", 26_283_332_608),
    # Krea-2 text encoder (Qwen3-VL 4B). Use CLIP type "krea2".
    _krea("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors", 5_242_467_968),
    _krea("text_encoders", "qwen3vl_4b_bf16.safetensors", 8_875_719_384),

    # --- Qwen-Image 2.1 (all optional) --------------------------------------
    # 7B の単一ストリーム DiT（2K ネイティブ・RGBA 出力）。int8 convrot は
    # 公式配布の量子化版（推奨）。
    _qwen21("diffusion_models", "qwen_image_2.1_int8_convrot.safetensors",
            7_256_783_064),
    _qwen21("diffusion_models", "qwen_image_2.1_bf16.safetensors",
            14_230_280_616),
    # Text encoder: Qwen3-VL 8B（CLIP type "qwen_image"。ComfyUI は 8B を
    # 検出すると自動で 2.1 用テンプレートに切り替える）。
    _qwen21("text_encoders", "qwen3vl_8b_int8_convrot.safetensors",
            9_350_798_360),
    _qwen21("text_encoders", "qwen3vl_8b_w4a8.safetensors", 6_312_105_364),
    _qwen21("text_encoders", "qwen3vl_8b_bf16.safetensors", 17_534_334_616),
    # 専用 VAE（64ch・16 倍圧縮・RGBA）。anima/krea2 の VAE とは非互換。
    _qwen21("vae", "qwen_image_2.1_vae_bf16.safetensors", 675_509_688),

    # --- プロンプト整形 LLM (Qwen-Image 2.1 Prompt Enhancer, T2I) ----------------
    # Qwen3.5-VL 9B の微調整モデル（int8 convrot, Comfy-Org 再パッケージ）。
    # 置き場は models/llm（app/prompt_llm.py）。隣の system_prompt.txt は本家
    # リポジトリの公式指示文で、scom は <名前>.system_prompt.txt があれば
    # それをシステムプロンプトに使う。ライセンスは Qwen Research License
    # （非商用）。
    ModelFile("llm",
              "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
              f"{PE_BASE}/text_encoders/"
              "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
              9_471_072_252, required=False),
    ModelFile("llm",
              "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.system_prompt.txt",
              "https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I/resolve/main/"
              "system_prompt.txt",
              0, required=False),
    # 汎用 LLM (Comfy-Org/Qwen3.5, bf16)。PE と違いシステムプロンプトに従う
    # ので、anima / SDXL 向けの「タグ列に変換」（scom 同梱の指示文）に使う。
    ModelFile("llm", "qwen3.5_4b_bf16.safetensors",
              "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/"
              "text_encoders/qwen3.5_4b_bf16.safetensors",
              9_319_828_320, required=False),
    ModelFile("llm", "qwen3.5_2b_bf16.safetensors",
              "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/"
              "text_encoders/qwen3.5_2b_bf16.safetensors",
              4_548_221_488, required=False),

    # --- SDXL (WAI-illustrious-SDXL preset) ----------------------------------
    # フル SDXL チェックポイント（VAE/CLIP 内蔵）。CheckpointLoaderSimple で読み、
    # 内蔵の VAE/CLIP を使うので VAE/TE は別途ダウンロードしない。civitai の
    # 最新版 (v17.0)。size=0: civitai の申告サイズと実バイトがずれると検証で
    # 失敗するため、Content-Length に任せる。
    ModelFile("diffusion_models", "waiIllustriousSDXL_v170.safetensors",
              "https://civitai.com/api/download/models/2883731",
              0, required=False),
]


def _write_manifest(mf: Path, manifest: list[ModelFile]) -> None:
    mf.write_text(json.dumps([asdict(m) for m in manifest], indent=2),
                  encoding="utf-8")


def load_manifest(paths: config.AppPaths) -> list[ModelFile]:
    """Read userdata/models.json, creating it from DEFAULT_MODELS if absent.

    New default entries shipped in later versions (e.g. Krea-2) are merged into
    an existing models.json, keyed by (kind, filename), so users who already ran
    the app see them without losing any hand-edited URLs/entries.
    """
    mf = paths.user_data / "models.json"
    if not mf.exists():
        manifest = list(DEFAULT_MODELS)
        _write_manifest(mf, manifest)
        return manifest
    data = json.loads(mf.read_text(encoding="utf-8"))
    manifest = [ModelFile(**{k: d[k] for k in _FIELDS if k in d}) for d in data]
    have = {(m.kind, m.filename) for m in manifest}
    added = [m for m in DEFAULT_MODELS if (m.kind, m.filename) not in have]
    if added:
        manifest.extend(added)
        _write_manifest(mf, manifest)
    return manifest


def target_path(paths: config.AppPaths, m: ModelFile) -> Path:
    return paths.models / m.kind / m.filename


def missing_required(paths: config.AppPaths,
                     manifest: Optional[list[ModelFile]] = None) -> list[ModelFile]:
    manifest = manifest or load_manifest(paths)
    out = []
    for m in manifest:
        if not m.required:
            continue
        p = target_path(paths, m)
        if not p.exists() or (m.size and p.stat().st_size != m.size):
            out.append(m)
    return out


def download_model(paths: config.AppPaths, m: ModelFile,
                   on_progress: Optional[Callable[[int, int], None]] = None,
                   cancel: Optional[Callable[[], bool]] = None) -> Path:
    dest = target_path(paths, m)
    return download(
        m.url, dest, on_progress=on_progress,
        expected_size=m.size or None, cancel=cancel,
    )
