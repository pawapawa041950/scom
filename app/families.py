"""Model families (表示モデル): everything scom needs to know per family.

A *family* is an image-model architecture scom supports (anima, krea2,
Qwen-Image 2.1, FLUX.3, SDXL ...). Its files are recognised from their safetensors
headers in :mod:`app.modelinfo`; this module is the single table of what each
family *requires* and *prefers*, so the UI, generation pre-flight checks, XYZ
cell alignment, LLM prompt style and the download presets all read the same
definition instead of hard-coding family names.

Adding a family:
  1. teach :mod:`app.modelinfo` to recognise its diffusion / TE / VAE / LoRA
     files (header signatures) and return the new id;
  2. add its download entries to :mod:`app.bootstrap.models`;
  3. append a :class:`Family` to ``FAMILIES`` below.
Everything else (preset dropdown, filtering, defaults, auto-alignment,
warnings, XYZ, merge/LoRA family checks) follows from the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import modelinfo

# modelinfo の判定結果のうち、系統ではない特別な値。
SHARED = modelinfo.SHARED    # anima/krea2 共通の Qwen-Image VAE
OTHER = modelinfo.OTHER      # 構造は分かったが対応する系統が無い
UNKNOWN = modelinfo.UNKNOWN  # 判定できない

STYLE_NATURAL = "natural"    # LLM 整形: 自然文
STYLE_TAGS = "tags"          # LLM 整形: タグ列


@dataclass(frozen=True)
class Family:
    id: str                   # 系統 id（modelinfo の判定結果・設定値と同じ）
    title: str                # 警告文などに出す名前
    clip_type: str            # CLIPLoader / DualCLIPLoader の type
    # True: text encoder を 2 個（DualCLIPLoader）。今は SDXL（clip_l+clip_g）。
    dual_te: bool = False
    # True: anima/krea2 共通の Qwen-Image VAE（modelinfo.SHARED）も使える。
    shared_vae: bool = False
    # 他系統と判定される text encoder も使える（FLUX.3 は krea2 と同じ
    # Qwen3-VL 4B を使うので、その TE は krea2 と判定される）。
    te_from: tuple[str, ...] = ()
    # False: 既定の clip_type 以外でも、他の系統専用の type でなければ可
    # （anima: stable_diffusion 等の汎用 type で動く）。
    strict_clip: bool = True
    # 生成前チェック（None = その項目は確認しない）。文言はそのまま表示する。
    te_warning: Optional[str] = None
    vae_warning: Optional[str] = None
    # モデル選択の既定: 名前にこれを含む diffusion を優先する。
    prefer: tuple[str, ...] = ()
    # 同じく text encoder（1 個目）の既定。空なら一覧の先頭。
    te_prefer: tuple[str, ...] = ()
    # 表示モデルを初めて開いたときの生成設定（steps / cfg / sampler /
    # scheduler）。保存済みの設定があればそちらが優先。
    first_defaults: dict = field(default_factory=dict)
    # diffusion のファイル名（小文字）に含まれる文字列ごとの上書き:
    # ((部分文字列, {steps/cfg/...}), ...)。最初に一致したものを使う。
    # 蒸留版と非蒸留版で steps / cfg が大きく違う系統用（FLUX.3）。
    variant_defaults: tuple[tuple[str, dict], ...] = ()
    llm_style: str = STYLE_NATURAL
    # 「設定…」の必須モデル一式: (表示名, [ファイル名 ...])。ファイルは
    # bootstrap/models.py のマニフェストにある名前。
    downloads: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def vae_families(self) -> tuple[str, ...]:
        """VAE として受け付ける判定結果。"""
        return (self.id, SHARED) if self.shared_vae else (self.id,)

    @property
    def te_families(self) -> tuple[str, ...]:
        """text encoder として受け付ける判定結果。"""
        return (self.id,) + self.te_from

    def allowed(self, kind: str) -> tuple[str, ...]:
        """``kind``（models/ のフォルダ名）のファイルとして受け付ける判定結果。"""
        if kind == "vae":
            return self.vae_families
        if kind == "text_encoders":
            return self.te_families
        return (self.id,)

    def defaults_for(self, diffusion: str) -> dict:
        """first_defaults に、diffusion の名前に合う variant_defaults を重ねたもの。"""
        d = dict(self.first_defaults)
        n = (diffusion or "").lower()
        for sub, over in self.variant_defaults:
            if sub in n:
                d.update(over)
                break
        return d


# 表示モデルのドロップダウンの並び順。先頭が既定（不明時の移行先）。
FAMILIES: tuple[Family, ...] = (
    Family(
        id=modelinfo.ANIMA, title="anima", clip_type="stable_diffusion",
        shared_vae=True, strict_clip=False,
        prefer=("base",),
        llm_style=STYLE_TAGS,
        downloads=(("Anima 必須モデル", (
            "anima-base-v1.0.safetensors",
            "qwen_image_vae.safetensors",
            "qwen_3_06b_base.safetensors",
            "lighttaew2_1.safetensors",       # 生成中プレビュー用の小型 VAE
        )),),
    ),
    Family(
        id=modelinfo.KREA2, title="Krea-2", clip_type="krea2",
        shared_vae=True,
        te_warning=("・Text encoder に Krea-2 用（Qwen3-VL）を選択してください"
                    "（未取得なら「設定…」からダウンロード）"),
        prefer=("turbo",),
        downloads=(("Krea2 必須モデル int8convrot", (
            "krea2_turbo_int8_convrot.safetensors",
            "qwen_image_vae.safetensors",
            "qwen3vl_4b_fp8_scaled.safetensors",
            "lighttaew2_1.safetensors",
        )),),
    ),
    Family(
        id=modelinfo.QWEN21, title="Qwen-Image 2.1", clip_type="qwen_image",
        te_warning=("・Text encoder に Qwen-Image 2.1 用（Qwen3-VL 8B）を"
                    "選択してください（未取得なら「設定…」からダウンロード）"),
        vae_warning=("・VAE に Qwen-Image 2.1 用（qwen_image_2.1_vae）を"
                     "選択してください（anima/krea2 の VAE とは非互換）"),
        prefer=("int8",),
        # ComfyUI 公式テンプレの既定（euler / simple / 25 steps / cfg 1）。
        first_defaults={"steps": 25, "cfg": 1.0, "sampler": "euler",
                        "scheduler": "simple"},
        downloads=(("Qwen-Image 2.1 必須モデル int8convrot", (
            "qwen_image_2.1_int8_convrot.safetensors",
            "qwen_image_2.1_vae_bf16.safetensors",
            "qwen3vl_8b_int8_convrot.safetensors",
            "taeqi2_1.safetensors",
        )),),
    ),
    Family(
        id=modelinfo.FLUX3, title="FLUX.3", clip_type="flux3",
        te_from=(modelinfo.KREA2,),   # Qwen3-VL 4B（krea2 と共通）
        te_warning=("・Text encoder に FLUX.3 用（Qwen3-VL 4B）を選択して"
                    "ください（未取得なら「設定…」からダウンロード）"),
        vae_warning=("・VAE に FLUX.3 用（flux2-vae）を選択してください"),
        # 蒸留版（_sd_）を優先。公式の既定: 蒸留版 16 steps・ガイダンス無し、
        # Base 50 steps・CFG 3.5（いずれも euler / simple）。
        prefer=("_sd_",),
        te_prefer=("qwen3vl_4b_int8", "qwen3vl_4b"),
        first_defaults={"steps": 50, "cfg": 3.5, "sampler": "euler",
                        "scheduler": "simple"},
        variant_defaults=(("_sd_", {"steps": 16, "cfg": 1.0}),),
        downloads=(("FLUX.3 Klein 必須モデル int8convrot（要 HF 認証）", (
            "flux3_image_klein_sd_int8_convrot.safetensors",
            "qwen3vl_4b_int8_convrot.safetensors",
            "flux2-vae.safetensors",
            "taef2.safetensors",
        )),),
    ),
    Family(
        id=modelinfo.SDXL, title="SDXL", clip_type="sdxl", dual_te=True,
        te_warning=("・「2つ目の text encoder を使用」を ON にし、"
                    "Text encoder 1/2 に SDXL 用（clip_l と clip_g）を"
                    "選択してください"),
        vae_warning="・VAE に SDXL 用（sdxl_vae 等）を選択してください",
        llm_style=STYLE_TAGS,
        # WAI はフル SDXL チェックポイント（VAE/CLIP 内蔵）なので本体（+
        # プレビュー用の小型 VAE）のみ。
        downloads=(("SDXL WAI-illustrious-SDXL 必須モデル", (
            "waiIllustriousSDXL_v170.safetensors",
            "taesdxl_decoder.safetensors",
        )),),
    ),
)

_BY_ID = {f.id: f for f in FAMILIES}
IDS: tuple[str, ...] = tuple(f.id for f in FAMILIES)
DEFAULT_ID: str = FAMILIES[0].id


def get(family_id: Optional[str]) -> Optional[Family]:
    """Family for an id (None for shared/other/unknown or anything else)."""
    return _BY_ID.get(family_id or "")


def is_family(family_id: Optional[str]) -> bool:
    return (family_id or "") in _BY_ID


def owned_clip_types() -> set[str]:
    """CLIP types that belong to one specific family (a non-strict family must
    not end up with another family's type)."""
    return {f.clip_type for f in FAMILIES if f.strict_clip}


def clip_ok(fam: Family, clip_type: str) -> bool:
    """Is ``clip_type`` usable for ``fam``?"""
    if fam.strict_clip:
        return clip_type == fam.clip_type
    return clip_type not in owned_clip_types()


def label(family_id: Optional[str]) -> str:
    """Short label for a modelinfo result (LoRA window etc.)."""
    if family_id == OTHER:
        return "対応外"
    if is_family(family_id):
        return str(family_id)
    return "不明"


def llm_style(family_id: Optional[str]) -> str:
    fam = get(family_id)
    return fam.llm_style if fam else STYLE_NATURAL


def download_presets() -> list[tuple[str, list[str]]]:
    """(表示名, ファイル名リスト) for every family, in table order."""
    return [(name, list(files)) for f in FAMILIES for name, files in f.downloads]
