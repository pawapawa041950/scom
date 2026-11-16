"""モデル公式のプロンプト整形資料（skills）を LLM 整形で使う。

「LLMで整形」の「skillsを利用」が ON のとき、scom 同梱の汎用の指示文
（prompt_llm.DEFAULT_SYSTEM_*）の代わりに、表示モデルの系統ごとの公式
資料をシステムプロンプトにする。資料は app/skills/ に公式リポジトリの
ファイルをそのまま置いてある（出所とコミットは app/skills/NOTICE.md）。

  * MOSS-Image: Agent Skills 形式の公式 skill（SKILL.md + layout.md +
    compile_prompt.py）。skill はファイルとシェルを持つエージェント向け
    なので、scom では「下書き JSON だけを答えさせ、公式の
    compile_prompt.py を scom が実行する」形に読み替える。下書きは
    caption（文章）か Layout（文字やパネルの配置を枠で指定）で、Layout は
    モデルに渡す 1 行の JSON にコンパイルされる。
  * Krea-2 / FLUX.3: skill は無い。公式が配っているプロンプト拡張用の
    システムプロンプトをそのまま使う。

系統を足すときは app/skills/ に資料を置いて _SKILLS に 1 件足す。
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import modelinfo

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

KIND_SKILL = "skill"            # Agent Skills 形式（SKILL.md）
KIND_PROMPT = "system_prompt"   # 公式のプロンプト拡張用システムプロンプト


@dataclass(frozen=True)
class Skill:
    family: str      # 系統 id（app/families.py）
    title: str       # 画面・ログに出す名前
    kind: str        # KIND_SKILL | KIND_PROMPT
    source: str      # 取得元
    # 公式の実装が決定的に生成している（temperature 0）ものは貪欲法で回す。
    greedy: bool = False
    # 出力が JSON になるものは、同じ記号の繰り返しを罰しない。
    presence_penalty: Optional[float] = None


_SKILLS: dict[str, Skill] = {s.family: s for s in (
    Skill(modelinfo.MOSS, "MOSS-Image Prompt Enhancer（公式 skill）",
          KIND_SKILL,
          "https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/"
          "moss-image-prompt-enhancer",
          presence_penalty=0.0),
    Skill(modelinfo.KREA2, "Krea-2 expansion（公式の拡張用指示文）",
          KIND_PROMPT,
          "https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt"),
    Skill(modelinfo.FLUX3, "FLUX.3 prompt upsampler（公式の拡張用指示文）",
          KIND_PROMPT,
          "https://github.com/black-forest-labs/flux-3-image-klein"
          "#prompt-upsampling",
          greedy=True),
)}


def get(family: Optional[str]) -> Optional[Skill]:
    """その系統の資料（無ければ None）。同梱ファイルが欠けていても None。"""
    skill = _SKILLS.get(family or "")
    if skill is None or not _files(skill.family)[0].is_file():
        return None
    return skill


def _files(family: str) -> tuple[Path, ...]:
    if family == modelinfo.MOSS:
        d = SKILLS_DIR / "moss-image-prompt-enhancer"
        return (d / "SKILL.md", d / "references" / "layout.md",
                d / "scripts" / "compile_prompt.py")
    if family == modelinfo.KREA2:
        return (SKILLS_DIR / "krea-2" / "expansion.txt",)
    if family == modelinfo.FLUX3:
        d = SKILLS_DIR / "flux-3"
        return (d / "pup_prompts_external_vlm.json", d / "pup_prompts.json")
    return ()


@lru_cache(maxsize=None)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_front_matter(md: str) -> str:
    """SKILL.md 先頭の YAML（--- 〜 ---）を除いた本文。"""
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end >= 0:
            return md[md.index("\n", end + 1) + 1:].lstrip("\n")
    return md


def is_flux3_upsampler(llm_file: str) -> bool:
    """FLUX.3 の公式 upsampler（Qwen3.5 9B の微調整版）か。"""
    n = llm_file.lower()
    return "flux3" in n and "upsampler" in n


def _is_qwen_image_pe(llm_file: str) -> bool:
    n = llm_file.lower()
    return "pe_t2i" in n or "pe_i2i" in n


def usable(family: Optional[str], llm_file: str) -> Optional[Skill]:
    """その系統の資料を、この LLM で使えるなら返す。

    特定モデル用に微調整された LLM は学習した書式でしか答えず、他の資料の
    指示には従わない（実測: Qwen-Image 2.1 の PE は MOSS の skill で 87 秒
    かけて書式違いの出力、FLUX.3 upsampler は独自形式の枠を出す）。
    そのため PE では使わず、FLUX.3 upsampler は FLUX.3 の資料だけに使う。"""
    skill = get(family)
    if skill is None or not llm_file:
        return None
    if _is_qwen_image_pe(llm_file):
        return None
    if is_flux3_upsampler(llm_file) and skill.family != modelinfo.FLUX3:
        return None
    return skill


# ----- MOSS-Image: skill をツール無しの LLM 用に読み替える -----------------------
_MOSS_HEAD = """You are running the official "MOSS-Image Prompt Enhancer" skill inside an image-generation app. The skill text and its Layout reference follow; the section "How this app runs the skill" at the end overrides any step that needs files, a shell or a conversation.

===== SKILL.md =====
"""

_MOSS_TAIL = """
===== How this app runs the skill =====
- You have no filesystem and no shell. Do not write files and do not run compile_prompt.py: the app runs the bundled compile_prompt.py on your answer and sends the compiled prompt to MOSS-Image.
- The user's message is the image request. It may be Japanese or a comma-separated tag list. You cannot ask questions: resolve every open decision yourself.
- The language of the request is not a requested output language. As the skill says, write all descriptive prose (captions, summary) in English unless the request explicitly asks for another output language; keep text that must be visible in the image verbatim in its original language.
- Reply with ONLY the draft as one JSON object: either {{"caption":"..."}} or a Layout draft exactly as specified above. No Markdown fences, no commentary before or after, no thinking out loud.
- {canvas}
"""

_MOSS_CANVAS_OK = ("The image canvas is fixed at width={w} and height={h} "
                   "pixels. A Layout draft must use exactly these values.")
_MOSS_CANVAS_BAD = ("The image canvas is {w}x{h} pixels, which is not a "
                    "multiple of 16 on both sides: Layout is unavailable, "
                    "reply with a caption draft.")


def _moss_system(width: int, height: int) -> str:
    skill_md, layout_md, _script = _files(modelinfo.MOSS)
    ok = (width > 0 and height > 0 and width % 16 == 0 and height % 16 == 0)
    canvas = (_MOSS_CANVAS_OK if ok else _MOSS_CANVAS_BAD).format(
        w=width, h=height)
    return (_MOSS_HEAD + _strip_front_matter(_read(skill_md)).rstrip()
            + "\n\n===== references/layout.md =====\n"
            + _read(layout_md).rstrip() + "\n"
            + _MOSS_TAIL.format(canvas=canvas))


@lru_cache(maxsize=1)
def _moss_compiler():
    """公式 skill 同梱の compile_prompt.py（内容はそのまま）を読み込む。"""
    path = _files(modelinfo.MOSS)[2]
    spec = importlib.util.spec_from_file_location(
        "scom_moss_compile_prompt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_object(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _moss_finalize(text: str, width: int, height: int) -> tuple[str, str]:
    """LLM の下書き（JSON）を公式の compile_prompt でモデル用のプロンプトに
    する。Layout が検証に通らないときは全体の説明文（summary.caption）だけ
    を使う（枠を手で直した JSON は作らない）。"""
    draft = _json_object(text)
    if draft is None:
        return "", ""     # JSON でない: 呼び出し側が通常の解釈に回す
    compiler = _moss_compiler()
    is_layout = "items" in draft or "summary" in draft
    try:
        if is_layout and (draft.get("width"), draft.get("height")) != (
                width, height):
            raise ValueError(
                f"下書きのキャンバス {draft.get('width')}x{draft.get('height')}"
                f" が生成サイズ {width}x{height} と違います")
        out = compiler.compile_prompt(draft)
        if is_layout:
            return out, f"Layout（枠 {len(draft['items'])} 個）"
        return out.strip(), "caption"
    except (ValueError, TypeError, KeyError, OverflowError) as e:
        summary = draft.get("summary")
        caption = (summary.get("caption") if isinstance(summary, dict)
                   else draft.get("caption"))
        if isinstance(caption, str) and caption.strip():
            return caption.strip(), (
                f"Layout の下書きが検証に通らなかったため説明文だけを使用（{e}）")
        return "", ""


# ----- 公開 API -----------------------------------------------------------------
def system_prompt(family: str, llm_file: str, width: int, height: int) -> str:
    """その系統の資料から作るシステムプロンプト。"""
    if family == modelinfo.MOSS:
        return _moss_system(int(width), int(height))
    if family == modelinfo.KREA2:
        return _read(_files(family)[0]).strip()
    if family == modelinfo.FLUX3:
        external, tuned = _files(family)
        # 公式 upsampler（微調整済み）には学習時と同じ短い指示文、それ以外の
        # LLM には外部 VLM 用の詳しい指示文（作例つき）。
        path = tuned if is_flux3_upsampler(llm_file) else external
        return str(json.loads(_read(path))["t2i"]).strip()
    raise KeyError(family)


def finalize(family: str, text: str, width: int, height: int) -> tuple[str, str]:
    """思考ブロックを除いた LLM の出力 → (モデルに渡すプロンプト, 補足)。
    ("", "") は「この資料固有の解釈はできなかった」（通常の解釈に回す）。"""
    if family == modelinfo.MOSS:
        return _moss_finalize(text, int(width), int(height))
    return "", ""
