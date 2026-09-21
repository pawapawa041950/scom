"""プロンプトを LLM に書き直させる（Qwen-Image 2.1 の PE モデルなど）。

ComfyUI 本体のノードだけで完結する:
  CLIPLoader(type=qwen_image) -> TextGenerate(生のチャット書式) -> PreviewAny
生成した文字列は /history から取り出す（app/comfy_backend.generate_text）。

使える LLM = models/llm/ に置いた safetensors（ComfyUI が「文章生成できる」
テキストエンコーダとして読めるもの: Qwen3.5 系 = Qwen-Image 2.1 の PE、
Gemma4、Qwen3 など）。models/llm は ComfyUI に text_encoders の追加パスとして
渡すので CLIPLoader から読める。scom の Text encoder 候補には出さない。

システムプロンプトの選び方（モデル系統ごとの「書き方」に合わせる）:
  * natural（Krea-2 / Qwen-Image 2.1 など自然文プロンプトのモデル）
      <名前>.system_prompt.txt があればそれ（PE の公式文をダウンロードして
      置く想定）、無ければ同梱の DEFAULT_SYSTEM_NATURAL。
  * tags（anima / SDXL Illustrious 系のタグ列プロンプト）
      <名前>.system_prompt_tags.txt があればそれ、無ければ DEFAULT_SYSTEM_TAGS。

出力は PE と同じ {"rewritten_prompt": "..."} 形式の JSON を期待するが、
思考ブロック・コードフェンス・生テキストにも寛容に対応する（parse_output）。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional

from . import config

LLM_KIND = "llm"   # config.MODEL_DIRS のキー（models/llm）
SUFFIX_NATURAL = ".system_prompt.txt"
SUFFIX_TAGS = ".system_prompt_tags.txt"

# 生成の再現性のため固定（同じ入力なら同じ整形結果 = キャッシュも効く）。
LLM_SEED = 42
MAX_LENGTH = 6144   # 思考ブロック込みの上限トークン。PE の答えは通常数百。

STYLE_NATURAL = "natural"
STYLE_TAGS = "tags"
_TAG_PRESETS = {"anima", "sdxl"}


def style_for_preset(preset: str) -> str:
    return STYLE_TAGS if preset in _TAG_PRESETS else STYLE_NATURAL


def is_pe_model(llm_file: str) -> bool:
    """Qwen-Image 2.1 の PE（プロンプト書き直し専用に微調整されたモデル）か。
    実測でシステムプロンプトの指示（タグ列で出せ等）を無視して自然文の
    段落しか返さないので、このモデルは常に natural として扱う。"""
    n = llm_file.lower()
    return "pe_t2i" in n or "pe_i2i" in n


def effective_style(preset: str, llm_file: str) -> str:
    if is_pe_model(llm_file):
        return STYLE_NATURAL
    return style_for_preset(preset)


DEFAULT_SYSTEM_NATURAL = """You rewrite a user's image request into a prompt for a text-to-image model that understands natural language.
Write one detailed English paragraph that describes the finished image as an observer would see it: the main subject(s) and their appearance, clothing, pose and expression; the setting and background; composition, camera framing and lighting; colors and materials; the overall style (photo, anime illustration, painting, etc.).
Rules:
- Keep every element the user asked for. Do not drop, replace or contradict anything. Invent details only where the user left them open.
- If the request is in another language (e.g. Japanese), translate it faithfully into English.
- Use concrete nouns with modifiers ("deep navy wool coat"), name positions ("on the left", "in the background"), and enumerate items instead of summarizing ("various decorations").
- No quality boosters such as "masterpiece", "best quality", "8k". No negative prompts. No camera brand names unless requested.
- Present tense, third person. 60-180 words.
Respond with ONLY a JSON object of the form {"rewritten_prompt": "<paragraph>"} and nothing else."""

DEFAULT_SYSTEM_TAGS = """You rewrite a user's image request into a prompt for an anime-style image generation model that was trained on Danbooru-style tags.
Output a single line of comma-separated lowercase tags (spaces inside a tag are fine, e.g. "long hair", "looking at viewer").
Rules:
- Keep every tag or phrase the user already wrote, verbatim and in the same order, at the start. Then append additional tags.
- Translate non-English descriptions (e.g. Japanese) into the equivalent Danbooru tags.
- Cover: character count and gender (1girl, 2boys, ...), hair (color, length, style), eyes, expression, clothing pieces and colors, accessories, pose and gaze, background/setting, composition and framing (full body, upper body, from above, ...), lighting/time of day, and art style if the user implies one.
- Prefer real Danbooru tags. Be specific ("red plaid skirt" -> "plaid skirt, red skirt"). Aim for 15-30 tags total.
- Do NOT add quality tags (masterpiece, best quality, absurdres), artist names, or negative tags unless the user wrote them.
Respond with ONLY a JSON object of the form {"rewritten_prompt": "<tags>"} and nothing else."""


# ----- LLM ファイルの判定 ------------------------------------------------------
def sidecar(llm_path: Path, suffix: str) -> Path:
    """<名前>.system_prompt*.txt のパス（拡張子だけを差し替える）。"""
    return llm_path.with_name(llm_path.stem + suffix)


def llm_dir() -> Path:
    return config.models_root() / LLM_KIND


def list_llm_files() -> list[str]:
    """models/llm 内のモデルファイル名（サブフォルダ込み、ComfyUI と同じ表記）。"""
    return config.scan_models(LLM_KIND)


# ----- システムプロンプト --------------------------------------------------------
def system_prompt_for(llm_path: Path, style: str) -> str:
    suffix = SUFFIX_TAGS if style == STYLE_TAGS else SUFFIX_NATURAL
    p = sidecar(llm_path, suffix)
    if p.exists():
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass
    return DEFAULT_SYSTEM_TAGS if style == STYLE_TAGS else DEFAULT_SYSTEM_NATURAL


def chat_text(system: str, user: str, thinking: bool = False) -> str:
    """Qwen のチャット書式。先頭が <|im_start|> なら ComfyUI 側のトークナイザは
    既定テンプレートを当てずそのまま使う（qwen35.py tokenize_with_weights）。
    末尾を assistant で止めるのでモデルが続きを生成する。thinking=False なら
    空の <think></think> を先に置いて思考を省略させる（ComfyUI の既定
    テンプレートと同じやり方）。scom は常に思考なしで使う: PE では 2.5 倍
    速く忠実さも同等、汎用 Qwen3.5 4B では思考 ON だと上限トークンまで
    考え続けて答えが出ない（実測 320 秒で空）。"""
    text = (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    if not thinking:
        text += "<think>\n</think>\n"
    return text


# ----- ComfyUI グラフ ------------------------------------------------------------
def build_graph(llm_file: str, text: str, style: str,
                seed: int = LLM_SEED, max_length: int = MAX_LENGTH) -> dict:
    """CLIPLoader -> TextGenerate -> PreviewAny。

    TextGenerate の sampling_mode は DynamicCombo なので、API 形式では
    "sampling_mode": "on" と "sampling_mode.<param>" の平坦なキーで渡す。
    サンプリング値は PE の README 推奨（temperature 1.0 / top_p 0.95 /
    top_k 20、T2I は presence_penalty 1.5）。タグ整形は繰り返し抑制を弱める。
    """
    presence = 1.5 if style == STYLE_NATURAL else 0.5
    return {
        "1": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": llm_file, "type": "qwen_image"}},
        "2": {"class_type": "TextGenerate",
              "inputs": {
                  "clip": ["1", 0],
                  "prompt": text,
                  "max_length": int(max_length),
                  "sampling_mode": "on",
                  "sampling_mode.temperature": 1.0,
                  "sampling_mode.top_k": 20,
                  "sampling_mode.top_p": 0.95,
                  "sampling_mode.min_p": 0.0,
                  "sampling_mode.repetition_penalty": 1.0,
                  "sampling_mode.seed": int(seed),
                  "sampling_mode.presence_penalty": presence,
                  "thinking": True,
                  "use_default_template": False,
              }},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
    }


# ----- 出力の解釈 ------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_KEY_RE = re.compile(
    r'"(?:rewritten_prompt|prompt|positive_prompt|positive)"\s*:\s*'
    r'"((?:\\.|[^"\\])*)"', re.DOTALL)


def parse_output(raw: str) -> tuple[str, str]:
    """LLM の生出力から (プロンプト, 推奨アスペクト比 or "") を取り出す。"""
    text = raw or ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:            # 閉じタグだけ残った形
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:           # 思考が上限で途切れた: 答えは無い
        return "", ""
    text = text.strip()
    ratio = ""
    # 1) JSON オブジェクト（コードフェンス内でも可）
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start:end + 1]
        try:
            data = json.loads(blob)
        except ValueError:
            data = None
        if isinstance(data, dict):
            ratio = str(data.get("wh_ratio") or "")
            for k in ("rewritten_prompt", "prompt", "positive_prompt",
                      "positive"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip(), ratio
    # 2) JSON が壊れている（閉じ括弧が無い等）: キーだけ正規表現で拾う
    m = _KEY_RE.search(text)
    if m:
        try:
            return json.loads('"' + m.group(1) + '"').strip(), ratio
        except ValueError:
            return m.group(1).strip(), ratio
    # 3) 生テキスト（フェンス・引用符を剥がす）
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text, ratio


# ----- キャッシュ --------------------------------------------------------------------
class RewriteCache:
    """{key: rewritten} を userdata/prompt_llm_cache.json に保存。

    key は LLM ファイル・スタイル・システムプロンプト・元プロンプト・seed の
    ハッシュ。同じ入力なら LLM を回さない（連続生成・タスク積みで効く）。
    """
    MAX_ENTRIES = 500

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                self._data = {str(k): str(v) for k, v in d.items()}
        except (OSError, ValueError):
            pass

    @staticmethod
    def key(llm_file: str, style: str, system: str, user: str,
            seed: int = LLM_SEED, thinking: bool = False) -> str:
        h = hashlib.sha1()
        for part in (llm_file, style, system, user, str(seed),
                     "think" if thinking else "nothink"):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        while len(self._data) > self.MAX_ENTRIES:
            self._data.pop(next(iter(self._data)))   # 古いものから捨てる
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=0),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass


# ----- 実行 ---------------------------------------------------------------------------
def rewrite(backend, llm_file: str, user_prompt: str, style: str,
            cache_dir: Path,
            cancel: Optional[Callable[[], bool]] = None,
            thinking: bool = False) -> tuple[str, bool]:
    """(整形後プロンプト, キャッシュ命中か)。失敗は BackendError。"""
    from .comfy_backend import BackendError

    llm_path = llm_dir() / llm_file
    system = system_prompt_for(llm_path, style)
    user = user_prompt.strip()
    if not user:
        return "", True
    cache = RewriteCache(Path(cache_dir) / "prompt_llm_cache.json")
    key = RewriteCache.key(llm_file, style, system, user, thinking=thinking)
    hit = cache.get(key)
    if hit:
        return hit, True
    graph = build_graph(llm_file, chat_text(system, user, thinking), style)
    raw = backend.generate_text(graph, cancel=cancel)
    text, _ratio = parse_output(raw)
    if not text:
        raise BackendError(
            "LLM の出力からプロンプトを取り出せませんでした"
            f"（先頭 200 文字: {raw[:200]!r}）")
    cache.put(key, text)
    return text, False
