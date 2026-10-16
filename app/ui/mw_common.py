"""MainWindow とその分割モジュール（mw_*.py）で共有する定数。"""
from __future__ import annotations

import re
from PySide6.QtGui import QColor, QTextFormat

MAX_SEED = 2**63 - 1
# LoRA トリガーワードをプロンプト欄に挿入したときの区別用マーキング。
# 挿入した文字範囲に専用の文字書式（背景色 + token プロパティ）を付け、
# 見た目（背景ハイライト）とロジック（token による区間追跡）の両方で
# 元のプロンプトと区別する。token を持つ区間はユーザーが編集しても書式が
# 残るので、編集後のワードごとまとめて削除できる。
_LORA_TOKEN_PROP = QTextFormat.UserProperty + 17
# プロンプトに貼られた WebUI 形式の LoRA 指定 <lora:名前:強度>（強度は省略可、
# 名前は拡張子なし・サブフォルダ有無を問わない）。貼られた時点で LoRA 設定に
# 取り込み、文字列は欄から取り除く（_absorb_lora_tags）。
_LORA_TAG_RE = re.compile(
    r"<lora:([^:<>]+?)\s*(?::\s*([-+]?\d*\.?\d+))?\s*>", re.IGNORECASE)
# 挿入部分の色（LoRA ウィンドウのトリガーワード表示に合わせた明るい配色）。
# 暗色テーマだと本文の文字色が明るいので、明背景に合わせて文字色も濃色に。
_LORA_INSERT_BG = QColor("#e7edf5")   # 明るい背景
_LORA_INSERT_FG = QColor("#22456e")   # 濃い文字色
# XYZ 軸 id -> その軸が支配する GenParams フィールド。モデル軸のプリセット
# 適用時、軸で振っている値をプリセット設定で上書きしないための対応表。
_XYZ_AXIS_FIELDS = {
    "seed": {"seed"}, "steps": {"steps"}, "cfg": {"cfg"},
    "sampler": {"sampler"}, "scheduler": {"scheduler"},
    "size": {"width", "height"}, "dtype": {"weight_dtype"},
}
# Image format option that disables saving generated images to disk.
NO_SAVE = "保存しない"
# Merge entries in the diffusion combo: shown at the top of the list with a
# colored prefix; the item data is "__merge__:<id>".
MERGE_PREFIX = "マージモデル："
MERGE_TOKEN = "__merge__"
# 適用中 LoRA の name に使う、キャッシュ上の差分 LoRA の識別子
# （"__cachedlora__:<id>"）。生成時に workflow.cached_lora_spec に展開する。
CACHED_LORA_TOKEN = "__cachedlora__"
MERGE_COLOR = "#1a7f37"  # green tint for merge entries
