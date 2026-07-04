"""Prompt presets loaded from a user-editable CSV file (prompts.csv).

Columns: 1 = 設定名, 2 = プロンプト, 3 = ネガティブプロンプト.
Prompts routinely contain commas, so values with commas must be quoted — any
spreadsheet app does this automatically. The file is written with a UTF-8 BOM
so Excel on Japanese Windows opens it correctly.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Seed content for a freshly created file: one example row users can copy.
TEMPLATE = (
    'サンプル,"masterpiece, best quality","worst quality, low quality, blurry"\n'
)


def ensure_file(path: Path) -> None:
    """Create the CSV with an example row if it does not exist yet."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE, encoding="utf-8-sig")


def load(path: Path) -> list[tuple[str, str, str]]:
    """Return [(name, prompt, negative), ...]; missing file -> empty list.

    Rows with an empty first column are skipped, extra columns are ignored,
    and short rows are padded with empty strings.
    """
    if not path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            prompt = row[1].strip() if len(row) > 1 else ""
            negative = row[2].strip() if len(row) > 2 else ""
            out.append((name, prompt, negative))
    return out
