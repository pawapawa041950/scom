"""Persisted UI settings (TOML).

All settings are auto-saved whenever the user changes a control, EXCEPT
``prompt`` and ``negative``: those are read from the file as *initial* values
only and are never written back from the UI (the file value stays as the user's
chosen default). The file lives next to the executable so it is easy to edit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore

from .workflow import DEFAULT_NEGATIVE

# Loaded as initial values, never persisted from the UI.
INITIAL_ONLY = ("prompt", "negative")

DEFAULTS: dict[str, Any] = {
    # models
    "preset": "all",   # all | anima | krea2 (filters the model dropdowns)
    "diffusion": "",
    "vae": "",
    "te1": "",
    "te2": "",
    "dual_te": False,
    "clip_type": "stable_diffusion",
    # Model merges: JSON list of entries, each
    #   {"id": int, "name": str, "models": [["file", weight], ...],
    #    "quant": ""|"fp8"|"int8_convrot", "low_memory": bool}
    # merge_seq is the last id handed out (numbering never reuses ids).
    "merges": "[]",
    "merge_seq": 0,
    # prompts (initial-only)
    "prompt": "",
    "negative": DEFAULT_NEGATIVE,
    # generation settings
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg": 4.0,
    "batch": 1,
    "sampler": "er_sde",
    "scheduler": "simple",
    "seed": "-1",
    "randomize": True,
    "dtype": "default",
    # image output
    "image_format": "png",   # png | jpg | webp
    "png_compress": 6,       # 0..9
    "jpg_quality": 92,       # 1..100
    "webp_quality": 90,      # 1..100
}


# Written as TOML multi-line literal strings ('''…''') so users can freely edit
# them by hand — including double quotes and newlines — without escaping.
MULTILINE_KEYS = ("prompt", "negative")


def load(path: Path) -> tuple[dict[str, Any], str | None]:
    """Return (settings, error).

    ``settings`` is DEFAULTS overlaid with the file's values. ``error`` is a
    human-readable message if the file exists but could not be parsed (in which
    case defaults are returned) — callers should surface it rather than let a
    broken file silently reset everything.
    """
    data = dict(DEFAULTS)
    if not path.exists():
        return data, None
    try:
        with open(path, "rb") as f:
            data.update(_toml.load(f))
        return data, None
    except (OSError, ValueError) as e:
        return data, str(e)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    s = (str(value)
         .replace("\\", "\\\\").replace('"', '\\"')
         .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{s}"'


def _fmt_multiline(value: Any) -> str:
    """Format a string as a TOML literal multi-line string for easy editing.

    Falls back to an escaped basic string for content that can't be expressed
    literally (contains ''' or starts with a newline, which TOML would trim).
    """
    s = str(value)
    if "'''" in s or s.startswith("\n"):
        return _fmt(s)
    return f"'''{s}'''"


def _existing_multiline(path: Path) -> dict[str, Any]:
    """Read prompt/negative currently on disk so save() never changes them."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            disk = _toml.load(f)
    except (OSError, ValueError):
        return {}
    return {k: disk[k] for k in MULTILINE_KEYS if k in disk}


def save(path: Path, data: dict[str, Any]) -> None:
    """Write all known keys (stable order) as a flat TOML table.

    prompt/negative are NEVER changed by the app: whatever is already in the
    file is kept verbatim. Only when the file has no such key yet (first run)
    is the value from ``data`` used to seed it.
    """
    keep = _existing_multiline(path)
    lines = [
        "# scom 設定ファイル（変更すると自動保存されます）。",
        "# prompt / negative は「初期値」専用です。アプリが書き換えることはありません。",
        "# 初期プロンプトを変えたいときは、下の ''' と ''' の間を自由に編集してください",
        "# （ダブルクォートや改行もそのまま書けます）。",
        "",
    ]
    for key in DEFAULTS:
        if key in MULTILINE_KEYS:
            # Preserve the on-disk value; fall back to data only to seed a new file.
            value = keep[key] if key in keep else data.get(key, DEFAULTS[key])
            lines.append(f"{key} = {_fmt_multiline(value)}")
        elif key in data:
            lines.append(f"{key} = {_fmt(data[key])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
