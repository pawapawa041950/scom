"""LoRA metadata: SHA256 hashing, civitai lookup, and a local cache.

LoRA ファイルの SHA256 を計算し、civitai の by-hash API からメタ情報
（モデル名・ベースモデル・トリガーワード・プレビュー画像）を取得する。
結果は userdata/lora_cache/ にキャッシュし、2回目以降・オフライン時は
ネットワークなしで表示できる:

  lora_cache/index.json      {relname: {size, mtime, sha256, meta}}
  lora_cache/thumbs/<sha>.jpg  縮小プレビュー画像

meta の中身（civitai レスポンスの抜粋）:
  {"found": bool, "name": str, "version": str, "base_model": str,
   "trained_words": [str], "url": str, "thumb": str(サムネのファイル名) }
civitai に登録がないファイルは {"found": False} を記憶して再問い合わせしない。
ネットワークエラー時は何もキャッシュしない（次回また試す）。
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/"
_USER_AGENT = "scom/1.0 (+https://github.com/)"
_TIMEOUT = 15  # seconds
_THUMB_WIDTH = 256

# civitai の baseModel 文字列 → 本アプリの系統。qwen 系 LoRA は anima/krea2
# のどちらでも使える可能性があるので両方にマッチさせる。
_FAMILY_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("sdxl", frozenset({"sdxl"})),
    ("pony", frozenset({"sdxl"})),
    ("illustrious", frozenset({"sdxl"})),
    ("noobai", frozenset({"sdxl"})),
    ("qwen", frozenset({"anima", "krea2"})),
    ("krea", frozenset({"krea2"})),
    ("flux", frozenset({"krea2"})),
)


def families_from_base_model(base_model: str) -> frozenset[str]:
    """App families a civitai baseModel string maps to (empty = unknown)."""
    b = (base_model or "").lower()
    for key, fams in _FAMILY_RULES:
        if key in b:
            return fams
    return frozenset()


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest().lower()


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def pick_preview_url(images: list[dict]) -> str:
    """Least-NSFW image url, downsized via civitai's width path segment."""
    best = ""
    best_level = 10 ** 9
    for img in images or []:
        if img.get("type") not in (None, "image"):
            continue  # 動画プレビューは対象外
        url = str(img.get("url", ""))
        if not url:
            continue
        level = int(img.get("nsfwLevel", 0) or 0)
        if level < best_level:
            best, best_level = url, level
    # civitai の画像 URL は変換指定をパスに持つ（/width=450/ や
    # /original=true/）。サムネ用に width=256 へ差し替える。
    return re.sub(r"/(?:width=\d+|original=true)/",
                  f"/width={_THUMB_WIDTH}/", best)


def fetch_civitai_meta(sha256: str) -> Optional[dict]:
    """Query civitai by hash. Returns the meta dict, or None when the hash is
    unknown to civitai (HTTP 404). Network trouble raises OSError."""
    try:
        raw = _http_get(CIVITAI_BY_HASH + sha256)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    data = json.loads(raw)
    model = data.get("model") or {}
    model_id = data.get("modelId")
    return {
        "found": True,
        "name": str(model.get("name") or ""),
        "version": str(data.get("name") or ""),
        "base_model": str(data.get("baseModel") or ""),
        "trained_words": [str(w) for w in data.get("trainedWords") or []],
        "url": (f"https://civitai.com/models/{model_id}" if model_id else ""),
        "preview_url": pick_preview_url(data.get("images") or []),
    }


class LoraCache:
    """File-backed cache of per-LoRA hash/meta/thumbnail.

    すべてのメソッドは1本のワーカースレッドから呼ぶ前提（排他なし）。
    """

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.thumbs = self.dir / "thumbs"
        self._index: dict[str, dict] = {}
        try:
            with open(self.dir / "index.json", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._index = data
        except (OSError, ValueError):
            self._index = {}

    def _save_index(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(self._index, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.dir / "index.json")

    def lookup(self, relname: str, path: Path) -> Optional[dict]:
        """Cached entry {sha256, meta} if it matches the file's size+mtime."""
        e = self._index.get(relname)
        if not e:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        if e.get("size") != st.st_size or e.get("mtime") != int(st.st_mtime):
            return None
        return e

    def store(self, relname: str, path: Path, sha256: str,
              meta: dict) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        self._index[relname] = {"size": st.st_size,
                                "mtime": int(st.st_mtime),
                                "sha256": sha256, "meta": meta}
        self._save_index()

    def thumb_file(self, sha256: str) -> Path:
        return self.thumbs / f"{sha256}.jpg"

    def ensure_thumb(self, sha256: str, preview_url: str) -> str:
        """Download the preview if not cached yet. Returns the local path
        ("" when there is no preview). Network trouble raises OSError."""
        if not preview_url:
            return ""
        dest = self.thumb_file(sha256)
        if dest.exists():
            return str(dest)
        data = _http_get(preview_url)
        self.thumbs.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return str(dest)
