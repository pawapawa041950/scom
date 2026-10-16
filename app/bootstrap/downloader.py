"""Resumable HTTP downloads with progress and optional size/sha256 checks."""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

ProgressCb = Callable[[int, int], None]  # (downloaded_bytes, total_bytes)

_CHUNK = 1024 * 256
_UA = {"User-Agent": "scom-bootstrapper/0.1"}


class DownloadError(RuntimeError):
    pass


def _hf_token() -> Optional[str]:
    """Hugging Face のアクセストークン（ゲート付きリポジトリ用）。環境変数
    HF_TOKEN / HUGGING_FACE_HUB_TOKEN、無ければ huggingface-cli login が
    保存したファイル（HF_HOME/token、既定 ~/.cache/huggingface/token）。"""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = (os.environ.get(var) or "").strip()
        if tok:
            return tok
    home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    try:
        tok = Path(home, "token").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return tok or None


def _request(url: str, headers: dict, method: Optional[str] = None
             ) -> urllib.request.Request:
    req = urllib.request.Request(url, headers=headers, method=method)
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        tok = _hf_token()
        if tok:
            # unredirected: リダイレクト先（署名付き URL の CDN）には送らない。
            req.add_unredirected_header("Authorization", f"Bearer {tok}")
    return req


def _http_error_message(url: str, e: urllib.error.HTTPError) -> str:
    if e.code in (401, 403) and "huggingface.co" in url:
        repo = "/".join(urllib.parse.urlsplit(url).path.split("/")[1:3])
        return (f"ダウンロードが拒否されました（HTTP {e.code}）。"
                f"https://huggingface.co/{repo} は利用規約への同意が必要な"
                "リポジトリです。ブラウザで同意したうえで、環境変数 HF_TOKEN に"
                "アクセストークンを設定する（または huggingface-cli login）か、"
                "手動でダウンロードして models フォルダに置いてください。")
    return f"failed to download {url}: {e}"


def _content_length(url: str) -> Optional[int]:
    req = _request(url, _UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def download(url: str, dest: Path,
             on_progress: Optional[ProgressCb] = None,
             expected_size: Optional[int] = None,
             sha256: Optional[str] = None,
             resume: bool = True,
             cancel: Optional[Callable[[], bool]] = None) -> Path:
    """Download ``url`` to ``dest`` (atomically via a .part file).

    Supports HTTP range resume. Verifies size/sha256 when provided. ``cancel``
    is polled between chunks and aborts with DownloadError if it returns True.
    """
    on_progress = on_progress or (lambda _d, _t: None)
    cancel = cancel or (lambda: False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    total = expected_size or _content_length(url) or 0
    have = part.stat().st_size if (resume and part.exists()) else 0
    if have and total and have >= total:
        have = 0  # stale/oversized partial; restart

    headers = dict(_UA)
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"

    req = _request(url, headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if have and resp.status != 206:
                # Server ignored the range; start over.
                have = 0
                mode = "wb"
            if not total:
                cl = resp.headers.get("Content-Length")
                if cl:
                    total = int(cl) + have
            downloaded = have
            with open(part, mode) as f:
                on_progress(downloaded, total)
                while True:
                    if cancel():
                        raise DownloadError("download cancelled")
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    on_progress(downloaded, total)
    except urllib.error.HTTPError as e:
        raise DownloadError(_http_error_message(url, e)) from e
    except urllib.error.URLError as e:
        raise DownloadError(f"failed to download {url}: {e}") from e

    if expected_size and part.stat().st_size != expected_size:
        actual = part.stat().st_size
        raise DownloadError(
            f"size mismatch for {dest.name}: got {actual}, expected {expected_size}"
        )
    if sha256:
        digest = _sha256(part)
        if digest.lower() != sha256.lower():
            raise DownloadError(f"sha256 mismatch for {dest.name}")

    os.replace(part, dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
