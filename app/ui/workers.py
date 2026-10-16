"""バックグラウンド処理（QThread で動かすワーカー）。main_window.py から分割。"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage
from pathlib import Path
from typing import Optional
from .. import config, metadata, prompt_llm, xyz
from ..comfy_backend import BackendError, ComfyBackend, Progress
from ..workflow import GenParams, build_graph


class _StartWorker(QObject):
    """Starts the ComfyUI backend off the UI thread."""
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend):
        super().__init__()
        self.backend = backend

    def run(self) -> None:
        try:
            self.backend.start(log=self.log.emit)
            self.done.emit()
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _GenWorker(QObject):
    progress = Signal(Progress)
    preview = Signal(bytes)
    cached = Signal(list)  # node ids served from the backend output cache
    timing = Signal(float)  # 純粋な推論(サンプリング)時間 [秒]
    done = Signal(list)    # list[bytes]
    failed = Signal(str)
    status = Signal(str)   # 前処理（プロンプト整形）の状況
    # LLM 整形: (元プロンプト, 整形後, キャッシュ命中か)
    rewritten = Signal(str, str, bool)
    # 整形後に組み直したグラフ（メタデータ保存用に本体へ戻す）
    prepared = Signal(object, dict)

    def __init__(self, backend: ComfyBackend, graph: dict,
                 params: Optional[GenParams] = None,
                 user_data: Optional[Path] = None):
        super().__init__()
        self.backend = backend
        self.graph = graph
        self.params = params      # None = graph をそのまま実行（マージ等）
        self.user_data = user_data
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _prepare(self) -> dict:
        """LLM 整形が指定されていればプロンプトを書き換えてグラフを組み直す。"""
        p = self.params
        if p is None or not p.llm_file:
            return self.graph
        self.status.emit(f"プロンプトを LLM で整形中… ({p.llm_file})")
        text, cached = prompt_llm.rewrite(
            self.backend, p.llm_file, p.prompt, p.llm_style,
            self.user_data or config.AppPaths().user_data,
            cancel=lambda: self._cancel)
        if text:
            p.prompt_original = p.prompt
            p.prompt = text
        self.rewritten.emit(p.prompt_original, p.prompt, cached)
        graph = build_graph(p)
        self.prepared.emit(p, graph)
        return graph

    def run(self) -> None:
        try:
            graph = self._prepare()
            images = self.backend.generate(
                graph,
                on_progress=self.progress.emit,
                on_preview=self.preview.emit,
                cancel=lambda: self._cancel,
                on_cached=self.cached.emit,
                on_timing=self.timing.emit,
            )
            self.done.emit(images)
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _XyzWorker(QObject):
    """Runs the XYZ plot cells sequentially off the UI thread.

    セル単位のエラーは placeholder (None) にして続行するが、最初のセルで
    失敗した場合は設定不備とみなして全体を中止する。
    """
    progress = Signal(Progress)
    preview = Signal(bytes)
    timing = Signal(float)          # セルごとの純粋な推論時間 [秒]
    cell_done = Signal(int, int)    # finished count, total
    cell_image = Signal(int, bytes)  # grid index, png (直後の逐次保存用)
    cell_error = Signal(int, str)   # grid index, message
    done = Signal(object)           # {grid_index: png bytes | None}
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend, jobs: list):
        super().__init__()
        self.backend = backend
        self.jobs = jobs  # list[(grid_index, graph)]
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        results: dict[int, Optional[bytes]] = {}
        try:
            for n, (idx, graph) in enumerate(self.jobs):
                if self._cancel:
                    raise BackendError("生成をキャンセルしました")
                try:
                    images = self.backend.generate(
                        graph,
                        on_progress=self.progress.emit,
                        on_preview=self.preview.emit,
                        cancel=lambda: self._cancel,
                        on_timing=self.timing.emit,
                    )
                except BackendError as e:
                    if self._cancel or n == 0:
                        raise
                    results[idx] = None
                    self.cell_error.emit(idx, str(e))
                else:
                    results[idx] = images[0] if images else None
                    if results[idx]:
                        self.cell_image.emit(idx, results[idx])
                self.cell_done.emit(n + 1, len(self.jobs))
            self.done.emit(results)
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _XyzComposeWorker(QObject):
    """XYZ の比較グリッド画像を UI スレッド外で合成・保存する。

    セルの PNG デコード → グリッド描画（QPainter on QImage は非GUIスレッド
    可）→ PNG エンコード → （指定があれば）メタデータ付き保存、のすべてが
    大きなグリッドでは数秒かかり、UI スレッドで行うとフリーズして見える。
    """
    done = Signal(object)   # {"png": bytes, "w": int, "h": int, "log": [str]}

    def __init__(self, cells: list, nx: int, ny: int, nz: int,
                 labels: list, legend: bool, margin: int,
                 save_path, params_text: str, extra: dict, embed: bool):
        super().__init__()
        self._cells = cells          # list[bytes | None] (grid order)
        self._nx, self._ny, self._nz = nx, ny, nz
        self._labels = labels
        self._legend = legend
        self._margin = margin
        self._save_path = save_path  # Path | None (None = 保存しない)
        self._params_text = params_text
        self._extra = extra
        self._embed = embed

    def run(self) -> None:
        log: list[str] = []
        images = [QImage.fromData(b) if b else None for b in self._cells]
        grid = xyz.compose_grid(
            images, self._nx, self._ny, self._nz,
            self._labels[0], self._labels[1], self._labels[2],
            draw_legend=self._legend, margin=self._margin)
        png = xyz.qimage_png_bytes(grid)
        if self._save_path is not None:
            try:
                metadata.save_with_metadata(
                    png, self._save_path, "png", 6, self._params_text,
                    extra=self._extra, embed=self._embed)
                note = "メタデータ付き" if self._embed else "メタデータなし"
                log.append(f"{self._save_path} に保存しました（{note}）")
            except Exception as e:  # noqa: BLE001
                log.append(f"グリッド画像の保存に失敗: {e}")
        self.done.emit({"png": png, "w": grid.width(), "h": grid.height(),
                        "log": log})
