"""Model manager dialog: pick which models to download, with per-file progress.
Reachable from the main window so downloads aren't limited to the first-run
flow. The model list + quick-select buttons come from the shared ModelSelector
(also used by the first-run setup screen).
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from .. import config
from ..bootstrap import models as models_mod
from .model_selector import ModelSelector
from .window_state import bind_geometry


def _fmt_progress(d: int, t: int) -> str:
    def g(n: int) -> str:
        return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"
    return f"{g(d)} / {g(t)}" if t else g(d)


class _DownloadWorker(QObject):
    progress = Signal(str, float, str)   # filename, fraction, detail
    file_done = Signal(str)
    file_failed = Signal(str, str)
    finished = Signal()

    def __init__(self, paths: config.AppPaths, items: list):
        super().__init__()
        self.paths = paths
        self.items = items
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        for m in self.items:
            if self._cancel:
                break
            try:
                models_mod.download_model(
                    self.paths, m,
                    on_progress=lambda d, t, _fn=m.filename: self.progress.emit(
                        _fn, (d / t if t else 0.0), _fmt_progress(d, t)),
                    cancel=lambda: self._cancel,
                )
                self.file_done.emit(m.filename)
            except Exception as e:  # noqa: BLE001
                self.file_failed.emit(m.filename, str(e))
        self.finished.emit()


class ModelsDialog(QDialog):
    def __init__(self, paths: config.AppPaths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("scom - モデル管理")
        self.resize(720, 560)
        bind_geometry(self, "models")
        self._paths = paths
        self._thread: QThread | None = None

        layout = QVBoxLayout(self)
        info = QLabel(
            "models/ にダウンロードするモデルを選択してください。"
            "既にあるファイルは印が付き、スキップされます。\n"
            "ここからダウンロードする以外に、自分でダウンロードしたものを"
            "models/ 配下の各フォルダ（diffusion_models / vae / text_encoders）に"
            "配置することでも利用可能です。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._selector = ModelSelector(paths, with_progress=True)
        layout.addWidget(self._selector, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_download = QPushButton("選択をダウンロード")
        self.btn_download.clicked.connect(self._on_download)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_close = QPushButton("閉じる")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_download)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    # ----- download --------------------------------------------------------
    def _on_download(self) -> None:
        sel = set(self._selector.selected_filenames())
        items = [m for m in self._selector.manifest if m.filename in sel]
        if not items:
            return
        # 選んだファイルの進捗行だけを表示してから開始する。
        self._selector.reveal_rows([m.filename for m in items])
        for m in items:
            self._selector.rows[m.filename].set_running(0.0, "待機中")
        self.btn_download.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)

        self._thread = QThread(self)
        self._worker = _DownloadWorker(self._paths, items)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, filename: str, fraction: float, detail: str) -> None:
        self._selector.rows[filename].set_running(fraction, detail)

    def _on_file_done(self, filename: str) -> None:
        self._selector.rows[filename].set_done("完了")
        cb = self._selector.checks[filename]
        cb.setChecked(False)
        cb.setEnabled(False)

    def _on_file_failed(self, filename: str, msg: str) -> None:
        self._selector.rows[filename].set_error("失敗")

    def _on_finished(self) -> None:
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _on_cancel(self) -> None:
        if getattr(self, "_worker", None):
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread is not None and self._thread.isRunning():
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
