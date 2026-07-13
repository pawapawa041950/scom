"""First-run setup dialog.

Two phases:
  1. Selection — show the detected GPU/torch plan and let the user choose which
     models to download (required ones pre-checked).
  2. Running — a per-component progress row for each step (uv, venv, torch,
     ComfyUI, deps) and one per selected model, each with its own bar.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ..bootstrap.setup import FirstRunSetup, StepUpdate, FIXED_STEPS
from . import ansi_log
from .model_selector import ModelSelector
from .widgets import ProgressRow
from .window_state import bind_geometry


class _SetupWorker(QObject):
    step = Signal(object)   # StepUpdate
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, setup: FirstRunSetup, selected: list[str]):
        super().__init__()
        self.setup = setup
        self.selected = selected
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.setup.run(
                on_step=self.step.emit,
                on_log=self.log.emit,
                cancel=lambda: self._cancel,
                selected_models=self.selected,
            )
            self.done.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class SetupDialog(QDialog):
    def __init__(self, setup: FirstRunSetup, parent=None):
        super().__init__(parent)
        self.setWindowTitle("scom - 初回セットアップ")
        self.setModal(True)
        self.resize(720, 560)
        bind_geometry(self, "setup")
        self._setup = setup
        self._paths = setup.paths
        self._rows: dict[str, ProgressRow] = {}

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_selection_page())
        self.stack.addWidget(self._build_running_page())

        root = QVBoxLayout(self)
        root.addWidget(self.stack)

    # ----- phase 1: selection ---------------------------------------------
    def _build_selection_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        intro = QLabel(
            "初回セットアップでは、このマシン向けに生成バックエンドを準備します。\n"
            "PyTorch・ComfyUI・選択したモデルをここでダウンロードします（初回のみ）。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        plan = self._setup.torch_plan()
        plan_lbl = QLabel(f"検出した環境:  {plan.reason}")
        plan_lbl.setStyleSheet("color:#39c;")
        plan_lbl.setWordWrap(True)
        layout.addWidget(plan_lbl)

        hint = QLabel(
            "生成には base・VAE・text encoder が必須です。preview 系は任意です。\n"
            "自分でダウンロードしたものを models/ 配下に配置済みなら、ここで"
            "選ばなくても利用できます。"
        )
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Shared with the model-manager dialog so the two screens stay in sync.
        self._selector = ModelSelector(self._paths, with_progress=False)
        layout.addWidget(self._selector, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel = QPushButton("キャンセル")
        cancel.clicked.connect(self.reject)
        start = QPushButton("セットアップ開始")
        start.setDefault(True)
        start.clicked.connect(self._on_start)
        btn_row.addWidget(cancel)
        btn_row.addWidget(start)
        layout.addLayout(btn_row)
        return page

    # ----- phase 2: running -----------------------------------------------
    def _build_running_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.label = QLabel("開始中…")
        layout.addWidget(self.label)

        self.rows_box = QVBoxLayout()
        rows_holder = QWidget()
        rows_holder.setLayout(self.rows_box)
        layout.addWidget(rows_holder)

        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall.setFormat("全体  %p%")
        layout.addWidget(self.overall)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        ansi_log.style_log(self.log)
        layout.addWidget(self.log, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_close = QPushButton("閉じる")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)
        return page

    def _row(self, step_id: str, title: str) -> ProgressRow:
        row = self._rows.get(step_id)
        if row is None:
            row = ProgressRow(title)
            row.set_pending()
            self._rows[step_id] = row
            self.rows_box.addWidget(row)
        return row

    # ----- start / events --------------------------------------------------
    def _on_start(self) -> None:
        self._selected = self._selector.selected_filenames()
        # Pre-create rows so the user sees the full plan up front.
        for sid, title in FIXED_STEPS:
            self._row(sid, title)
        for fn in self._selected:
            self._row("model:" + fn, fn)
        self._total_steps = len(self._rows)
        self._done_steps = 0

        self.stack.setCurrentIndex(1)
        self._thread = QThread(self)
        self._worker = _SetupWorker(self._setup, self._selected)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.step.connect(self._on_step)
        self._worker.log.connect(self._on_log)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_step(self, u: StepUpdate) -> None:
        row = self._row(u.step_id, u.title)
        if u.title:
            row.set_title(u.title)
        if u.status == "running":
            row.set_running(u.fraction, u.detail)
            self.label.setText(f"{u.title} — {u.detail}")
        elif u.status == "done":
            row.set_done()
            self._mark_step_finished()
        elif u.status == "skipped":
            row.set_skipped()
            self._mark_step_finished()
        elif u.status == "error":
            row.set_error(u.detail)

    def _mark_step_finished(self) -> None:
        self._done_steps += 1
        if self._total_steps:
            self.overall.setValue(int(1000 * self._done_steps / self._total_steps))

    def _on_log(self, text: str) -> None:
        ansi_log.append_ansi(self.log, text)

    def _on_done(self) -> None:
        self.overall.setValue(1000)
        self.label.setText("セットアップ完了")
        self.accept()

    def _on_failed(self, msg: str) -> None:
        self.label.setText("セットアップに失敗しました")
        ansi_log.append_ansi(self.log, "エラー: " + msg)
        ansi_log.append_ansi(
            self.log,
            "閉じて再起動すると再試行できます（完了済みのステップはスキップされます）。",
        )
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _on_cancel(self) -> None:
        if getattr(self, "_worker", None):
            self._worker.cancel()
        self.label.setText("キャンセル中…")
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.isRunning():
            self._worker.cancel()
            thread.quit()
            thread.wait(3000)
        super().closeEvent(event)
