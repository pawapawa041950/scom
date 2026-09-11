"""Settings dialog: model downloads (per-file progress) + backend options
(SageAttention). Reachable from the main window so downloads aren't limited to
the first-run flow. The model list + quick-select buttons come from the shared
ModelSelector (also used by the first-run setup screen).
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout,
)

from .. import config
from ..bootstrap import environment, models as models_mod
from ..bootstrap.setup import (
    SetupError, ck_attention_available, install_sage_attention, sage_installed,
)
from .model_selector import ModelSelector
from .window_state import bind_geometry


# SageAttention インストールジョブ（親なしスレッド）の生存参照。ダイアログを
# 閉じてもインストールは完走する。
_SAGE_JOBS: list = []


class _DownloadWorker(QObject):
    # filename, downloaded bytes, total bytes（float: 2GB 超でも安全に運ぶ）
    progress = Signal(str, float, float)
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
                    on_progress=lambda d, t, _fn=m.filename:
                        self.progress.emit(_fn, float(d), float(t)),
                    cancel=lambda: self._cancel,
                )
                self.file_done.emit(m.filename)
            except Exception as e:  # noqa: BLE001
                self.file_failed.emit(m.filename, str(e))
        self.finished.emit()


class _SageInstallWorker(QObject):
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, paths: config.AppPaths):
        super().__init__()
        self.paths = paths

    def run(self) -> None:
        try:
            install_sage_attention(self.paths, self.log.emit)
            self.done.emit()
        except SetupError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _CkCheckWorker(QObject):
    """Comfy Kitchen INT8 attention の対応判定（バックエンド python を起動して
    comfy_kitchen を import するため数秒かかる）。"""
    result = Signal(bool, str)

    def __init__(self, paths: config.AppPaths):
        super().__init__()
        self.paths = paths

    def run(self) -> None:
        ok, reason = ck_attention_available(self.paths)
        self.result.emit(ok, reason)


class ModelsDialog(QDialog):
    # SageAttention 設定の変更（persist はメインウィンドウが行う）。
    sage_toggled = Signal(bool)
    ck_toggled = Signal(bool)

    def __init__(self, paths: config.AppPaths, sage_enabled: bool = False,
                 ck_enabled: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("scom - 設定")
        self.resize(720, 560)
        bind_geometry(self, "models")
        self._paths = paths
        self._thread: QThread | None = None
        self._sage_thread: QThread | None = None

        layout = QVBoxLayout(self)
        info = QLabel(
            "ダウンロードするモデル一式を選択してください。"
            "既にあるファイルはスキップされます。\n"
            "ここからダウンロードする以外に、自分でダウンロードしたものを"
            "models/ 配下の各フォルダ（diffusion_models / vae / text_encoders）に"
            "配置することでも利用可能です。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._selector = ModelSelector(paths, with_progress=True)
        self._selector.btn_download.clicked.connect(self._on_download)
        layout.addWidget(self._selector, stretch=1)

        # ----- backend options ----------------------------------------------
        opt_box = QGroupBox("バックエンド")
        opt_lay = QVBoxLayout(opt_box)
        self.chk_sage = QCheckBox(
            "SageAttention を使用（生成を高速化・出力が僅かに変化）")
        self.chk_sage.setToolTip(
            "量子化 attention による推論高速化。ON にすると未導入の場合は"
            "バックエンド環境へ自動インストールします。\n"
            "出力は標準 attention とわずかに変わります（同一 seed でも"
            "ビット一致しません）。反映にはアプリの再起動が必要です。")
        self.lbl_sage = QLabel("")
        self.lbl_sage.setWordWrap(True)
        self.lbl_sage.setStyleSheet("color:#888;")
        opt_lay.addWidget(self.chk_sage)
        opt_lay.addWidget(self.lbl_sage)
        # Comfy Kitchen INT8 attention（ComfyUI v0.35+）。SageAttention と排他。
        self.chk_ck = QCheckBox(
            "Comfy Kitchen INT8 attention を使用（生成を高速化・出力が僅かに変化）")
        self.chk_ck.setToolTip(
            "ComfyUI 同梱の comfy-kitchen による INT8 attention。追加インストール"
            "不要で、SageAttention の代わりに使えます（同時には使えません）。\n"
            "出力は標準 attention とわずかに変わります。反映にはアプリの再起動が"
            "必要です。")
        self.lbl_ck = QLabel("")
        self.lbl_ck.setWordWrap(True)
        self.lbl_ck.setStyleSheet("color:#888;")
        opt_lay.addWidget(self.chk_ck)
        opt_lay.addWidget(self.lbl_ck)
        layout.addWidget(opt_box)
        self._init_sage_checkbox(sage_enabled)
        self._init_ck_checkbox(ck_enabled)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QPushButton("閉じる")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    # ----- SageAttention ----------------------------------------------------
    def _init_sage_checkbox(self, enabled: bool) -> None:
        """環境の対応可否を判定して初期状態を決める（対応外なら無効化）。"""
        gpu = environment.detect_gpu()
        from ..bootstrap.setup import _Manifest
        torch_tag = str(_Manifest(self._paths.manifest_path)
                        .get("torch_tag") or "")
        ok, reason = environment.sage_supported(gpu, torch_tag)
        if not ok:
            self.chk_sage.setChecked(False)
            self.chk_sage.setEnabled(False)
            self.lbl_sage.setText(reason)
            self.lbl_sage.setStyleSheet("color:#c33;")
            return
        self.chk_sage.setChecked(bool(enabled))
        if sage_installed(self._paths):
            self.lbl_sage.setText("インストール済み（切替はアプリ再起動後に反映）")
        else:
            self.lbl_sage.setText(
                "未インストール（ON にすると自動でインストールします）")
        # 初期化後に接続 = 復元では発火しない。
        self.chk_sage.toggled.connect(self._on_sage_toggled)

    def _on_sage_toggled(self, checked: bool) -> None:
        # 設定は先に反映する（未インストールのまま ON でも、起動側が
        # パッケージ実在を確認してからフラグを付けるので安全）。
        self.sage_toggled.emit(bool(checked))
        if not checked:
            self.lbl_sage.setText("無効にしました（アプリ再起動後に反映）")
            return
        if self.chk_ck.isChecked():
            self.chk_ck.setChecked(False)   # 排他（ck 側の toggled で設定も OFF）
        if sage_installed(self._paths):
            self.lbl_sage.setText("有効にしました（アプリ再起動後に反映）")
            return
        # 未導入 → バックエンド venv へバックグラウンドでインストール。
        # スレッドは親なし + モジュール参照で保持し、途中でウィンドウを
        # 閉じても完走させる（接続先は bound method なので破棄時に自動切断）。
        self.chk_sage.setEnabled(False)
        self.lbl_sage.setText("SageAttention をインストール中…")
        thread = QThread()
        worker = _SageInstallWorker(self._paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._on_sage_log)
        worker.done.connect(self._on_sage_installed)
        worker.failed.connect(self._on_sage_install_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        _SAGE_JOBS.append((thread, worker))
        thread.finished.connect(
            lambda t=thread, w=worker: _SAGE_JOBS.remove((t, w)))
        self._sage_thread = thread
        thread.start()

    def _on_sage_log(self, m: str) -> None:
        self.lbl_sage.setText(m[-120:])

    def _on_sage_installed(self) -> None:
        self.chk_sage.setEnabled(True)
        self.lbl_sage.setText(
            "インストール完了。アプリを再起動すると有効になります。")

    def _on_sage_install_failed(self, msg: str) -> None:
        self.chk_sage.setEnabled(True)
        self.chk_sage.blockSignals(True)
        self.chk_sage.setChecked(False)
        self.chk_sage.blockSignals(False)
        self.sage_toggled.emit(False)   # 楽観反映を取り消す
        self.lbl_sage.setText("インストールに失敗しました")
        QMessageBox.warning(self, "SageAttention", f"インストールに失敗しました:\n{msg}")

    # ----- Comfy Kitchen INT8 attention ------------------------------------
    def _init_ck_checkbox(self, enabled: bool) -> None:
        self.chk_ck.setChecked(bool(enabled))
        self.lbl_ck.setText(
            "有効（アプリ再起動後に反映）" if enabled
            else "ON にすると対応可否を確認してから有効にします")
        # 初期化後に接続 = 復元では発火しない。
        self.chk_ck.toggled.connect(self._on_ck_toggled)

    def _on_ck_toggled(self, checked: bool) -> None:
        if not checked:
            self.ck_toggled.emit(False)
            self.lbl_ck.setText("無効にしました（アプリ再起動後に反映）")
            return
        if self.chk_sage.isChecked():
            self.chk_sage.setChecked(False)  # 排他（sage 側の toggled で設定も OFF）
        # 未対応環境で ON にすると ComfyUI が起動時に落ちるため、設定へ反映する
        # 前にバックエンド python で判定する（親なしスレッド、_SAGE_JOBS 流用）。
        self.chk_ck.setEnabled(False)
        self.lbl_ck.setText("対応を確認中…")
        thread = QThread()
        worker = _CkCheckWorker(self._paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._on_ck_checked)
        worker.result.connect(thread.quit)
        _SAGE_JOBS.append((thread, worker))
        thread.finished.connect(
            lambda t=thread, w=worker: _SAGE_JOBS.remove((t, w)))
        thread.start()

    def _on_ck_checked(self, ok: bool, reason: str) -> None:
        self.chk_ck.setEnabled(True)
        if ok:
            self.ck_toggled.emit(True)
            self.lbl_ck.setStyleSheet("color:#888;")
            self.lbl_ck.setText("有効にしました（アプリ再起動後に反映）")
            return
        self.chk_ck.blockSignals(True)
        self.chk_ck.setChecked(False)
        self.chk_ck.blockSignals(False)
        self.lbl_ck.setText(reason)
        self.lbl_ck.setStyleSheet("color:#c33;")

    # ----- download --------------------------------------------------------
    def _on_download(self) -> None:
        sel = set(self._selector.selected_filenames())
        items = [m for m in self._selector.manifest if m.filename in sel]
        if not items:
            return
        self._selector.btn_download.setEnabled(False)

        self._thread = QThread(self)
        self._worker = _DownloadWorker(self._paths, items)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # 進捗は各プリセット行の右に数値で出す（selector が集計・表示）。
        self._worker.progress.connect(self._selector.update_progress)
        self._worker.file_done.connect(self._selector.mark_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_file_failed(self, filename: str, msg: str) -> None:
        self._selector.mark_file_failed(filename)

    def _on_finished(self) -> None:
        self._selector.btn_download.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread is not None and self._thread.isRunning():
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(3000)
        # SageAttention インストール中はスレッドを止めない（親なし +
        # _SAGE_JOBS 保持で完走する。UI への接続は破棄時に自動切断）。
        super().closeEvent(event)
