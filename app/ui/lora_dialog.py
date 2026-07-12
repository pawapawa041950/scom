"""LoRA ブラウザウィンドウ（非モーダル・親なし）。

models/loras のファイルをサムネイル付きグリッドで一覧表示する。サムネと
メタ情報（トリガーワード・ベースモデル等）は SHA256 を元に civitai から
取得し userdata/lora_cache/ にキャッシュされる（app/lora.py）。ハッシュ
計算とネットワークアクセスはワーカースレッドで行い、グリッドはまず
ファイル名だけで即表示 → 取得できた順にサムネが埋まる。

適用そのものはメインウィンドウが行う（apply/remove シグナル）。適用中の
状態は set_applied() で常にメイン側から供給される。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QSize, QThread, QUrl, Signal, QObject
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from .. import config, lora

_ICON_SIZE = 144
_GRID_SIZE = QSize(168, 196)
_APPLIED_BG = QColor("#1a4d2e")  # ✓ 適用中バッジの背景（緑系）


class _MetaWorker(QObject):
    """LoRA ごとのハッシュ計算 + civitai 取得（1本のスレッドで順次実行）。"""
    info = Signal(str, dict)     # relname, {"sha", "thumb", "meta"}
    status = Signal(str)
    done = Signal()

    def __init__(self, names: list[str], lora_dir: Path, cache_dir: Path):
        super().__init__()
        self._names = list(names)
        self._dir = Path(lora_dir)
        self._cache_dir = Path(cache_dir)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cache = lora.LoraCache(self._cache_dir)
        offline = 0  # 連続ネットワーク失敗数; 2回で以後の問い合わせを諦める
        for n, relname in enumerate(self._names):
            if self._cancel:
                break
            path = self._dir / relname
            entry = cache.lookup(relname, path)
            if entry is None:
                self.status.emit(
                    f"ハッシュ計算中 ({n + 1}/{len(self._names)}): {relname}")
                try:
                    sha = lora.sha256_file(path)
                except OSError:
                    continue
                meta: Optional[dict] = None
                if offline < 2:
                    self.status.emit(
                        f"civitai 照会中 ({n + 1}/{len(self._names)}): "
                        f"{relname}")
                    try:
                        meta = lora.fetch_civitai_meta(sha)
                        offline = 0
                    except (OSError, ValueError):
                        offline += 1
                        meta = "error"  # ネットワーク失敗: キャッシュしない
                if meta == "error" or offline >= 2:
                    # ハッシュだけ即席エントリで通知（次回開いたとき再照会）
                    self.info.emit(relname, {"sha": sha, "thumb": "",
                                             "meta": {}})
                    continue
                cache.store(relname, path, sha,
                            meta if meta else {"found": False})
                entry = cache.lookup(relname, path)
                if entry is None:
                    continue
            sha = str(entry.get("sha256", ""))
            meta = entry.get("meta") or {}
            thumb = ""
            if meta.get("found"):
                try:
                    thumb = cache.ensure_thumb(
                        sha, str(meta.get("preview_url", "")))
                except (OSError, ValueError):
                    thumb = ""
            self.info.emit(relname, {"sha": sha, "thumb": thumb,
                                     "meta": dict(meta)})
        self.status.emit("")
        self.done.emit()


class LoraDialog(QDialog):
    apply_requested = Signal(str, float)     # relname, strength
    remove_requested = Signal(str)           # relname
    insert_prompt_requested = Signal(str)    # text to append to the prompt

    def __init__(self, lora_dir: Path, cache_dir: Path,
                 preset_fn: Callable[[], str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("LoRA")
        self.setMinimumSize(900, 560)
        self._lora_dir = Path(lora_dir)
        self._cache_dir = Path(cache_dir)
        self._preset_fn = preset_fn
        self._items: dict[str, QListWidgetItem] = {}
        self._meta: dict[str, dict] = {}   # relname -> {"sha","thumb","meta"}
        self._applied: dict[str, float] = {}
        self._thread: Optional[QThread] = None
        self._worker: Optional[_MetaWorker] = None
        self._placeholder = self._make_placeholder()

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("検索…")
        self.ed_search.textChanged.connect(self._apply_filter)
        self.cb_filter = QComboBox()
        self.cb_filter.addItem("表示モデルに合わせる", "preset")
        self.cb_filter.addItem("すべて", "all")
        self.cb_filter.setToolTip(
            "civitai のベースモデル情報から現在の表示モデル系統に合う "
            "LoRA だけを表示します（情報が無いものは常に表示）")
        self.cb_filter.currentIndexChanged.connect(self._apply_filter)
        btn_rescan = QPushButton("再スキャン")
        btn_rescan.clicked.connect(self.rescan)
        top.addWidget(self.ed_search, stretch=1)
        top.addWidget(QLabel("ベースモデル:"))
        top.addWidget(self.cb_filter)
        top.addWidget(btn_rescan)
        root.addLayout(top)

        panes = QHBoxLayout()
        root.addLayout(panes, stretch=1)

        # ----- left: thumbnail grid ---------------------------------------
        self.lst = QListWidget()
        self.lst.setViewMode(QListWidget.IconMode)
        self.lst.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self.lst.setGridSize(_GRID_SIZE)
        self.lst.setResizeMode(QListWidget.Adjust)
        self.lst.setMovement(QListWidget.Static)
        self.lst.setWordWrap(True)
        self.lst.setUniformItemSizes(True)
        self.lst.currentItemChanged.connect(self._on_selection)
        self.lst.itemDoubleClicked.connect(self._on_double_click)
        panes.addWidget(self.lst, stretch=1)

        # ----- right: detail pane ------------------------------------------
        detail = QVBoxLayout()
        self.lbl_thumb = QLabel("プレビューなし")
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setFixedSize(256, 256)
        self.lbl_thumb.setStyleSheet(
            "QLabel { background:#1e1e1e; color:#888; border:1px solid #333; }")
        self.lbl_name = QLabel("")
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setStyleSheet("font-weight: bold;")
        self.lbl_base = QLabel("")
        self.lbl_words = QLabel("")
        self.lbl_words.setWordWrap(True)
        self.lbl_words.setTextFormat(Qt.RichText)
        self.lbl_words.setToolTip("クリックでプロンプト欄に追記します")
        self.lbl_words.linkActivated.connect(
            lambda w: self.insert_prompt_requested.emit(w))
        self.btn_civitai = QPushButton("civitai を開く")
        self.btn_civitai.setEnabled(False)
        self.btn_civitai.clicked.connect(self._open_civitai)

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("強度"))
        self.sp_strength = QDoubleSpinBox()
        self.sp_strength.setRange(-4.0, 4.0)
        self.sp_strength.setDecimals(2)
        self.sp_strength.setSingleStep(0.05)
        self.sp_strength.setValue(1.0)
        strength_row.addWidget(self.sp_strength)
        strength_row.addStretch(1)

        self.btn_apply = QPushButton("適用")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply_clicked)

        detail.addWidget(self.lbl_thumb, alignment=Qt.AlignHCenter)
        detail.addWidget(self.lbl_name)
        detail.addWidget(self.lbl_base)
        detail.addWidget(QLabel("トリガーワード:"))
        detail.addWidget(self.lbl_words)
        detail.addWidget(self.btn_civitai)
        detail.addStretch(1)
        detail.addLayout(strength_row)
        detail.addWidget(self.btn_apply)
        side = QWidget()
        side.setLayout(detail)
        side.setFixedWidth(280)
        panes.addWidget(side)

        bottom = QHBoxLayout()
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #888;")
        self.lbl_applied = QLabel("適用中: なし")
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        bottom.addWidget(self.lbl_status, stretch=1)
        bottom.addWidget(self.lbl_applied)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

        self.rescan()

    # ----- grid population ---------------------------------------------------
    @staticmethod
    def _make_placeholder() -> QIcon:
        pix = QPixmap(_ICON_SIZE, _ICON_SIZE)
        pix.fill(QColor("#2a2a2a"))
        return QIcon(pix)

    def rescan(self) -> None:
        """(Re)list models/loras and restart the metadata worker."""
        self._stop_worker()
        names = config.scan_models("loras")
        self.lst.clear()
        self._items.clear()
        # 既知メタは残す（ワーカーがキャッシュから同じ内容を再供給する）
        for relname in names:
            item = QListWidgetItem(self._placeholder, relname)
            item.setData(Qt.UserRole, relname)
            item.setToolTip(relname)
            self.lst.addItem(item)
            self._items[relname] = item
        self._refresh_applied_marks()
        self._apply_filter()
        if not names:
            self.lbl_status.setText(
                f"LoRA がありません: {self._lora_dir} に配置してください")
            return
        self._worker = _MetaWorker(names, self._lora_dir, self._cache_dir)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.info.connect(self._on_info)
        self._worker.status.connect(self.lbl_status.setText)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._on_worker_finished)
        self._thread.start()

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10_000)
        self._thread = None
        self._worker = None

    def _on_worker_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _on_info(self, relname: str, info: dict) -> None:
        self._meta[relname] = info
        item = self._items.get(relname)
        if item is None:
            return
        thumb = str(info.get("thumb", ""))
        if thumb:
            pix = QPixmap(thumb)
            if not pix.isNull():
                item.setIcon(QIcon(pix))
        self._apply_filter()
        if item is self.lst.currentItem():
            self._show_detail(relname)

    # ----- filtering -----------------------------------------------------------
    def _visible(self, relname: str) -> bool:
        text = self.ed_search.text().strip().lower()
        if text and text not in relname.lower():
            return False
        if self.cb_filter.currentData() == "preset":
            meta = (self._meta.get(relname) or {}).get("meta") or {}
            fams = lora.families_from_base_model(
                str(meta.get("base_model", "")))
            if fams and self._preset_fn() not in fams:
                return False
        return True

    def _apply_filter(self, *_a) -> None:
        for relname, item in self._items.items():
            item.setHidden(not self._visible(relname))

    # ----- applied state (supplied by the main window) --------------------------
    def set_applied(self, applied: dict[str, float]) -> None:
        self._applied = dict(applied)
        self._refresh_applied_marks()
        if applied:
            self.lbl_applied.setText("適用中: " + ", ".join(
                f"{Path(n).stem}×{w:g}" for n, w in applied.items()))
        else:
            self.lbl_applied.setText("適用中: なし")
        current = self.lst.currentItem()
        if current is not None:
            self._show_detail(str(current.data(Qt.UserRole)))

    def _refresh_applied_marks(self) -> None:
        for relname, item in self._items.items():
            if relname in self._applied:
                item.setText(f"✓ {relname}")
                item.setBackground(_APPLIED_BG)
            else:
                item.setText(relname)
                item.setBackground(QBrush())

    # ----- detail pane -----------------------------------------------------------
    def _on_selection(self, current, _previous) -> None:
        if current is None:
            self.btn_apply.setEnabled(False)
            return
        self._show_detail(str(current.data(Qt.UserRole)))

    def _show_detail(self, relname: str) -> None:
        info = self._meta.get(relname) or {}
        meta = info.get("meta") or {}
        thumb = str(info.get("thumb", ""))
        pix = QPixmap(thumb) if thumb else QPixmap()
        if not pix.isNull():
            self.lbl_thumb.setPixmap(pix.scaled(
                self.lbl_thumb.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        else:
            self.lbl_thumb.setPixmap(QPixmap())
            self.lbl_thumb.setText(
                "プレビューなし" if meta else "情報取得中…")
        if meta.get("found"):
            title = meta.get("name") or relname
            if meta.get("version"):
                title += f"（{meta['version']}）"
            self.lbl_name.setText(title)
            base = str(meta.get("base_model", ""))
            fams = lora.families_from_base_model(base)
            fam_note = f" → {'/'.join(sorted(fams))}" if fams else ""
            self.lbl_base.setText(f"ベースモデル: {base or '不明'}{fam_note}")
            words = list(meta.get("trained_words") or [])
            if words:
                links = ", ".join(
                    f'<a href="{w}">{w}</a>' for w in words)
                self.lbl_words.setText(links)
            else:
                self.lbl_words.setText("（なし）")
            self._civitai_url = str(meta.get("url", ""))
        else:
            self.lbl_name.setText(relname)
            self.lbl_base.setText("ベースモデル: 不明")
            self.lbl_words.setText("（civitai に情報がありません）"
                                   if meta else "（情報取得中…）")
            self._civitai_url = ""
        self.btn_civitai.setEnabled(bool(self._civitai_url))
        applied = relname in self._applied
        if applied:
            self.sp_strength.setValue(float(self._applied[relname]))
        self.btn_apply.setText("解除" if applied else "適用")
        self.btn_apply.setEnabled(True)

    def _open_civitai(self) -> None:
        if getattr(self, "_civitai_url", ""):
            QDesktopServices.openUrl(QUrl(self._civitai_url))

    # ----- apply / remove -----------------------------------------------------
    def _current_relname(self) -> Optional[str]:
        item = self.lst.currentItem()
        return None if item is None else str(item.data(Qt.UserRole))

    def _on_apply_clicked(self) -> None:
        relname = self._current_relname()
        if relname is None:
            return
        if relname in self._applied:
            self.remove_requested.emit(relname)
        else:
            self.apply_requested.emit(relname, self.sp_strength.value())

    def _on_double_click(self, item) -> None:
        relname = str(item.data(Qt.UserRole))
        if relname in self._applied:
            self.remove_requested.emit(relname)
        else:
            self.apply_requested.emit(relname, self.sp_strength.value())

    # ----- lifecycle -------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._stop_worker()
        super().closeEvent(event)
