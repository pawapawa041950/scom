"""モデルマージ管理ウィンドウ（非モーダル・2ペイン）。

左ペイン: 「＋ 新規作成」と作成済みマージモデルの一覧（各行: ゴミ箱 +
RAM状態マーク ●/○ + 名前。右クリックで名前の変更）。下部に「メモリ全解放」。
右ペイン: 新規作成時は編集可能なレシピエディタ（1行 = モデル + 相対比率、
量子化・省メモリオプション）。既存エントリ選択時は同じUIを読み取り専用で表示
し、「複製して新規作成」で内容を引き継いだ新規作成に移れる。

このウィンドウはレシピの管理だけを行い、実行はシグナル経由でメインウィンドウ
に依頼する。エントリ一覧とRAM状態は set_entries() で常にメイン側から供給される。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout,
    QInputDialog, QRadioButton,
    QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from ..workflow import (
    entry_ref_ids, is_merge_spec, merge_source_file, merge_source_label,
    merge_spec_from_entry, norm_source,
)
from .widgets import WideComboBox
from .window_state import bind_geometry

# モデル欄の項目データ: マージ項目は "merge:<id>"（Windows のファイル名に
# ":" は使えないのでファイル名と衝突しない）。タプル等の Python オブジェクトは Qt が同一オブジェクトかどうかで
# 比べるため findData() で見つからず、存在する項目が「削除済み」扱いに
# なっていた。
_MERGE_ITEM = "merge:"


def _merge_item_data(entry_id) -> str:
    return f"{_MERGE_ITEM}{int(entry_id)}"


# Returns "anima" | "krea2" | "qwen21" | "sdxl" | "unknown" for a diffusion
# file name.
FamilyFn = Callable[[str], str]

NOTE_TEXT = (
    "マージモデルは画像生成時にキャッシュに残っていなければ再マージが走ります。"
    "キャッシュ上のマージモデルはRAMが逼迫してきたときに使われていないモデル"
    "から自動で削除されます。"
)


class MergeDialog(QDialog):
    # Recipe execution / management requests, handled by the main window.
    # entries [(model, weight)], loras [(lora, strength)],
    # diffs [(model_a, model_b, strength)], quant, low_mem
    merge_requested = Signal(object, object, object, str, bool)
    # 作成済み項目の設定を上書き: entry_id + merge_requested と同じ引数
    update_requested = Signal(int, object, object, object, str, bool)
    save_requested = Signal(object, object, object, str, bool, str)  # + filename
    # 差分から LoRA を作成: model_a, model_b, rank, name, to_cache
    # （model_a / model_b はファイル名、または入れ子マージの dict。name は
    # to_cache=True ならキャッシュ上の表示名、False なら models/loras の
    # ファイル名）
    extract_requested = Signal(object, object, int, str, bool)
    delete_requested = Signal(int)                     # entry id
    rename_requested = Signal(int, str)                # entry id, new name
    free_memory_requested = Signal()

    def __init__(self, models: list[str], family_fn: FamilyFn,
                 model_dir: Path, loras: list[str] = (),
                 lora_family_fn: Optional[FamilyFn] = None,
                 lora_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("モデルマージ")
        self.setMinimumWidth(780)
        bind_geometry(self, "merge")
        self._models = list(models)
        self._family = family_fn
        self._model_dir = model_dir
        self._loras = list(loras)
        self._lora_family = lora_family_fn or (lambda _n: "unknown")
        self._lora_dir = lora_dir
        self._rows: list[dict] = []
        self._lora_rows: list[dict] = []
        self._diff_rows: list[dict] = []
        self._entries: list[dict] = []
        self._built_ids: set[int] = set()
        self._editable = True

        root = QVBoxLayout(self)
        note = QLabel(NOTE_TEXT)
        note.setWordWrap(True)
        root.addWidget(note)
        panes = QHBoxLayout()
        root.addLayout(panes, stretch=1)

        # マージ実行中の進捗（メインウィンドウから set_merge_* で供給）。
        # 非表示でも高さを確保する — 表示/非表示で最小サイズが変わると
        # QWindowsWindow::setGeometry の警告が出るため（XYZ ウィンドウと同様）。
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        sp = self.progress.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.progress.setSizePolicy(sp)
        root.addWidget(self.progress)

        # ----- left pane: entry list --------------------------------------
        left = QVBoxLayout()
        self.lst = QListWidget()
        self.lst.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst.customContextMenuRequested.connect(self._on_context_menu)
        self.lst.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.lst, stretch=1)
        btn_free = QPushButton("メモリ全解放")
        btn_free.setToolTip(
            "バックエンドのキャッシュとロード済みモデルをすべて解放します。"
            "マージモデルの一覧は残り、次回使用時に自動で再構築されます")
        btn_free.clicked.connect(lambda: self.free_memory_requested.emit())
        left.addWidget(btn_free)
        panes.addLayout(left, 1)

        # ----- right pane: recipe editor / read-only view ------------------
        right = QVBoxLayout()
        right.addWidget(QLabel(
            "マージするモデルと比率（相対値）を指定してください。"
            "比率は合計に対する割合として扱われます。"))

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(4)
        right.addLayout(self._rows_layout)

        self.btn_add = QPushButton("＋ モデルを追加")
        self.btn_add.clicked.connect(lambda: (self._add_row(), self._refresh()))
        add_row = QHBoxLayout()
        add_row.addWidget(self.btn_add)
        add_row.addStretch(1)
        right.addLayout(add_row)

        self.lbl_warn = QLabel("アーキテクチャの違うモデルが選択されています")
        self.lbl_warn.setStyleSheet("color: red;")
        self.lbl_warn.hide()
        right.addWidget(self.lbl_warn)

        # ----- 焼き込む LoRA（任意） -----
        lbl_lora = QLabel(
            "焼き込む LoRA / モデル差分（任意）: マージ結果の重みに畳み込みます。"
            "生成時に LoRA を掛けるのと同じ効果で、量子化の前に適用されます"
            "（text encoder 側の重みを持つ LoRA はその部分だけ無視されます）。"
            "モデル差分は「モデル1 − モデル2」× 強度をそのまま足し込みます"
            "（ランク制限の無い LoRA 相当）。")
        lbl_lora.setWordWrap(True)
        right.addWidget(lbl_lora)
        self._lora_rows_layout = QVBoxLayout()
        self._lora_rows_layout.setSpacing(4)
        right.addLayout(self._lora_rows_layout)
        self._diff_rows_layout = QVBoxLayout()
        self._diff_rows_layout.setSpacing(4)
        right.addLayout(self._diff_rows_layout)
        self.btn_add_lora = QPushButton("＋ LoRA を追加")
        self.btn_add_lora.clicked.connect(
            lambda: (self._add_lora_row(), self._refresh()))
        self.btn_add_diff = QPushButton("＋ モデルの差分を追加")
        self.btn_add_diff.setToolTip(
            "2 つのモデルの差分（モデル1 − モデル2）をマージ結果に足し込みます")
        self.btn_add_diff.clicked.connect(
            lambda: (self._add_diff_row(), self._refresh()))
        add_lora_row = QHBoxLayout()
        add_lora_row.addWidget(self.btn_add_lora)
        add_lora_row.addWidget(self.btn_add_diff)
        add_lora_row.addStretch(1)
        right.addLayout(add_lora_row)
        self.lbl_lora_warn = QLabel(
            "モデルと系統の違う LoRA が選択されています（効かないか、"
            "マージが失敗します）")
        self.lbl_lora_warn.setStyleSheet("color: red;")
        self.lbl_lora_warn.hide()
        right.addWidget(self.lbl_lora_warn)
        self.lbl_nested_warn = QLabel(
            "量子化済みのマージモデルを素材にしています。逆量子化した重みを"
            "使うため量子化誤差が乗ります（素材にする側は量子化なしを推奨）")
        self.lbl_nested_warn.setStyleSheet("color: #c80;")
        self.lbl_nested_warn.hide()
        right.addWidget(self.lbl_nested_warn)
        self.lbl_dirty = QLabel(
            "※ 設定が変更されています。「モデルマージ（この項目を上書き）」で"
            "更新すると、古い設定のキャッシュは破棄されます")
        self.lbl_dirty.setWordWrap(True)
        self.lbl_dirty.setStyleSheet("color: #c80;")
        self.lbl_dirty.hide()
        right.addWidget(self.lbl_dirty)

        self.lbl_info = QLabel("")
        right.addWidget(self.lbl_info)

        self.chk_quant = QCheckBox("量子化する")
        self.chk_quant.setToolTip(
            "マージ結果の重みを低精度化してメモリ/VRAM を節約します")
        self.cb_quant = WideComboBox()
        self.cb_quant.addItem("fp8", "fp8")
        self.cb_quant.addItem("int8 ConvRot", "int8_convrot")
        self.cb_quant.addItem("int4 ConvRot（ハイブリッド）", "int4_convrot")
        self.cb_quant.addItem("int4 ConvRot（完全4bit）", "int4_convrot_full")
        self.cb_quant.setItemData(
            0, "Linear 層を float8_e4m3fn（per-tensor scale）に量子化。"
               "サイズ約1/2、わずかに精度低下", Qt.ToolTipRole)
        self.cb_quant.setItemData(
            1, "回転（Hadamard）ベースの INT8。ほぼ無損失で fp8 より高品質、"
               "RTX 30 系以上の INT8 Tensor Core で高速。"
               "DiT ブロックのみ量子化（sensitive 層は高精度維持）",
            Qt.ToolTipRole)
        self.cb_quant.setItemData(
            2, "回転（Hadamard）ベースの INT4/INT8 ハイブリッド（convrot_w4a4）。"
               "Q/K/V/up 系は int4、誤差の大きい出力側 projection は int8、"
               "行列積は int8 で実行（速度は int8 と同等・サイズはより小さい）。"
               "int8 よりわずかに精度低下（adaLN 等の sensitive 層は高精度維持）",
            Qt.ToolTipRole)
        self.cb_quant.setItemData(
            3, "対象の Linear 層をすべて int4 にする最小サイズ版。"
               "krea2 級の大モデル向き。anima 等の小モデルでは劣化が"
               "目立つためハイブリッド推奨（adaLN 等の sensitive 層は"
               "高精度維持）", Qt.ToolTipRole)
        self.cb_quant.setEnabled(False)
        self.chk_quant.toggled.connect(self._sync_quant_enabled)
        # 変更表示（※ 設定が変更されています）を追従させる。
        self.chk_quant.toggled.connect(self._sync_action_buttons)
        self.cb_quant.currentIndexChanged.connect(self._sync_action_buttons)
        quant_row = QHBoxLayout()
        quant_row.addWidget(self.chk_quant)
        quant_row.addWidget(self.cb_quant)
        quant_row.addStretch(1)
        right.addLayout(quant_row)

        self.chk_lowmem = QCheckBox("省メモリモード（1モデルずつ逐次マージ）")
        self.chk_lowmem.setToolTip(
            "モデルを1個ずつ読み込んで順に合成します。ピークメモリが"
            "「結果1個分＋ソース1個分」程度まで下がる代わり、途中経過を"
            "bf16 で保持するためごくわずかに精度が落ちます。"
            "OFF は全モデルを同時に開いて fp32 で一括合成します（最高精度）")
        right.addWidget(self.chk_lowmem)
        self.chk_lowmem.toggled.connect(self._sync_action_buttons)

        right.addStretch(1)

        # 出力先（ラジオ）→ 実行ボタン（モデルマージ / 差分を LoRA 化）→ 閉じる。
        self.rb_cache = QRadioButton("キャッシュに配置")
        self.rb_cache.setToolTip(
            "バックエンドのメインメモリ上に置きます。マージモデルは一覧に登録"
            "されてモデル選択に並び、差分 LoRA は LoRA ウィンドウに"
            "「キャッシュ」として並びます（消えても使用時に自動で再作成）")
        self.rb_file = QRadioButton("ファイルとして保存")
        self.rb_file.setToolTip(
            "マージモデルは models/diffusion_models、差分 LoRA は models/loras に"
            "safetensors として保存します")
        self.rb_cache.setChecked(True)
        self._dest_group = QButtonGroup(self)
        self._dest_group.addButton(self.rb_cache)
        self._dest_group.addButton(self.rb_file)
        self.rb_cache.toggled.connect(self._sync_action_buttons)
        self.btn_merge = QPushButton("モデルマージ")
        self.btn_merge.setToolTip(
            "表示中の構成でマージします（出力先は上の選択に従う）")
        self.btn_merge.clicked.connect(self._on_merge_clicked)
        self.btn_extract = QPushButton("差分を LoRA 化")
        self.btn_extract.setToolTip(
            "モデルを 2 個だけ指定したときに使えます。「モデル1 − モデル2」の"
            "差分を低ランク近似した LoRA を作ります（強度 1.0 でモデル2 + "
            "LoRA ≒ モデル1。比率・LoRA 行・差分行は無視。出力先は上の選択に"
            "従う）")
        self.btn_extract.setEnabled(False)
        self.btn_extract.clicked.connect(
            lambda: self._on_extract(self.rb_cache.isChecked()))
        self.btn_dup = QPushButton("複製して新規作成")
        self.btn_dup.setToolTip("この構成をコピーした新規作成に切り替えます")
        self.btn_dup.clicked.connect(self._on_duplicate)
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        row_dest = QHBoxLayout()
        row_dest.addWidget(QLabel("出力先:"))
        row_dest.addWidget(self.rb_cache)
        row_dest.addWidget(self.rb_file)
        row_dest.addStretch(1)
        row_run = QHBoxLayout()
        row_run.addStretch(1)
        row_run.addWidget(self.btn_merge)
        row_run.addWidget(self.btn_extract)
        row_close = QHBoxLayout()
        row_close.addStretch(1)
        row_close.addWidget(self.btn_dup)
        row_close.addWidget(btn_close)
        for r in (row_dest, row_run, row_close):
            right.addLayout(r)
        panes.addLayout(right, 2)

        while len(self._rows) < 2:
            self._add_row()
        self.set_entries([], set())

    # ----- left pane -------------------------------------------------------
    def set_entries(self, entries: list[dict], built_ids: set[int]) -> None:
        """Rebuild the entry list (called by the main window on any change)."""
        self._entries = [dict(e) for e in entries]
        self._built_ids = set(built_ids)
        selected = self.selected_entry_id()
        self.lst.blockSignals(True)
        self.lst.clear()
        self.lst.addItem(QListWidgetItem("＋ 新規作成"))
        for e in self._entries:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, int(e["id"]))
            self.lst.addItem(item)
            self.lst.setItemWidget(item, self._make_row_widget(e))
        row = 0
        if selected is not None:
            for i in range(1, self.lst.count()):
                if self.lst.item(i).data(Qt.UserRole) == selected:
                    row = i
        self.lst.setCurrentRow(row)
        self.lst.blockSignals(False)
        self._on_row_changed(self.lst.currentRow())

    def _refill_source_combos(self) -> None:
        """モデル欄の候補（入れ子マージ）を現在の一覧・選択に合わせ直す
        （選択中の値は保つ）。"""
        for r in self._rows:
            self._fill_source_combo(r["combo"], self._source_of(r["combo"]))
        for r in self._diff_rows:
            for c in (r["combo_a"], r["combo_b"]):
                self._fill_source_combo(c, self._source_of(c))

    # ----- model source combos (files + saved merges) --------------------------
    def _fill_source_combo(self, combo, current=None) -> None:
        """候補 = 登録済みマージ項目（"[マージ] 名前"、上位）+ ファイル。
        項目の data はファイル名（str）か ("merge", id)。"""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("", "")
        excluded = self._excluded_source_ids()
        usable = [e for e in self._entries if int(e["id"]) not in excluded]
        for e in usable:
            combo.addItem(merge_source_label(e), _merge_item_data(e["id"]))
            # メイン画面のモデル欄と同じ色（main_window.MERGE_COLOR）で区別する。
            combo.setItemData(combo.count() - 1, QBrush(QColor("#1a7f37")),
                              Qt.ForegroundRole)
        if usable and self._models:
            combo.insertSeparator(combo.count())
        for n in self._models:
            combo.addItem(n, n)
        combo.blockSignals(False)
        if current:
            self._select_source(combo, current)

    def _source_of(self, combo):
        """選択中のソース: "" | ファイル名 | 入れ子マージの dict（スナップ
        ショット）。削除済み項目のプレースホルダは dict をそのまま持つ。"""
        data = combo.currentData()
        if isinstance(data, str) and data.startswith(_MERGE_ITEM):
            e = self._entry_by_id(int(data[len(_MERGE_ITEM):]))
            return merge_spec_from_entry(e) if e is not None else ""
        if isinstance(data, dict):
            return data
        return str(data or "")

    def _select_source(self, combo, src) -> None:
        if is_merge_spec(src):
            idx = combo.findData(_merge_item_data(src.get("id", 0) or 0))
            if idx < 0:
                # 参照先がもう無い（削除済み）: スナップショットをそのまま
                # 候補に足して表示・再利用できるようにする。
                combo.addItem(f"{merge_source_label(src)}（削除済み）", src)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)
            return
        idx = combo.findData(str(src)) if src else 0
        combo.setCurrentIndex(max(idx, 0))

    # ----- entry references (nested merges) ------------------------------------
    def _ids(self) -> set:
        return {int(e["id"]) for e in self._entries}

    def _is_broken(self, entry: dict) -> bool:
        """削除済みのマージを（入れ子のどこかで）素材にしている項目。"""
        return bool(entry_ref_ids(entry) - self._ids())

    def _excluded_source_ids(self) -> set:
        """素材の候補から外す項目: 使用不可の項目と、編集中の項目自身・
        それを素材にしている項目（選ぶと循環するため）。"""
        out = {int(e["id"]) for e in self._entries if self._is_broken(e)}
        eid = self.selected_entry_id()
        if eid is not None:
            out.add(int(eid))
            out |= {int(e["id"]) for e in self._entries
                    if int(eid) in entry_ref_ids(e)}
        return out

    def _make_row_widget(self, entry: dict) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 2, 4, 2)
        trash = QPushButton("🗑")
        trash.setFixedWidth(28)
        trash.setToolTip("このマージモデルを一覧から削除し、メモリからも解放します")
        trash.clicked.connect(
            lambda *_a, eid=int(entry["id"]): self._on_trash(eid))
        if self._is_broken(entry):
            lbl = QLabel(f"✕ {entry['name']}（素材が削除済み）")
            lbl.setStyleSheet("color: #c33;")
            lbl.setToolTip(
                "素材にしていたマージモデルが削除されたため使用できません。"
                "選択して素材を変更し「モデルマージ」で更新すると再び使えます")
        else:
            mark = "●" if int(entry["id"]) in self._built_ids else "○"
            lbl = QLabel(f"{mark} {entry['name']}")
            lbl.setToolTip(
                "● = メモリ上に構築済み / ○ = 未構築（生成時に自動マージ）")
        h.addWidget(trash)
        h.addWidget(lbl, stretch=1)
        return w

    def selected_entry_id(self) -> Optional[int]:
        item = self.lst.currentItem()
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return int(data) if data is not None else None

    def select_entry(self, entry_id: int) -> None:
        for i in range(1, self.lst.count()):
            if self.lst.item(i).data(Qt.UserRole) == entry_id:
                self.lst.setCurrentRow(i)
                return

    def _entry_by_id(self, entry_id: int) -> Optional[dict]:
        for e in self._entries:
            if int(e["id"]) == entry_id:
                return e
        return None

    def _on_trash(self, entry_id: int) -> None:
        e = self._entry_by_id(entry_id)
        name = e["name"] if e else str(entry_id)
        res = QMessageBox.question(
            self, "削除の確認", f"「{name}」を一覧から削除しますか？")
        if res == QMessageBox.Yes:
            self.delete_requested.emit(entry_id)

    def _on_context_menu(self, pos) -> None:
        item = self.lst.itemAt(pos)
        if item is None or item.data(Qt.UserRole) is None:
            return  # no menu on 新規作成
        entry_id = int(item.data(Qt.UserRole))
        e = self._entry_by_id(entry_id)
        if e is None:
            return
        menu = QMenu(self)
        act = QAction("名前の変更", menu)
        menu.addAction(act)
        if menu.exec(self.lst.mapToGlobal(pos)) is act:
            name, ok = QInputDialog.getText(
                self, "名前の変更", "新しい名前:", text=e["name"])
            name = name.strip()
            if ok and name:
                self.rename_requested.emit(entry_id, name)

    def _on_row_changed(self, row: int) -> None:
        # 候補（自分自身・循環になる項目を除く）は選択ごとに変わる。
        self._refill_source_combos()
        entry_id = self.selected_entry_id()
        self._set_editable(True)
        if entry_id is None:
            self._refresh()
            return
        e = self._entry_by_id(entry_id)
        if e is None:
            return
        # 作成済みの項目も編集できる（「モデルマージ」で上書き更新）。
        self._load_recipe(*self._stored_recipe(e))
        self._refresh()

    @staticmethod
    def _stored_recipe(e: dict):
        return ([(n, float(w)) for n, w in e["models"]],
                str(e.get("quant", "")),
                bool(e.get("low_memory", False)),
                [(n, float(s)) for n, s in e.get("loras", [])],
                [(a, b, float(s)) for a, b, s in e.get("diffs", [])])

    def _is_dirty(self) -> bool:
        """選択中の作成済み項目から設定が変わっているか。"""
        eid = self.selected_entry_id()
        e = self._entry_by_id(eid) if eid is not None else None
        if e is None:
            return False
        def canon(models, quant, low, loras, diffs):
            return (
                [[norm_source(n), float(w)] for n, w in models],
                str(quant), bool(low),
                [[str(n), float(s)] for n, s in loras],
                [[norm_source(a), norm_source(b), float(s)]
                 for a, b, s in diffs])
        return canon(*self._stored_recipe(e)) != canon(*self._displayed_recipe())

    # ----- right pane: rows --------------------------------------------------
    def _add_row(self, name: str = "", weight: float = 1.0) -> None:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        combo = WideComboBox()
        self._fill_source_combo(combo, name)
        combo.currentIndexChanged.connect(self._refresh)

        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1000.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(weight)
        spin.valueChanged.connect(self._refresh)

        pct = QLabel("")
        pct.setFixedWidth(44)
        pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        trash = QPushButton("🗑")
        trash.setFixedWidth(32)
        trash.setToolTip("この行を削除")
        trash.clicked.connect(lambda *_a, widget=w: self._remove_row(widget))
        # Every row has a trash icon; a single model is a valid "merge"
        # (RAM-pinned / quantized copy), so only the last row is protected
        # (see _refresh, which disables the icon when one row remains).

        h.addWidget(combo, stretch=1)
        h.addWidget(spin)
        h.addWidget(pct)
        h.addWidget(trash)

        self._rows_layout.addWidget(w)
        self._rows.append({"widget": w, "combo": combo, "spin": spin,
                           "pct": pct, "trash": trash})

    def _remove_row(self, widget: QWidget) -> None:
        if len(self._rows) <= 1:
            # 最後の 1 行は消さずに値を空に戻す（比率も既定の 1）。
            for row in self._rows:
                if row["widget"] is widget:
                    row["combo"].setCurrentIndex(0)
                    row["spin"].setValue(1.0)
            self._refresh()
            return
        for row in self._rows:
            if row["widget"] is widget:
                self._rows.remove(row)
                self._rows_layout.removeWidget(widget)
                widget.deleteLater()
                break
        self._refresh()

    # ----- right pane: LoRA rows -------------------------------------------
    def _add_lora_row(self, name: str = "", strength: float = 1.0) -> None:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        combo = WideComboBox()
        combo.addItem("")
        combo.addItems(self._loras)
        combo.setCurrentText(name)
        combo.currentTextChanged.connect(self._refresh)
        spin = QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(strength)
        spin.setToolTip("強度（生成時の LoRA 強度と同じ意味。負の値も可）")
        spin.valueChanged.connect(self._refresh)
        trash = QPushButton("🗑")
        trash.setFixedWidth(32)
        trash.setToolTip("この LoRA を外す")
        trash.clicked.connect(lambda *_a, widget=w: self._remove_lora_row(widget))
        h.addWidget(QLabel("LoRA:"))
        h.addWidget(combo, stretch=1)
        h.addWidget(spin)
        h.addWidget(trash)
        for x in (combo, spin, trash):
            x.setEnabled(self._editable)
        self._lora_rows_layout.addWidget(w)
        self._lora_rows.append({"widget": w, "combo": combo, "spin": spin,
                                "trash": trash})

    def _remove_lora_row(self, widget: QWidget, refresh: bool = True) -> None:
        for row in self._lora_rows:
            if row["widget"] is widget:
                self._lora_rows.remove(row)
                self._lora_rows_layout.removeWidget(widget)
                widget.deleteLater()
                break
        if refresh:
            self._refresh()

    def lora_entries(self) -> list[tuple[str, float]]:
        """Selected (lora, strength) pairs; rows with no file are skipped."""
        out = []
        for row in self._lora_rows:
            name = row["combo"].currentText().strip()
            if name:
                out.append((name, float(row["spin"].value())))
        return out

    # ----- right pane: model-difference rows (A − B) --------------------------
    def _add_diff_row(self, a: str = "", b: str = "",
                      strength: float = 1.0) -> None:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        combos = []
        for name in (a, b):
            c = WideComboBox()
            self._fill_source_combo(c, name)
            c.currentIndexChanged.connect(self._refresh)
            combos.append(c)
        spin = QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(strength)
        spin.setToolTip("強度: 結果 += 強度 × (モデル1 − モデル2)")
        spin.valueChanged.connect(self._refresh)
        trash = QPushButton("🗑")
        trash.setFixedWidth(32)
        trash.setToolTip("この差分を外す")
        trash.clicked.connect(lambda *_a, widget=w: self._remove_diff_row(widget))
        h.addWidget(QLabel("差分:"))
        h.addWidget(combos[0], stretch=1)
        h.addWidget(QLabel("−"))
        h.addWidget(combos[1], stretch=1)
        h.addWidget(spin)
        h.addWidget(trash)
        for x in (combos[0], combos[1], spin, trash):
            x.setEnabled(self._editable)
        self._diff_rows_layout.addWidget(w)
        self._diff_rows.append({"widget": w, "combo_a": combos[0],
                                "combo_b": combos[1], "spin": spin,
                                "trash": trash})

    def _remove_diff_row(self, widget: QWidget, refresh: bool = True) -> None:
        for row in self._diff_rows:
            if row["widget"] is widget:
                self._diff_rows.remove(row)
                self._diff_rows_layout.removeWidget(widget)
                widget.deleteLater()
                break
        if refresh:
            self._refresh()

    def diff_entries(self) -> list[tuple[str, str, float]]:
        """(model_a, model_b, strength) rows with both models chosen."""
        out = []
        for row in self._diff_rows:
            a = self._source_of(row["combo_a"])
            b = self._source_of(row["combo_b"])
            if a and b:
                out.append((a, b, float(row["spin"].value())))
        return out

    def _load_recipe(self, models: list[tuple[str, float]], quant: str,
                     low_memory: bool,
                     loras: list[tuple[str, float]] = (),
                     diffs: list[tuple[str, str, float]] = ()) -> None:
        """Fill the right pane with a recipe (used for view and duplicate)."""
        while self._lora_rows:
            self._remove_lora_row(self._lora_rows[-1]["widget"], refresh=False)
        for n, s in loras:
            self._add_lora_row(n, float(s))
        while self._diff_rows:
            self._remove_diff_row(self._diff_rows[-1]["widget"], refresh=False)
        for a, b, s in diffs:
            self._add_diff_row(a, b, float(s))
        # A fresh recipe starts with two empty rows; a saved one shows
        # exactly its models (one row for a single-model entry).
        want = max(len(models), 2 if not models else 1)
        while len(self._rows) > want:
            self._remove_row(self._rows[-1]["widget"])
        while len(self._rows) < want:
            self._add_row()
        for row, (name, w) in zip(self._rows, models + [("", 1.0)] * 2):
            self._select_source(row["combo"], name)
            row["spin"].setValue(float(w))
        self.chk_quant.setChecked(bool(quant))
        if quant:
            idx = self.cb_quant.findData(quant)
            if idx >= 0:
                self.cb_quant.setCurrentIndex(idx)
        self.chk_lowmem.setChecked(bool(low_memory))
        self._refresh()

    def _set_editable(self, editable: bool) -> None:
        self._editable = editable
        for row in self._rows:
            row["combo"].setEnabled(editable)
            row["spin"].setEnabled(editable)
            row["trash"].setEnabled(editable)
        for row in self._lora_rows:
            row["combo"].setEnabled(editable)
            row["spin"].setEnabled(editable)
            row["trash"].setEnabled(editable)
        for row in self._diff_rows:
            for x in (row["combo_a"], row["combo_b"], row["spin"], row["trash"]):
                x.setEnabled(editable)
        self.btn_add.setEnabled(editable)
        self.btn_add_lora.setEnabled(editable)
        self.btn_add_diff.setEnabled(editable)
        self.chk_quant.setEnabled(editable)
        self.chk_lowmem.setEnabled(editable)
        self._sync_quant_enabled()
        self._sync_action_buttons()

    def _sync_action_buttons(self, *_a) -> None:
        """「モデルマージ」: 新規作成ならキャッシュに新しく登録、作成済み
        項目の選択中は（キャッシュ配置なら）その項目を上書き更新。
        「差分を LoRA 化」はモデルがちょうど 2 個のときだけ。"""
        entry_sel = self.selected_entry_id() is not None
        self.btn_dup.setVisible(entry_sel)
        self.btn_merge.setEnabled(self._editable)
        if entry_sel and self.rb_cache.isChecked():
            self.btn_merge.setText("モデルマージ（この項目を上書き）")
            self.btn_merge.setToolTip(
                "選択中のマージモデルを表示中の設定で作り直します。古い設定の"
                "キャッシュは破棄され、この項目を素材にしているマージモデルも"
                "新しい設定で作り直されます")
        else:
            self.btn_merge.setText("モデルマージ")
            self.btn_merge.setToolTip(
                "表示中の構成でマージします（出力先は上の選択に従う）")
        self.btn_extract.setEnabled(len(self._displayed_recipe()[0]) == 2)
        dirty = entry_sel and self._is_dirty()
        self.lbl_dirty.setVisible(dirty)

    def _has_deleted_refs(self, entries, diffs) -> bool:
        missing = entry_ref_ids({"models": entries, "diffs": diffs}) - self._ids()
        if missing:
            QMessageBox.warning(
                self, "削除済みのマージモデル",
                "素材に削除済みのマージモデルが含まれているため実行できません。"
                "該当の行を別のモデルに変更してください。")
            return True
        return False

    def _on_merge_clicked(self) -> None:
        if self.rb_cache.isChecked():
            eid = self.selected_entry_id()
            if eid is not None:
                self._on_update(eid)
            else:
                self._on_merge()
        else:
            self._on_save()

    def _on_update(self, entry_id: int) -> None:
        entries = self.entries()
        if not self._validate(entries):
            return
        if self._has_deleted_refs(entries, self.diff_entries()):
            return
        self.update_requested.emit(int(entry_id), entries,
                                   self.lora_entries(), self.diff_entries(),
                                   self.quant_value(),
                                   self.chk_lowmem.isChecked())

    def _sync_quant_enabled(self, *_a) -> None:
        self.cb_quant.setEnabled(self._editable and self.chk_quant.isChecked())

    # ----- state --------------------------------------------------------------
    def entries(self) -> list:
        """Selected (source, weight) pairs; rows with no model are skipped.
        A source is a file name or a nested-merge dict."""
        out = []
        for row in self._rows:
            src = self._source_of(row["combo"])
            if src:
                out.append((src, float(row["spin"].value())))
        return out

    def quant_value(self) -> str:
        """Selected quantization mode: "" | "fp8" | "int8_convrot"
        | "int4_convrot" | "int4_convrot_full"."""
        if not self.chk_quant.isChecked():
            return ""
        return self.cb_quant.currentData() or ""

    def _refresh(self, *_args) -> None:
        entries = self.entries()
        total = sum(w for _n, w in entries)
        for row in self._rows:
            name = self._source_of(row["combo"])
            w = float(row["spin"].value())
            row["pct"].setText(f"{w / total * 100:.0f}%"
                               if name and total > 0 else "")
            row["trash"].setEnabled(self._editable)

        # Architecture check: warn when known families disagree (rows are
        # deliberately unfiltered, so mixing anima/krea2 files is possible).
        # Nested merges are judged by their leaf (first) model.
        sources = [n for n, _w in entries]
        sources += [n for a, b, _s in self.diff_entries() for n in (a, b)]
        fams = {self._family(merge_source_file(n)) for n in sources}
        # 量子化済みのマージを素材にすると、逆量子化した重みを使うので
        # その誤差が乗る（入れ子にする内側は量子化なしが望ましい）。
        self.lbl_nested_warn.setVisible(any(
            is_merge_spec(n) and n.get("quant") for n in sources))
        self.lbl_warn.setVisible(
            len(fams & {"anima", "krea2", "qwen21", "sdxl"}) > 1)
        # LoRA family check: a LoRA of another family (or of an unsupported
        # architecture, "other") has no keys to fold into this model.
        known = fams & {"anima", "krea2", "qwen21", "sdxl"}
        bad = False
        if known:
            for n, _s in self.lora_entries():
                lf = self._lora_family(n)
                if lf == "other" or (lf in ("anima", "krea2", "qwen21", "sdxl")
                                     and lf not in known):
                    bad = True
        self.lbl_lora_warn.setVisible(bad)

        size = 0
        for n, _w in entries:
            if is_merge_spec(n):
                continue
            try:
                size += (self._model_dir / n).stat().st_size
            except OSError:
                pass
        self.lbl_info.setText(
            f"選択モデル合計: {size / 1e9:.1f} GB" if size else "")
        self._sync_action_buttons()

    # ----- actions --------------------------------------------------------------
    def _displayed_recipe(self) -> tuple[list[tuple[str, float]], str, bool,
                                         list[tuple[str, float]],
                                         list[tuple[str, str, float]]]:
        """The recipe currently in the editor (作成済み項目の選択中も、
        編集中の内容): (models, quant, low_memory, loras, diffs)."""
        return (self.entries(), self.quant_value(),
                self.chk_lowmem.isChecked(), self.lora_entries(),
                self.diff_entries())

    def _validate(self, entries) -> bool:
        # 1 model is fine: that is a quantize-only (or copy-only) build.
        if len(entries) < 1:
            QMessageBox.warning(self, "入力不足",
                                "モデルを1個以上選択してください。")
            return False
        return True

    def _on_merge(self) -> None:
        entries = self.entries()
        if not self._validate(entries):
            return
        if self._has_deleted_refs(entries, self.diff_entries()):
            return
        self.merge_requested.emit(entries, self.lora_entries(),
                                  self.diff_entries(), self.quant_value(),
                                  self.chk_lowmem.isChecked())

    def _on_extract(self, to_cache: bool = False) -> None:
        """差分から LoRA を作成: モデル1 − モデル2（表示中の 2 行）。
        to_cache=True はキャッシュ（メインメモリ）、False はファイルに保存。"""
        models = self._displayed_recipe()[0]
        if len(models) != 2:
            QMessageBox.warning(self, "入力不足",
                                "モデルをちょうど 2 個指定してください。")
            return
        (a, _wa), (b, _wb) = models
        if a == b:
            QMessageBox.warning(self, "入力不足", "同じモデル同士の差分は空です。")
            return
        la, lb = merge_source_label(a), merge_source_label(b)
        title = ("差分から LoRA を作成しキャッシュに配置" if to_cache
                 else "差分から LoRA を作成しファイルに保存")
        rank, ok = QInputDialog.getInt(
            self, title,
            f"モデル1: {la}\nモデル2: {lb}\n\n「モデル1 − モデル2」を LoRA にします。\n"
            "ランク（大きいほど忠実、ファイルも大きい）:", 64, 1, 1024, 1)
        if not ok:
            return

        def stem(src) -> str:
            if is_merge_spec(src):
                return "".join(ch if ch not in "\\/:*?\"<>|" else "_"
                               for ch in str(src.get("name", "merge")))
            return Path(src).stem

        if to_cache:
            name, ok = QInputDialog.getText(
                self, title, "LoRA ウィンドウに表示する名前:",
                text=f"{stem(a)}-{stem(b)}_r{rank}")
            name = name.strip()
            if ok and name:
                self.extract_requested.emit(a, b, int(rank), name, True)
            return
        default = f"{stem(a)}-{stem(b)}_r{rank}.safetensors"
        name, ok = QInputDialog.getText(
            self, title,
            "保存ファイル名（models/loras 内）:", text=default)
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        if self._lora_dir is not None and (self._lora_dir / name).exists():
            res = QMessageBox.question(
                self, "上書き確認", f"{name} は既に存在します。上書きしますか？")
            if res != QMessageBox.Yes:
                return
        self.extract_requested.emit(a, b, int(rank), name, False)

    def _on_save(self) -> None:
        entries, quant, low_memory, loras, diffs = self._displayed_recipe()
        if not self._validate(entries):
            return
        name, ok = QInputDialog.getText(
            self, "ファイルとして保存",
            "保存ファイル名（models/diffusion_models 内）:",
            text="merged.safetensors")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        if (self._model_dir / name).exists():
            res = QMessageBox.question(
                self, "上書き確認", f"{name} は既に存在します。上書きしますか？")
            if res != QMessageBox.Yes:
                return
        self.save_requested.emit(entries, loras, diffs, quant, low_memory,
                                 name)

    def _on_duplicate(self) -> None:
        entries, quant, low_memory, loras, diffs = self._displayed_recipe()
        self.lst.setCurrentRow(0)  # switch to 新規作成 (makes pane editable)
        self._load_recipe(entries, quant, low_memory, loras, diffs)

    # ----- progress supplied by the main window ------------------------------
    def set_merge_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        if running:
            # 最初の進捗イベントが届くまではビジー（不定）表示。
            self.progress.setMaximum(0)
            self.progress.setValue(0)

    def set_merge_progress(self, value: int, maximum: int,
                           note: str = "") -> None:
        if not maximum:
            return
        self.progress.setVisible(True)
        self.progress.setMaximum(maximum)
        self.progress.setValue(value)
        self.progress.setFormat(f"{note} {value}/{maximum}".strip())
