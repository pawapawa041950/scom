"""XYZ プロット設定ウィンドウ（非モーダル）。

3軸それぞれに「軸タイプ + 値リスト」を指定して実行をメインウィンドウに依頼
する。マージウィンドウと同じ親なし方式で、メインウィンドウの裏にも回れる。
実行そのものはメインウィンドウが行い、進捗は set_running()/set_progress()
で供給される。
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
)

from .widgets import WideComboBox
from .. import xyz

_AXIS_NAMES = "XYZ"


class XyzDialog(QDialog):
    # spec dict（axes/legend/save_cells/margin）を添えて実行をメインに依頼。
    run_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(self, model_choices: list[str],
                 state: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("XYZ プロット")
        self.setMinimumWidth(720)
        self._models = list(model_choices)
        self._rows: list[dict] = []

        root = QVBoxLayout(self)
        note = QLabel(
            "各軸にパラメータと値リストを指定し、全組み合わせを生成して"
            "1枚のグリッド画像にまとめます（output に保存）。"
            "Seed はメイン画面の値で固定、Batch は 1 として扱われます。")
        note.setWordWrap(True)
        root.addWidget(note)

        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        for i in range(3):
            grid.addWidget(QLabel(f"{_AXIS_NAMES[i]}軸"), i, 0)
            cb = WideComboBox()
            for a in xyz.AXES:
                cb.addItem(a.label, a.id)
                if a.tooltip:
                    cb.setItemData(cb.count() - 1, a.tooltip, Qt.ToolTipRole)
            ed = QLineEdit()
            ed.setToolTip(xyz.VALUE_SYNTAX_HELP)
            btn = QPushButton("候補▾")
            btn.setToolTip("この軸で選べる値を一覧から追記します")
            count = QLabel("")
            count.setMinimumWidth(64)
            grid.addWidget(cb, i, 1)
            grid.addWidget(ed, i, 2)
            grid.addWidget(btn, i, 3)
            grid.addWidget(count, i, 4)
            row = {"combo": cb, "edit": ed, "btn": btn, "count": count}
            self._rows.append(row)
            cb.currentIndexChanged.connect(
                lambda *_a, r=row: self._on_axis_changed(r))
            ed.textChanged.connect(self._update_counts)
            btn.clicked.connect(lambda *_a, r=row: self._show_choices_menu(r))
        root.addLayout(grid)

        hint = QLabel(
            "値の書式: 1, 2, 3 ／ 範囲 1-5 ／ ステップ 1-9 (+2) ／ "
            "分割 0-1 [5] ／ カンマを含む値は \"...\" で囲む")
        hint.setStyleSheet("color: #888;")
        root.addWidget(hint)

        opt_row = QHBoxLayout()
        self.chk_legend = QCheckBox("凡例を描画")
        self.chk_legend.setChecked(True)
        self.chk_save_cells = QCheckBox("各セル画像も個別に保存")
        self.chk_save_cells.setToolTip(
            "グリッド画像に加えて、セルごとの画像もメイン画面の Format 設定で"
            "output に保存します")
        opt_row.addWidget(self.chk_legend)
        opt_row.addWidget(self.chk_save_cells)
        opt_row.addSpacing(16)
        opt_row.addWidget(QLabel("セル余白"))
        self.sp_margin = QSpinBox()
        self.sp_margin.setRange(0, 64)
        self.sp_margin.setSuffix(" px")
        opt_row.addWidget(self.sp_margin)
        opt_row.addStretch(1)
        root.addLayout(opt_row)

        self.lbl_total = QLabel("")
        root.addWidget(self.lbl_total)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        btns = QHBoxLayout()
        # メイン画面の連続と同じ流儀: ONの間、完了するたびに同じ設定で
        # 次の XYZ 生成を自動開始する（実行中でも OFF にできる）。
        self.chk_continuous = QCheckBox("連続")
        self.chk_continuous.setToolTip(
            "ONの間、完了するたびに同じ設定で次の XYZ 生成を自動で開始します"
            "（メイン画面の「生成ごとに seed をランダム化」がONなら毎回"
            "新しい seed になります）")
        self.btn_run = QPushButton("実行")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self.cancel_requested.emit())
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        btns.addStretch(1)
        btns.addWidget(self.chk_continuous)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(btn_close)
        root.addLayout(btns)

        # 既定: X=Steps, Y/Z=なし（保存済み状態があれば復元）
        self._rows[0]["combo"].setCurrentIndex(self._axis_index("steps"))
        if state:
            self._apply_state(state)
        for row in self._rows:
            self._on_axis_changed(row)
        self._update_counts()

    # ----- state -------------------------------------------------------------
    @staticmethod
    def _axis_index(axis_id: str) -> int:
        for i, a in enumerate(xyz.AXES):
            if a.id == axis_id:
                return i
        return 0

    def state(self) -> dict:
        """Persistable dialog state (restored via the constructor)."""
        st = {"legend": self.chk_legend.isChecked(),
              "save_cells": self.chk_save_cells.isChecked(),
              "margin": self.sp_margin.value()}
        for name, row in zip("xyz", self._rows):
            st[f"{name}_type"] = row["combo"].currentData()
            st[f"{name}_values"] = row["edit"].text()
        return st

    def _apply_state(self, st: dict) -> None:
        try:
            for name, row in zip("xyz", self._rows):
                row["combo"].setCurrentIndex(
                    self._axis_index(str(st.get(f"{name}_type", "none"))))
                row["edit"].setText(str(st.get(f"{name}_values", "")))
            self.chk_legend.setChecked(bool(st.get("legend", True)))
            self.chk_save_cells.setChecked(bool(st.get("save_cells", False)))
            self.sp_margin.setValue(int(st.get("margin", 0)))
        except (TypeError, ValueError):
            pass  # 壊れた保存状態は既定値のまま

    # ----- axis rows -----------------------------------------------------------
    def _axis_def(self, row: dict) -> xyz.AxisDef:
        return xyz.axis_by_id(row["combo"].currentData() or "none")

    def _axis_choices(self, axis: xyz.AxisDef) -> list[str]:
        if axis.id == "model":
            return self._models
        return list(axis.choices)

    def _on_axis_changed(self, row: dict) -> None:
        axis = self._axis_def(row)
        row["btn"].setVisible(bool(self._axis_choices(axis)))
        row["edit"].setEnabled(axis.kind != "none")
        tip = axis.tooltip or xyz.VALUE_SYNTAX_HELP
        row["edit"].setToolTip(tip)
        self._update_counts()

    def _show_choices_menu(self, row: dict) -> None:
        axis = self._axis_def(row)
        choices = self._axis_choices(axis)
        if not choices:
            return
        menu = QMenu(self)
        act_all = menu.addAction("すべて追加")
        menu.addSeparator()
        acts = {menu.addAction(c): c for c in choices}
        picked = menu.exec(row["btn"].mapToGlobal(row["btn"].rect().bottomLeft()))
        if picked is None:
            return
        add = choices if picked is act_all else [acts[picked]]
        self._append_values(row["edit"], add)

    @staticmethod
    def _append_values(edit: QLineEdit, values: list[str]) -> None:
        cur = edit.text().strip()
        joined = ", ".join(values)
        if cur:
            sep = "" if cur.endswith(",") else ","
            edit.setText(f"{cur}{sep} {joined}")
        else:
            edit.setText(joined)

    # ----- counts / validation --------------------------------------------------
    def _parsed(self, row: dict) -> list:
        """Values of one axis row. Raises ValueError."""
        axis = self._axis_def(row)
        if axis.kind == "none":
            return [None]
        return xyz.parse_values(axis, row["edit"].text(), self._models)

    def _update_counts(self, *_a) -> None:
        total = 1
        parts = []
        for name, row in zip(_AXIS_NAMES, self._rows):
            axis = self._axis_def(row)
            if axis.kind == "none":
                row["count"].setText("")
                continue
            try:
                n = len(self._parsed(row))
                row["count"].setText(f"{n} 値")
                row["count"].setStyleSheet("")
            except ValueError:
                row["count"].setText("不正")
                row["count"].setStyleSheet("color: red;")
                total = 0
                continue
            total = total * n if total else 0
            parts.append(f"{name}:{n}")
        if total and parts:
            self.lbl_total.setText(
                f"生成枚数: {total} 枚（{' × '.join(parts)}）")
        elif not parts:
            self.lbl_total.setText("軸が1つも指定されていません")
        else:
            self.lbl_total.setText("値の書式にエラーがあります")

    # ----- run -------------------------------------------------------------------
    def _on_run(self) -> None:
        axes = []
        try:
            used = 0
            for row in self._rows:
                axis = self._axis_def(row)
                values = self._parsed(row)
                if axis.kind != "none":
                    used += 1
                axes.append({
                    "id": axis.id,
                    "values": values,
                    "labels": [xyz.value_label(axis, v) for v in values],
                })
            if not used:
                raise ValueError("少なくとも1つの軸を指定してください")
        except ValueError as e:
            QMessageBox.warning(self, "入力エラー", str(e))
            return
        self.run_requested.emit({
            "axes": axes,
            "legend": self.chk_legend.isChecked(),
            "save_cells": self.chk_save_cells.isChecked(),
            "margin": self.sp_margin.value(),
        })

    # ----- progress supplied by the main window ----------------------------------
    def set_running(self, running: bool, total: int = 0) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        for row in self._rows:
            row["combo"].setEnabled(not running)
            row["edit"].setEnabled(
                not running and self._axis_def(row).kind != "none")
            row["btn"].setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setMaximum(total)
            self.progress.setValue(0)
            self.progress.setFormat(f"0/{total}")

    def set_progress(self, done: int) -> None:
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{self.progress.maximum()}")
