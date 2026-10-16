"""MainWindow の LoRA 関連（LoRA ウィンドウ、適用中 LoRA の表示、トリガー
ワードの挿入、<lora:…> の取り込み、キャッシュ上の差分 LoRA）。
main_window.py から分割。"""
from __future__ import annotations

import json
from PySide6.QtCore import QPoint
from PySide6.QtGui import QCursor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
)
from pathlib import Path
from typing import Optional
from .. import config, lora as lora_meta
from ..workflow import (
    build_extract_lora_graph,
    cached_lora_spec,
    merge_source_file,
    merge_source_label,
    norm_source,
    source_input,
)
from .mw_common import CACHED_LORA_TOKEN, _LORA_INSERT_BG, _LORA_INSERT_FG, _LORA_TAG_RE, _LORA_TOKEN_PROP


class LoraMixin:
    """MainWindow mixin（self は MainWindow）。"""

    # ----- cached difference LoRAs ------------------------------------------
    def _load_cached_loras(self) -> list[dict]:
        try:
            raw = json.loads(str(self.settings.get("cached_loras", "[]")))
            return [{"id": int(e["id"]), "name": str(e["name"]),
                     "model_a": norm_source(e["model_a"]),
                     "model_b": norm_source(e["model_b"]),
                     "rank": int(e.get("rank", 64))} for e in raw]
        except (ValueError, TypeError, KeyError):
            return []

    def _cached_lora_by_token(self, token) -> Optional[dict]:
        if not (isinstance(token, str)
                and token.startswith(CACHED_LORA_TOKEN + ":")):
            return None
        try:
            entry_id = int(token.split(":", 1)[1])
        except ValueError:
            return None
        return next((e for e in self._cached_loras
                     if int(e["id"]) == entry_id), None)

    def _lora_display(self, name: str) -> str:
        """Short label of an applied LoRA (file stem / cache entry name)."""
        e = self._cached_lora_by_token(name)
        if e is not None:
            return f"[キャッシュ] {e['name']}"
        if isinstance(name, str) and name.startswith(CACHED_LORA_TOKEN):
            return "[キャッシュ] (削除済み)"
        return Path(name).stem

    def _cached_lora_items(self) -> list[dict]:
        """LoRA ウィンドウに並べる差分 LoRA（系統はモデル1の先頭モデル）。"""
        out = []
        for e in self._cached_loras:
            out.append({
                "token": f"{CACHED_LORA_TOKEN}:{e['id']}",
                "name": e["name"],
                "family": self._diffusion_family(
                    merge_source_file(e["model_a"])),
                "desc": (f"差分: {merge_source_label(e['model_a'])} − "
                         f"{merge_source_label(e['model_b'])}／ランク {e['rank']}"),
            })
        return out

    def _lora_param(self, name: str):
        """GenParams.loras の name: ファイル名、または差分 LoRA の spec。"""
        e = self._cached_lora_by_token(name)
        if e is not None:
            return cached_lora_spec(e)
        if isinstance(name, str) and name.startswith(CACHED_LORA_TOKEN):
            raise ValueError("適用中の差分 LoRA がキャッシュ一覧から"
                             "削除されています。LoRA を解除してください。")
        return name

    @staticmethod
    def _cached_lora_key(e: dict) -> str:
        """バックエンドの差分 LoRA キャッシュのキー（ノードの _lora_key と同じ）。"""
        return json.dumps([norm_source(e["model_a"]), norm_source(e["model_b"]),
                           int(e["rank"])], sort_keys=True, ensure_ascii=False)

    def _on_cached_lora_update_requested(self, lora_id: int, model_a, model_b,
                                         rank: int, name: str) -> None:
        """マージウィンドウで選んだ差分 LoRA 項目を新しい設定で作り直す。
        古い設定のキャッシュは解放する（適用中の LoRA は新しい設定で効く）。"""
        e = next((x for x in self._cached_loras
                  if int(x["id"]) == int(lora_id)), None)
        if e is None:
            return
        a, b = norm_source(model_a), norm_source(model_b)
        if self._missing_files_error(
                self._missing_merge_files([(a, 1.0), (b, 1.0)]),
                "差分 LoRA の作成", self._merge_dlg):
            return
        try:
            graph = build_extract_lora_graph(a, b, int(rank), to_cache=True)
        except ValueError as ex:
            QMessageBox.warning(self, "入力不足", str(ex))
            return
        if self.backend.is_running():
            try:
                self.backend.release_cached_lora(
                    source_input(e["model_a"]), source_input(e["model_b"]),
                    int(e["rank"]))
            except OSError as ex:
                self.append_log(f"メモリ解放に失敗: {ex}")
        self._cached_lora_built.discard(int(e["id"]))
        e.update({"name": name, "model_a": a, "model_b": b, "rank": int(rank)})
        self.append_log(f"差分 LoRA「{name}」の設定を更新しました（古いキャッシュは破棄）")
        self._schedule_save()
        self._refresh_lora_dialog()
        self._rebuild_lora_rows()
        self._push_merge_state()
        self._run_merge(graph, saving=False, entry_id=int(e["id"]),
                        kind="extract_cache")

    def _on_cached_lora_delete(self, token: str) -> None:
        e = self._cached_lora_by_token(token)
        if e is None:
            return
        self._cached_lora_built.discard(int(e["id"]))
        self._cached_loras = [x for x in self._cached_loras
                              if int(x["id"]) != int(e["id"])]
        if any(x["name"] == token for x in self._loras):
            self._loras = [x for x in self._loras if x["name"] != token]
            self._rebuild_lora_rows()
        if self.backend.is_running():
            try:
                self.backend.release_cached_lora(
                    source_input(e["model_a"]), source_input(e["model_b"]),
                    int(e["rank"]))
            except OSError as ex:
                self.append_log(f"メモリ解放に失敗: {ex}")
        self.append_log(f"差分 LoRA「{e['name']}」をキャッシュから削除しました")
        self._schedule_save()
        self._refresh_lora_dialog()
        self._push_merge_state()

    def _refresh_lora_dialog(self) -> None:
        if self._lora_dlg is None:
            return
        try:
            self._lora_dlg.rescan()
            self._push_lora_state()
        except RuntimeError:
            self._lora_dlg = None

    # ----- LoRA -------------------------------------------------------------
    def _open_lora_dialog(self) -> None:
        from .lora_dialog import LoraDialog
        # Non-modal, at most one instance; re-opening replaces it so the file
        # list is always current (parentless so it can go behind us).
        if self._lora_dlg is not None:
            try:
                self._lora_dlg.close()
                self._lora_dlg.deleteLater()
            except RuntimeError:
                pass
        dlg = LoraDialog(config.models_root() / "loras",
                         self.paths.user_data / "lora_cache",
                         self._current_preset, None,
                         cached_fn=self._cached_lora_items)
        dlg.cached_delete_requested.connect(self._on_cached_lora_delete)
        dlg.apply_requested.connect(self._on_lora_apply)
        dlg.remove_requested.connect(self._on_lora_remove)
        dlg.toggle_prompt_requested.connect(self._on_lora_toggle_prompt)
        self._lora_dlg = dlg
        self._push_lora_state()
        self._push_lora_inserted()
        dlg.show()

    def _push_lora_state(self) -> None:
        """Refresh the LoRA window's applied marks (if it is open)."""
        if self._lora_dlg is None:
            return
        try:
            self._lora_dlg.set_applied(
                {e["name"]: float(e["strength"]) for e in self._loras})
        except RuntimeError:
            self._lora_dlg = None

    def _on_lora_apply(self, name: str, strength: float) -> None:
        for e in self._loras:
            if e["name"] == name:
                e["strength"] = float(strength)
                break
        else:
            self._loras.append({"name": name, "strength": float(strength)})
            self.append_log(
                f"LoRA を適用: {self._lora_display(name)} ×{strength:g}")
        self._rebuild_lora_rows()
        self._push_lora_state()
        self._schedule_save()

    def _on_lora_remove(self, name: str) -> None:
        before = len(self._loras)
        self._loras = [e for e in self._loras if e["name"] != name]
        if len(self._loras) != before:
            self.append_log(f"LoRA を解除: {self._lora_display(name)}")
        self._rebuild_lora_rows()
        self._push_lora_state()
        self._schedule_save()

    def _on_lora_strength_changed(self, name: str, value: float) -> None:
        for e in self._loras:
            if e["name"] == name:
                e["strength"] = float(value)
        self._push_lora_state()
        self._schedule_save()

    def _on_lora_toggle_prompt(self, token: str, negative: bool,
                               text: str) -> None:
        """LoRA のトリガーワードをトグルする。未挿入なら挿入（区別マーク付き）、
        挿入済みなら（ユーザー編集後でも）その区間ごと削除する。"""
        field = self.txt_negative if negative else self.txt_prompt
        regions = self._find_token_regions(field, token)
        if regions:
            self._remove_token_regions(field, regions)
        else:
            self._insert_token_words(field, token, text)
        self._push_lora_inserted()

    @staticmethod
    def _plain_char_format() -> QTextCharFormat:
        """token を持たない通常書式（区切りや以降の入力がハイライトされない
        ようにするため）。"""
        fmt = QTextCharFormat()
        fmt.clearBackground()
        return fmt

    def _token_char_format(self, token: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setBackground(_LORA_INSERT_BG)
        fmt.setForeground(_LORA_INSERT_FG)
        fmt.setProperty(_LORA_TOKEN_PROP, token)
        return fmt

    @staticmethod
    def _find_token_regions(field, token: str) -> list[tuple[int, int]]:
        """指定 token の文字書式を持つ連続区間 (start, end) を左から順に返す。
        内部編集でフラグメントが分割されていても隣接分は1区間に統合する。"""
        doc = field.document()
        frags: list[tuple[int, int]] = []
        block = doc.begin()
        while block != doc.end():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and \
                        frag.charFormat().property(_LORA_TOKEN_PROP) == token:
                    start = frag.position()
                    frags.append((start, start + frag.length()))
                it += 1
            block = block.next()
        frags.sort()
        regions: list[tuple[int, int]] = []
        for s, e in frags:
            if regions and s <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(regions[-1][1], e))
            else:
                regions.append((s, e))
        return regions

    def _remove_token_regions(self, field,
                              regions: list[tuple[int, int]]) -> None:
        """token 区間を削除する。挿入時に付けた直後の色なしスペースと、
        隣接する区切り ", " も1つ巻き込んで取り除き、", ," や余分な空白が
        残らないようにする。位置ズレを避けるため右端の区間から削除する。"""
        text = field.toPlainText()
        cursor = field.textCursor()
        for start, end in sorted(regions, reverse=True):
            s, e = start, end
            lead = text[s - 2:s] == ", "
            # 挿入時の色なし後続スペースを巻き込む。ただし前側の区切りも
            # 取る場合、スペースの先にユーザーの追記があるなら残す
            # （両方消すと前後のテキストが癒着するため）。
            if text[e:e + 1] == " " and (not lead or not text[e + 1:].strip()):
                e += 1
            if lead:                        # 直前の区切りを巻き込む
                s -= 2
            elif text[e:e + 2] == ", ":     # 先頭要素なら直後の区切りを
                e += 2
            cursor.setPosition(s)
            cursor.setPosition(e, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        field.setCurrentCharFormat(self._plain_char_format())

    def _insert_token_words(self, field, token: str, text: str) -> None:
        """欄末尾に、区別マーク付きでワードを追記する。

        色付き区間の直後に入力すると Qt は左隣の書式（=色）を引き継ぐため、
        区間の直後に色なしスペースを1つ置く。前側は色なしの ", " 区切りが
        同じ役割を果たす。これで続けて追記しても色は付かない。
        """
        words = [w.strip() for w in text.split(",") if w.strip()]
        if not words:
            return
        joined = ", ".join(words)
        full = field.toPlainText()
        stripped = full.rstrip()
        cursor = field.textCursor()
        # 末尾の余分な空白を除いてから追記する。
        cursor.setPosition(len(stripped))
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        if stripped:
            cursor.insertText(", ", self._plain_char_format())
        cursor.insertText(joined, self._token_char_format(token))
        cursor.insertText(" ", self._plain_char_format())
        field.setTextCursor(cursor)
        # 続けて入力してもハイライトされないよう書式を通常へ戻す。
        field.setCurrentCharFormat(self._plain_char_format())

    def _active_lora_tokens(self) -> set[str]:
        """両プロンプト欄に現在挿入されている LoRA トリガーワードの token 集合。"""
        tokens: set[str] = set()
        for field in (self.txt_prompt, self.txt_negative):
            doc = field.document()
            block = doc.begin()
            while block != doc.end():
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if frag.isValid():
                        tok = frag.charFormat().property(_LORA_TOKEN_PROP)
                        if tok:
                            tokens.add(str(tok))
                    it += 1
                block = block.next()
        return tokens

    def _resolve_lora_name(self, name: str) -> Optional[str]:
        """<lora:名前> の名前を models/loras の相対パスに解決する。拡張子や
        サブフォルダの有無、大文字小文字、区切り文字の向きは問わない。"""
        want = name.strip().replace("/", "\\").lower()
        if not want:
            return None
        for rel in self._all_models.get("loras", []):
            p = Path(rel)
            if want in (rel.lower(), str(p.with_suffix("")).lower(),
                        p.name.lower(), p.stem.lower()):
                return rel
        return None

    def _absorb_lora_tags(self, field) -> None:
        """欄内の <lora:名前:強度> を LoRA 設定に取り込み、文字列を取り除く
        （WebUI の記法をそのまま貼れるように）。該当ファイルが無いものは
        そのまま残し、一度だけログに出す。"""
        if getattr(self, "_loading", False) or self._absorbing_lora_tags:
            return
        text = field.toPlainText()
        if "<lora:" not in text.lower():
            return
        regions: list[tuple[int, int]] = []
        for m in _LORA_TAG_RE.finditer(text):
            name = m.group(1).strip()
            rel = self._resolve_lora_name(name)
            if rel is None:
                if name not in self._lora_tag_warned:
                    self._lora_tag_warned.add(name)
                    self.append_log(
                        f"<lora:{name}> に該当する LoRA が models/loras に"
                        "見つかりません（文字列はそのまま残します）")
                continue
            try:
                strength = float(m.group(2)) if m.group(2) else 1.0
            except ValueError:
                strength = 1.0
            self.append_log(
                f"プロンプトの <lora:{name}> を LoRA 設定に取り込みます")
            self._on_lora_apply(rel, strength)
            regions.append((m.start(), m.end()))
        if not regions:
            return
        self._absorbing_lora_tags = True
        try:
            self._remove_token_regions(field, regions)
        finally:
            self._absorbing_lora_tags = False

    def _push_lora_inserted(self) -> None:
        """挿入済み token を LoRA ウィンドウ・カードのポップアップへ通知
        （リンクの挿入済みハイライト更新用）。"""
        tokens = self._active_lora_tokens()
        if self._lora_dlg is not None:
            try:
                self._lora_dlg.set_inserted(tokens)
            except RuntimeError:
                self._lora_dlg = None
        if self._lora_popup is not None and self._lora_popup.isVisible():
            self._lora_popup.set_active(tokens)

    # ----- LoRA card hover popup -------------------------------------------
    def _ensure_lora_popup(self):
        if self._lora_popup is None:
            from .lora_dialog import TriggerWordsPopup
            self._lora_popup = TriggerWordsPopup(self)
            self._lora_popup.toggle_requested.connect(self._on_lora_toggle_prompt)
            self._lora_popup.hover_changed.connect(self._on_lora_popup_hover)
        return self._lora_popup

    def _show_lora_popup(self, relname: str, anchor) -> None:
        # 差分 LoRA のトリガーワードも LoRA ウィンドウで編集でき、token を
        # キーに保存される（ファイルが無いので civitai の初期値は無い）。
        pos, neg = lora_meta.effective_trigger_words(
            relname, config.models_root() / "loras",
            self.paths.user_data / "lora_cache")
        popup = self._ensure_lora_popup()
        popup.set_content(relname, pos, neg, self._active_lora_tokens())
        if not popup.has_words():        # トリガーワードが無ければ出さない
            popup.hide()
            return
        self._lora_pop_timer.stop()
        self._lora_pop_anchor = anchor
        # サイズは set_content で確定済み。画面内に収まる位置へクランプする
        # （はみ出すと OS 側で再配置され、setGeometry 警告の原因になる）。
        below = anchor.mapToGlobal(QPoint(0, anchor.height() + 2))
        screen = (anchor.screen() or self.screen()).availableGeometry()
        y = below.y()
        if y + popup.height() - 1 > screen.bottom():
            # 下にはみ出すならアンカーの上側へ回す。
            y = anchor.mapToGlobal(QPoint(0, 0)).y() - popup.height() - 2
        # 最終的に画面内へ確実に収める（上下・左右とも）。
        x = max(screen.left(),
                min(below.x(), screen.right() - popup.width() + 1))
        y = max(screen.top(), min(y, screen.bottom() - popup.height() + 1))
        popup.move(x, y)
        popup.show()
        popup.raise_()

    def _on_lora_popup_hover(self, over: bool) -> None:
        # ポップアップ上にマウスがある間は閉じない。離れたら猶予後に閉じる。
        if over:
            self._lora_pop_timer.stop()
        else:
            self._lora_pop_timer.start()

    def _hide_lora_popup(self, force: bool = False) -> None:
        """カーソルがまだカード or ポップアップ上にあれば閉じない（チップの
        子ウィジェット＝強度スピン等の上でも Leave が飛ぶため、実際の位置で
        判定する）。force=True は無条件で閉じる（チップ再構築時など）。"""
        if self._lora_popup is None or not self._lora_popup.isVisible():
            return
        if not force:
            gp = QCursor.pos()
            for w in (self._lora_popup, getattr(self, "_lora_pop_anchor", None)):
                try:
                    if (w is not None and w.isVisible()
                            and w.rect().contains(w.mapFromGlobal(gp))):
                        self._lora_pop_timer.start()   # まだ上にある → 保持
                        return
                except RuntimeError:
                    pass                               # チップが破棄済み
        self._lora_popup.hide()
        self._lora_pop_anchor = None

    def _rebuild_lora_rows(self) -> None:
        """Rebuild the applied-LoRA chips after the LoRA button (flow layout;
        index 0 is the button itself)."""
        self._hide_lora_popup(force=True)
        while self._lora_flow.count() > 1:
            item = self._lora_flow.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for e in self._loras:
            name = e["name"]
            chip = QFrame()
            chip.setObjectName("loraChip")
            chip.setStyleSheet(
                "#loraChip { border: 1px solid #666; border-radius: 4px; }")
            # ホバーでトリガーワードのポップアップを出す（Enter/Leave を監視）。
            # _lora_chip は常にポップアップの位置基準（子から入っても同じ場所）。
            chip._lora_name = name
            chip._lora_chip = chip
            chip.installEventFilter(self)
            h = QHBoxLayout(chip)
            h.setContentsMargins(6, 1, 4, 1)
            h.setSpacing(4)
            trash = QPushButton("🗑")
            trash.setFixedWidth(24)
            trash.setFlat(True)
            trash.setToolTip("この LoRA を解除")
            trash.clicked.connect(
                lambda *_a, n=name: self._on_lora_remove(n))
            lbl = QLabel(self._lora_display(name))
            cached = self._cached_lora_by_token(name)
            lbl.setToolTip(
                name if cached is None else
                f"差分 LoRA（キャッシュ）: {merge_source_label(cached['model_a'])}"
                f" − {merge_source_label(cached['model_b'])}／"
                f"ランク {cached['rank']}")
            spin = QDoubleSpinBox()
            spin.setRange(-4.0, 4.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(float(e["strength"]))
            spin.setFixedWidth(64)
            spin.setToolTip("LoRA の適用強度（model / clip 共通）")
            spin.valueChanged.connect(
                lambda v, n=name: self._on_lora_strength_changed(n, v))
            h.addWidget(trash)
            h.addWidget(lbl)
            h.addWidget(spin)
            # 子ウィジェットに直接カーソルが入ってもポップアップが出るように
            # 同じ監視をぶら下げる（Enter はカーソル直下のウィジェットに届く）。
            for child in (trash, lbl, spin):
                child._lora_name = name
                child._lora_chip = chip
                child.installEventFilter(self)
            self._lora_flow.addWidget(chip)
