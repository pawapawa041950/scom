"""MainWindow のマージモデル関連（マージウィンドウ、マージの実行・上書き・
削除、素材ファイルの確認、差分 LoRA の作成）。main_window.py から分割。"""
from __future__ import annotations

import json
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox
from typing import Optional
from .. import config, modelinfo
from ..comfy_backend import Progress
from ..workflow import (
    GenParams,
    build_extract_lora_graph,
    build_merge_graph,
    entry_ref_ids,
    is_cached_lora,
    merge_pin_key,
    merge_recipe,
    merge_source_file,
    merge_source_files,
    merge_spec_from_entry,
    norm_source,
    replace_merge_spec,
)
from .mw_common import CACHED_LORA_TOKEN, MERGE_TOKEN
from .workers import _GenWorker


class MergeMixin:
    """MainWindow mixin（self は MainWindow）。"""

    # ----- model merge (マージモデル) ---------------------------------------
    def _load_merges(self) -> list[dict]:
        """Load merge entries from settings, migrating the old single-recipe
        keys (merge_models/merge_quant/merge_fp8/merge_low_memory)."""
        try:
            merges = json.loads(str(self.settings.get("merges", "[]")))
            out = []
            for e in merges:
                out.append({
                    "id": int(e["id"]),
                    "name": str(e["name"]),
                    "models": [(norm_source(n), float(w))
                               for n, w in e["models"]],
                    "quant": str(e.get("quant", "")),
                    "low_memory": bool(e.get("low_memory", False)),
                    "loras": [(str(n), float(s))
                              for n, s in e.get("loras", []) or []],
                    "diffs": [(norm_source(a), norm_source(b), float(s))
                              for a, b, s in e.get("diffs", []) or []],
                })
            if out:
                return out
        except (ValueError, TypeError, KeyError):
            pass
        # Migration: a pre-multi-merge settings file with a single recipe.
        try:
            old = [(str(n), float(w)) for n, w in
                   json.loads(str(self.settings.get("merge_models", "[]")))]
        except (ValueError, TypeError):
            old = []
        if len(old) < 2:
            return []
        quant = str(self.settings.get("merge_quant", ""))
        if not quant and self.settings.get("merge_fp8"):
            quant = "fp8"
        return [{"id": 1, "name": "マージモデル1", "models": old,
                 "quant": quant,
                 "low_memory": bool(self.settings.get("merge_low_memory", False)),
                 "loras": [], "diffs": []}]

    def _lora_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family(
            "loras", config.models_root() / "loras" / relname)

    def _merge_family(self, entry: dict) -> str:
        """Family of a merge entry, judged from its first source model
        (same convention as _family_config_warning)."""
        if not entry.get("models"):
            return "unknown"
        return self._diffusion_family(merge_source_file(entry["models"][0][0]))

    def _merge_selected(self) -> bool:
        data = self.cb_diffusion.currentData()
        return isinstance(data, str) and data.startswith(MERGE_TOKEN)

    def _selected_merge_entry(self) -> Optional[dict]:
        data = self.cb_diffusion.currentData()
        if not (isinstance(data, str) and data.startswith(MERGE_TOKEN)):
            return None
        try:
            entry_id = int(data.split(":", 1)[1])
        except (IndexError, ValueError):
            return None
        return self._merge_entry_by_id(entry_id)

    def _merge_missing_refs(self, entry: dict) -> set:
        """Ids of merges this entry uses (at any depth) that no longer exist."""
        ids = {int(e["id"]) for e in self._merges}
        return entry_ref_ids(entry) - ids

    def _merge_broken(self, entry: dict) -> bool:
        """削除済みのマージを（入れ子のどこかで）素材にしている項目は使用
        不可（モデル一覧に出さず、生成・マージもしない）。"""
        return bool(self._merge_missing_refs(entry))

    def _merge_dependents(self, entry_id: int) -> list[dict]:
        """entry_id を（入れ子のどこかで）素材にしている項目。"""
        return [e for e in self._merges if int(e["id"]) != int(entry_id)
                and int(entry_id) in entry_ref_ids(e)]

    def _merge_entry_by_id(self, entry_id: int) -> Optional[dict]:
        for e in self._merges:
            if int(e["id"]) == entry_id:
                return e
        return None

    def _push_merge_state(self) -> None:
        """Refresh the merge window's entry list (if it is open)."""
        if self._merge_dlg is None:
            return
        try:
            self._merge_dlg.set_entries(self._merges, self._merge_built_ids,
                                        self._cached_loras,
                                        self._cached_lora_built)
        except RuntimeError:  # underlying Qt object was deleted
            self._merge_dlg = None

    def _open_merge_dialog(self) -> None:
        from .merge_dialog import MergeDialog
        # Non-modal, at most one instance; re-opening replaces it so the model
        # list is always current.
        if self._merge_dlg is not None:
            try:
                self._merge_dlg.close()
                self._merge_dlg.deleteLater()
            except RuntimeError:
                pass
        # Parentless on purpose: a QDialog with a parent always stays on top
        # of it, but this window should be able to go behind the main window.
        dlg = MergeDialog(
            self._all_models.get("diffusion_models", []),
            self._diffusion_family,
            config.models_root() / "diffusion_models",
            self._all_models.get("loras", []), self._lora_family,
            config.models_root() / "loras", None)
        dlg.merge_requested.connect(self._on_merge_requested)
        dlg.update_requested.connect(self._on_merge_update_requested)
        dlg.lora_update_requested.connect(self._on_cached_lora_update_requested)
        dlg.lora_delete_requested.connect(
            lambda lid: self._on_cached_lora_delete(
                f"{CACHED_LORA_TOKEN}:{int(lid)}"))
        dlg.save_requested.connect(self._on_save_requested)
        dlg.extract_requested.connect(self._on_extract_requested)
        dlg.delete_requested.connect(self._on_merge_delete)
        dlg.rename_requested.connect(self._on_merge_rename)
        dlg.free_memory_requested.connect(self._on_free_memory)
        self._merge_dlg = dlg
        self._push_merge_state()
        if self._merge_running:
            dlg.set_merge_running(True)
        dlg.show()

    def _missing_merge_files(self, models=(), diffs=(), loras=()) -> list[str]:
        """マージに必要なファイル（入れ子の中身・差分行・焼き込む LoRA を
        含む）のうち、models フォルダに無いもの。"""
        root = config.models_root()
        mfiles, lfiles = merge_source_files(models, diffs, loras)
        missing = sorted(f for f in mfiles
                         if not (root / "diffusion_models" / f).is_file())
        missing += sorted("LoRA: " + f for f in lfiles
                          if not (root / "loras" / f).is_file())
        return missing

    def _params_missing_files(self, p: GenParams) -> list[str]:
        """生成に使うマージモデル・差分 LoRA（キャッシュ）の元ファイルの
        うち、見つからないもの。"""
        missing = []
        if p.merge_models:
            missing += self._missing_merge_files(p.merge_models, p.merge_diffs,
                                                 p.merge_loras)
        for n, _w in p.loras:
            if is_cached_lora(n):
                missing += self._missing_merge_files(
                    [(n["model_a"], 1.0), (n["model_b"], 1.0)])
        return sorted(set(missing))

    def _missing_files_error(self, missing: list[str], what: str,
                             parent=None) -> bool:
        """見つからないファイルがあればエラーを出して True（呼び出し側は中止）。"""
        if not missing:
            return False
        msg = (f"{what}に必要なモデルが見つからないため中止しました:\n"
               + "\n".join("・" + m for m in missing))
        self.append_log("エラー: " + msg.replace("\n", " "))
        QMessageBox.critical(parent or self, "モデルが見つかりません", msg)
        return True

    def _missing_ref_warning(self, models, diffs) -> bool:
        """素材に削除済みのマージが含まれていたら警告して True。"""
        missing = self._merge_missing_refs({"models": models, "diffs": diffs})
        if missing:
            QMessageBox.warning(
                self._merge_dlg or self, "削除済みのマージモデル",
                "素材に削除済みのマージモデルが含まれているため実行できません。"
                "該当の行を別のモデルに変更してください。")
            return True
        return False

    def _on_merge_requested(self, entries, loras, diffs, quant: str,
                            low_memory: bool) -> None:
        """マージ button: register a new entry and build it in backend RAM."""
        models = [(norm_source(n), float(w)) for n, w in entries]
        loras = [(str(n), float(s)) for n, s in loras]
        diffs = [(norm_source(a), norm_source(b), float(s))
                 for a, b, s in diffs]
        if self._missing_ref_warning(models, diffs):
            return
        if self._missing_files_error(
                self._missing_merge_files(models, diffs, loras), "マージ",
                self._merge_dlg):
            return
        try:
            graph = build_merge_graph(models, quant, low_memory,
                                      merge_loras=loras, merge_diffs=diffs)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        self._merge_seq += 1
        entry = {"id": self._merge_seq, "name": f"マージモデル{self._merge_seq}",
                 "models": models, "quant": str(quant),
                 "low_memory": bool(low_memory), "loras": loras,
                 "diffs": diffs}
        self._merges.append(entry)
        self._schedule_save()
        self._apply_preset_filter()  # the new entry appears in the dropdown
        self._push_merge_state()
        if self._merge_dlg is not None:
            self._merge_dlg.select_entry(entry["id"])
        self._run_merge(graph, saving=False, entry_id=entry["id"])

    def _on_save_requested(self, entries, loras, diffs, quant: str,
                           low_memory: bool, filename: str) -> None:
        models = [(norm_source(n), float(w)) for n, w in entries]
        loras = [(str(n), float(s)) for n, s in loras]
        diffs = [(norm_source(a), norm_source(b), float(s))
                 for a, b, s in diffs]
        if self._missing_ref_warning(models, diffs):
            return
        if self._missing_files_error(
                self._missing_merge_files(models, diffs, loras), "マージ",
                self._merge_dlg):
            return
        try:
            graph = build_merge_graph(models, quant, low_memory,
                                      save_to=filename, merge_loras=loras,
                                      merge_diffs=diffs)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        self._run_merge(graph, saving=True, entry_id=None)

    def _release_entry_pin(self, entry: dict) -> None:
        """この項目の（現在の設定の）マージ結果をバックエンドから解放する。"""
        self._merge_built_ids.discard(int(entry["id"]))
        if not self.backend.is_running():
            return
        try:
            self.backend.release_merge(
                merge_recipe(entry["models"], entry.get("loras", []),
                             entry.get("diffs", [])),
                entry["quant"], entry["low_memory"])
        except OSError as e:
            self.append_log(f"メモリ解放に失敗: {e}")

    def _on_merge_update_requested(self, entry_id: int, entries, loras,
                                   diffs, quant: str,
                                   low_memory: bool) -> None:
        """作成済みマージの設定を上書きして作り直す。古い設定のキャッシュは
        解放し、この項目を素材にしている他の項目の中身（スナップショット）
        も新しい設定に差し替える（そちらの古いキャッシュも解放）。"""
        entry = self._merge_entry_by_id(int(entry_id))
        if entry is None:
            return
        models = [(norm_source(n), float(w)) for n, w in entries]
        loras = [(str(n), float(s)) for n, s in loras]
        diffs = [(norm_source(a), norm_source(b), float(s))
                 for a, b, s in diffs]
        if self._missing_ref_warning(models, diffs):
            return
        if self._missing_files_error(
                self._missing_merge_files(models, diffs, loras), "マージ",
                self._merge_dlg):
            return
        dependents = self._merge_dependents(entry["id"])
        refs = entry_ref_ids({"models": models, "diffs": diffs})
        if int(entry["id"]) in refs or refs & {int(d["id"]) for d in dependents}:
            QMessageBox.warning(
                self._merge_dlg or self, "循環参照",
                "この項目自身、またはこの項目を素材にしているマージモデルは"
                "素材に指定できません。")
            return
        try:
            graph = build_merge_graph(models, quant, low_memory,
                                      merge_loras=loras, merge_diffs=diffs)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        for e in [entry] + dependents:
            self._release_entry_pin(e)
        entry.update({"models": models, "quant": str(quant),
                      "low_memory": bool(low_memory), "loras": loras,
                      "diffs": diffs})
        new_spec = merge_spec_from_entry(entry)
        for d in dependents:
            d["models"] = [(replace_merge_spec(n, entry["id"], new_spec), w)
                           for n, w in d["models"]]
            d["diffs"] = [(replace_merge_spec(a, entry["id"], new_spec),
                           replace_merge_spec(b, entry["id"], new_spec), s)
                          for a, b, s in d.get("diffs", [])]
        self._schedule_save()
        self._apply_preset_filter()
        self._push_merge_state()
        if self._merge_dlg is not None:
            self._merge_dlg.select_entry(entry["id"])
        msg = f"{entry['name']} の設定を更新しました（古いキャッシュは破棄）"
        if dependents:
            msg += ("。素材にしている " + "、".join(d["name"] for d in dependents)
                    + " も新しい設定で作り直されます")
        self.append_log(msg)
        self._run_merge(graph, saving=False, entry_id=entry["id"])

    def _on_extract_requested(self, model_a, model_b, rank: int,
                              name: str, to_cache: bool) -> None:
        """「差分から LoRA を作成」: model_a − model_b を低ランク近似する
        （バックエンドの ScomExtractLora）。to_cache=True はバックエンドの RAM
        に置いて LoRA ウィンドウに「キャッシュ」として並べ、False は
        models/loras/<name> に保存。model_a / model_b はファイル名か入れ子
        マージの指定（dict）。"""
        a, b = norm_source(model_a), norm_source(model_b)
        if self._missing_files_error(
                self._missing_merge_files([(a, 1.0), (b, 1.0)]),
                "差分 LoRA の作成", self._merge_dlg):
            return
        try:
            graph = build_extract_lora_graph(a, b, int(rank),
                                             "" if to_cache else name,
                                             to_cache=to_cache)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        if to_cache:
            # 同じ (モデル1, モデル2, ランク) の項目は作り直さず名前だけ更新。
            entry = next((e for e in self._cached_loras
                          if e["model_a"] == a and e["model_b"] == b
                          and int(e["rank"]) == int(rank)), None)
            if entry is None:
                self._cached_lora_seq += 1
                entry = {"id": self._cached_lora_seq, "name": name,
                         "model_a": a, "model_b": b, "rank": int(rank)}
                self._cached_loras.append(entry)
            else:
                entry["name"] = name
            self._schedule_save()
            self._refresh_lora_dialog()
            # マージモデルと同じく、マージウィンドウの一覧に項目として残す。
            self._push_merge_state()
            if self._merge_dlg is not None:
                self._merge_dlg.select_lora(entry["id"])
            self._run_merge(graph, saving=False, entry_id=int(entry["id"]),
                            kind="extract_cache")
            return
        self._run_merge(graph, saving=True, entry_id=None, kind="extract")

    def _on_merge_delete(self, entry_id: int) -> None:
        entry = self._merge_entry_by_id(entry_id)
        self._merges = [e for e in self._merges
                        if int(e["id"]) != entry_id]
        self._merge_built_ids.discard(entry_id)
        # Free its pinned model from backend RAM (per-entry, via our node's
        # management route).
        if entry is not None and self.backend.is_running():
            try:
                self.backend.release_merge(
                    merge_recipe(entry["models"], entry.get("loras", []),
                                 entry.get("diffs", [])),
                    entry["quant"], entry["low_memory"])
            except OSError as e:
                self.append_log(f"メモリ解放に失敗: {e}")
        was_selected = (self.cb_diffusion.currentData()
                        == f"{MERGE_TOKEN}:{entry_id}")
        self._apply_preset_filter()
        if was_selected:
            # Fall back to the first regular model (after the merge entries
            # still shown under the current preset).
            first_regular = 0
            for i in range(self.cb_diffusion.count()):
                data = self.cb_diffusion.itemData(i)
                if not (isinstance(data, str) and data.startswith(MERGE_TOKEN)):
                    first_regular = i
                    break
            self.cb_diffusion.setCurrentIndex(
                min(first_regular, self.cb_diffusion.count() - 1))
        self._schedule_save()
        self._push_merge_state()
        name = entry["name"] if entry else f"id={entry_id}"
        self.append_log(f"{name} を一覧から削除し、メモリを解放しました")
        broken = [e["name"] for e in self._merges if self._merge_broken(e)]
        if broken:
            self.append_log(
                "削除したマージモデルを素材にしている " + "、".join(broken)
                + " は使用できなくなりました（マージウィンドウで素材を変更"
                "すれば再び使えます）")

    def _on_merge_rename(self, entry_id: int, name: str) -> None:
        e = self._merge_entry_by_id(entry_id)
        if e is None or not name.strip():
            return
        e["name"] = name.strip()
        self._apply_preset_filter()  # dropdown labels follow the new name
        self._schedule_save()
        self._push_merge_state()

    def _on_free_memory(self) -> None:
        if not self._backend_ready():
            QMessageBox.warning(self, "未準備", "バックエンドが起動していません。")
            return
        try:
            self.backend.release_all_merges()
            self.backend.free_memory()
        except OSError as e:
            self.append_log(f"メモリ解放に失敗: {e}")
            return
        self._merge_built_ids.clear()
        self._cached_lora_built.clear()
        self._push_merge_state()
        self.append_log("バックエンドのキャッシュとモデル（マージモデル・"
                        "差分 LoRA を含む）をすべて解放しました"
                        "（次回使用時に自動で再構築されます）")

    def _sync_merge_states(self) -> None:
        """Refresh ●/○ from the backend's pin cache (the source of truth)."""
        if not self.backend.is_running():
            return
        try:
            pinned = set(self.backend.merge_pinned())
        except OSError:
            return
        self._merge_built_ids = {
            int(e["id"]) for e in self._merges
            if merge_pin_key(e["models"], e["quant"], e["low_memory"],
                             e.get("loras", []), e.get("diffs", []))
            in pinned}
        try:
            cached = set(self.backend.cached_loras())
            self._cached_lora_built = {
                int(e["id"]) for e in self._cached_loras
                if self._cached_lora_key(e) in cached}
        except OSError:
            pass
        self._push_merge_state()

    def _run_merge(self, graph: dict, saving: bool,
                   entry_id: Optional[int], kind: str = "merge") -> None:
        """Run a merge-only prompt (build in RAM / save to file / extract a
        LoRA: ``kind`` = "merge" | "extract") off-thread."""
        if not self._backend_ready():
            QMessageBox.warning(
                self, "未準備",
                "バックエンドがまだ起動していません。起動完了後、"
                "生成時に自動でマージされます。")
            return
        if self._gen_thread is not None or self._xyz_thread is not None:
            QMessageBox.information(
                self, "実行中",
                "生成の完了後に実行してください（生成時に自動でマージされます）。")
            return
        self._last_gen_ok = False  # keep continuous mode from chaining a merge
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        if kind in ("extract", "extract_cache"):
            note = "2 モデルの差分から LoRA を抽出中…"
        else:
            note = ("マージモデルを保存中…" if saving
                    else "マージモデルをメインメモリ上に構築中…")
        self.status.showMessage(note)
        self.append_log(note)

        self._gen_thread = QThread(self)
        self._gen_worker = _GenWorker(self.backend, graph)
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.progress.connect(self._on_progress)
        # NOTE: must be a bound method, not a lambda. Signals connected to a
        # plain callable run in the EMITTING (worker) thread; only QObject
        # bound methods get queued to the GUI thread. A lambda here executed
        # _on_merge_done -> dialog widget updates off the GUI thread
        # ("Cannot set parent ... different thread" warnings).
        self._merge_run_ctx = (saving, entry_id, kind)
        self._merge_running = True
        self._gen_worker.progress.connect(self._on_merge_progress)
        self._gen_worker.done.connect(self._on_merge_worker_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._end_merge_progress)
        self._gen_worker.failed.connect(self._end_merge_progress)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()
        if self._merge_dlg is not None:
            try:
                self._merge_dlg.set_merge_running(True)
            except RuntimeError:
                self._merge_dlg = None

    def _on_merge_worker_done(self, _images: list) -> None:
        saving, entry_id, kind = getattr(self, "_merge_run_ctx",
                                         (False, None, "merge"))
        if kind == "extract_cache":
            if entry_id is not None:
                self._cached_lora_built.add(int(entry_id))
                self._push_merge_state()
            self.status.showMessage("差分 LoRA をキャッシュに配置しました")
            self.append_log(
                "2 モデルの差分から LoRA を抽出し、バックエンドのキャッシュ"
                "（メインメモリ）に配置しました。LoRA ウィンドウに「キャッシュ」"
                "として表示されます（キャッシュから消えても使用時に自動で"
                "再作成されます）")
            self._refresh_lora_dialog()
            return
        if kind == "extract":
            self.status.showMessage("差分 LoRA を保存しました")
            self.append_log(
                "2 モデルの差分から LoRA を抽出し、models/loras に保存しました"
                "（捉えた差分の割合はバックエンドのログ「scom extract」を参照）")
            self.refresh_models()  # the new LoRA appears in the LoRA window
            self._refresh_lora_dialog()
            return
        self._on_merge_done(saving, entry_id)

    # NOTE: どちらも bound method で接続すること（_run_merge の NOTE 参照）。
    def _on_merge_progress(self, p: Progress) -> None:
        """Mirror merge progress onto the merge window's progress bar."""
        if self._merge_dlg is None:
            return
        try:
            self._merge_dlg.set_merge_progress(p.value, p.maximum, p.note)
        except RuntimeError:
            self._merge_dlg = None

    def _end_merge_progress(self, *_a) -> None:
        self._merge_running = False
        if self._merge_dlg is None:
            return
        try:
            self._merge_dlg.set_merge_running(False)
        except RuntimeError:
            self._merge_dlg = None

    def _on_merge_done(self, saved: bool, entry_id: Optional[int]) -> None:
        if entry_id is not None:
            self._merge_built_ids.add(entry_id)
        self._sync_merge_states()
        if saved:
            self.status.showMessage("マージモデルを保存しました")
            self.append_log("マージモデルを models/diffusion_models に保存しました")
            self.refresh_models()  # the new file appears in the dropdowns
        else:
            e = self._merge_entry_by_id(entry_id) if entry_id else None
            name = e["name"] if e else "マージモデル"
            self.status.showMessage(f"{name} を構築しました")
            self.append_log(
                f"{name} をメインメモリ上に構築しました"
                "（同じ構成での生成はこのモデルを再利用します）")
            # Building it means the user intends to generate with it.
            idx = self.cb_diffusion.findData(f"{MERGE_TOKEN}:{entry_id}")
            if idx >= 0:
                self.cb_diffusion.setCurrentIndex(idx)
        self._push_merge_state()
