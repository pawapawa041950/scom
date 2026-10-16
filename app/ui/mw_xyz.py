"""MainWindow の XYZ プロット関連（候補・実行・待機タスク・セルの系統整列・
グリッド合成）。main_window.py から分割。"""
from __future__ import annotations

import json
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QMessageBox
from pathlib import Path
from typing import Optional
from .. import config, families, metadata, modelinfo, xyz
from ..workflow import GenParams, build_graph, cached_lora_spec, norm_source
from .mw_common import MERGE_PREFIX, NO_SAVE, _XYZ_AXIS_FIELDS
from .workers import _XyzComposeWorker, _XyzWorker


class XyzMixin:
    """MainWindow mixin（self は MainWindow）。"""

    # ----- XYZ plot ---------------------------------------------------------
    def _open_xyz_dialog(self) -> None:
        from .xyz_dialog import XyzDialog
        # 実行中は既存ウィンドウを前面に出すだけ（作り直すと進捗表示が切れる）。
        if self._xyz_thread is not None and self._xyz_dlg is not None:
            try:
                self._xyz_dlg.show()
                self._xyz_dlg.raise_()
                self._xyz_dlg.activateWindow()
                return
            except RuntimeError:
                self._xyz_dlg = None
        if self._xyz_dlg is not None:
            try:
                self._xyz_dlg.close()
                self._xyz_dlg.deleteLater()
            except RuntimeError:
                pass
        try:
            state = json.loads(str(self.settings.get("xyz", "{}")))
        except (ValueError, TypeError):
            state = {}
        # Parentless on purpose: same as the merge window, so it can go
        # behind the main window. Model-axis choices list merge entries
        # first (as "マージモデル：<name>" tokens), like the main dropdown.
        dlg = XyzDialog(self._xyz_model_choices(), state, None,
                        rescan_fn=self._rescan_xyz_model_choices,
                        lora_choices_fn=self._xyz_lora_choices)
        dlg.run_requested.connect(self._on_xyz_requested)
        dlg.cancel_requested.connect(self._on_xyz_cancel_requested)
        # チェック状態は実行しなくても記憶する（次回開いたとき再現）。
        # コンストラクタでの復元後に接続するので、復元自体では発火しない。
        dlg.chk_save_cells.toggled.connect(
            lambda c: self._persist_xyz_flag("save_cells", c))
        dlg.chk_show_grid.toggled.connect(
            lambda c: self._persist_xyz_flag("show_grid", c))
        dlg.chk_save_grid.toggled.connect(
            lambda c: self._persist_xyz_flag("save_grid", c))
        self._xyz_dlg = dlg
        if self._xyz_thread is not None and self._xyz_ctx is not None:
            dlg.set_running(True, int(self._xyz_ctx.get("total", 0)))
        dlg.set_queue(len(self._xyz_queue))
        dlg.show()

    def _xyz_model_choices(self) -> list[str]:
        return ([MERGE_PREFIX + e["name"] for e in self._merges
                 if not self._merge_broken(e)]
                + self._all_models.get("diffusion_models", []))

    _XYZ_CACHED_PREFIX = "[キャッシュ]"

    def _xyz_lora_choices(self) -> list[str]:
        """XYZ の LoRA 軸の候補: 表示モデルの系統に合う LoRA（判定不能も
        含む）と差分 LoRA（キャッシュ）を "名前:1" の形で。"""
        preset = self._current_preset()
        root = config.models_root() / "loras"
        out = []
        for rel in self._all_models.get("loras", []):
            fam = modelinfo.cached_family(root / rel)
            if fam in (None, modelinfo.UNKNOWN, preset):
                out.append(f"{Path(rel).with_suffix('')}:1")
        for c in self._cached_lora_items():
            if c["family"] in (modelinfo.UNKNOWN, preset):
                out.append(f"{self._XYZ_CACHED_PREFIX}{c['name']}:1")
        return out

    def _resolve_xyz_lora(self, value: str):
        """LoRA 軸の値 1 つを (GenParams.loras の name, 強度) に解決する
        （「なし」は None）。見つからなければ ValueError。"""
        parsed = xyz.parse_lora_value(value)
        if parsed is None:
            return None
        name, strength = parsed
        if name.startswith(self._XYZ_CACHED_PREFIX):
            want = name[len(self._XYZ_CACHED_PREFIX):].strip()
            for e in self._cached_loras:
                if e["name"] == want:
                    return cached_lora_spec(e), strength
            raise ValueError(f"LoRA 軸: 差分 LoRA「{want}」がキャッシュ一覧に"
                             "見つかりません")
        rel = self._resolve_lora_name(name)
        if rel is None:
            raise ValueError(f"LoRA 軸: 「{name}」に該当する LoRA が "
                             "models/loras に見つかりません")
        return rel, strength

    def _rescan_xyz_model_choices(self) -> list[str]:
        """XYZ ウィンドウの「再スキャン」: モデルを読み直して候補を返す
        （メイン画面のドロップダウンも同時に最新になる）。"""
        self.refresh_models()
        self.append_log("モデルを再スキャンしました（XYZ ウィンドウ）")
        return self._xyz_model_choices()

    def _persist_xyz_flag(self, key: str, checked: bool) -> None:
        """XYZ ウィンドウのチェック状態だけを即座に永続化する（他の入力欄は
        実行時にまとめて保存されるので、途中入力を上書きしないよう merge）。"""
        try:
            st = json.loads(str(self.settings.get("xyz", "{}")))
            if not isinstance(st, dict):
                st = {}
        except (ValueError, TypeError):
            st = {}
        st[key] = bool(checked)
        self.settings["xyz"] = json.dumps(st, ensure_ascii=False)
        self._schedule_save()

    def _xyz_parent(self):
        """Message-box parent: the XYZ window if alive, else the main window."""
        if self._xyz_dlg is not None:
            try:
                if self._xyz_dlg.isVisible():
                    return self._xyz_dlg
            except RuntimeError:
                self._xyz_dlg = None
        return self

    def _push_xyz_running(self, running: bool, total: int = 0) -> None:
        if self._xyz_dlg is None:
            return
        try:
            self._xyz_dlg.set_running(running, total)
            self._xyz_dlg.set_queue(len(self._xyz_queue))
        except RuntimeError:
            self._xyz_dlg = None

    def _push_xyz_queue(self) -> None:
        if self._xyz_dlg is None:
            return
        try:
            self._xyz_dlg.set_queue(len(self._xyz_queue))
        except RuntimeError:
            self._xyz_dlg = None

    def _on_xyz_requested(self, spec: dict, auto: bool = False) -> None:
        """XYZ ウィンドウの「実行」。アイドルなら即開始、XYZ 実行中なら
        押した時点の設定で組んだ実行を待機タスクとして積む（メイン画面の
        生成ボタンと同じ流儀）。``auto`` は連続モードの自動実行（多セル
        確認を出さずにループを続ける）。"""
        if not self._backend_ready():
            QMessageBox.warning(self._xyz_parent(), "未準備",
                                "ComfyUI の準備がまだ完了していません。")
            return
        if self._gen_thread is not None:
            QMessageBox.information(self._xyz_parent(), "実行中",
                                    "生成の完了後に実行してください。")
            return
        run = self._prepare_xyz_run(spec, auto)
        if run is None:
            return
        if self._xyz_thread is not None:
            self._xyz_queue.append(run)
            self.append_log(
                f"XYZ プロットをタスクに積みました（待機 {len(self._xyz_queue)} 件, "
                f"{run['ctx']['total']} セル, seed={run['ctx']['base'].seed}）")
            self._push_xyz_queue()
            return
        self._start_xyz_run(run)

    def _prepare_xyz_run(self, spec: dict, auto: bool = False
                         ) -> Optional[dict]:
        """押した時点のメイン画面の値と XYZ の軸から実行を組み立てる
        （{"ctx", "jobs"}）。入力エラー・確認キャンセル時は None。"""
        try:
            base = self._collect_params()
            base.batch_size = 1  # 1セル = 1枚
            base.llm_file = ""   # XYZ は LLM 整形の対象外（セルごとに回さない）
            if base.hires_enabled:
                self.append_log(
                    "XYZ: Hires fix 有効（各セルが2段生成になります）")
            axes = [xyz.axis_by_id(a["id"]) for a in spec["axes"]]
            values = [a["values"] for a in spec["axes"]]
            # LoRA 軸の値（"名前:強度"）はここでファイル / 差分 LoRA に解決
            # する（凡例ラベルは入力どおりの文字列のまま）。
            values = [[self._resolve_xyz_lora(v) for v in vals]
                      if a.id == "lora" else vals
                      for a, vals in zip(axes, values)]
            has_model_axis = any(a.id == "model" for a in axes)
            if not has_model_axis:
                # モデル軸があるときは全セルでモデルが差し替わるので、ベース
                # 選択に対する系統チェックは意味を持たない（整合は下で
                # セルごとに取る）。
                warn = self._config_warning(base)
                if warn:
                    QMessageBox.warning(self._xyz_parent(),
                                        "モデル設定を確認してください", warn)
                    return
            plan = xyz.plan_cells(base, axes, values)
            if has_model_axis:
                self._xyz_align_logged = set()
                # 系統ごとのプリセット設定（メモリ上）。アクティブな表示モデル
                # は現在の UI 値、他は切り替え時に退避した値を使う（設定
                # ファイルからは読まない）。
                confs = dict(self._preset_conf)
                confs[self._current_preset()] = self._collect_preset_conf()
                # 軸で振っているパラメータはプリセット値で上書きしない。
                locked = set()
                for a in axes:
                    locked |= _XYZ_AXIS_FIELDS.get(a.id, set())
                plan = [(idx, self._resolve_xyz_model_cell(p, confs, locked))
                        for idx, p in plan]
            jobs = [(idx, build_graph(p)) for idx, p in plan]
        except ValueError as e:
            QMessageBox.warning(self._xyz_parent(), "入力エラー", str(e))
            return None
        missing = sorted({m for _idx, p in plan
                          for m in self._params_missing_files(p)})
        if self._missing_files_error(missing, "XYZ プロット（マージモデル）",
                                     self._xyz_parent()):
            return None
        total = len(jobs)
        if total > 64 and not auto:
            res = QMessageBox.question(
                self._xyz_parent(), "確認",
                f"{total} 枚の画像を生成します。実行しますか？")
            if res != QMessageBox.Yes:
                return None
        # ウィンドウの入力状態を保存（次回開いたとき復元される）。
        if self._xyz_dlg is not None:
            try:
                self.settings["xyz"] = json.dumps(
                    self._xyz_dlg.state(), ensure_ascii=False)
                self._schedule_save()
            except RuntimeError:
                self._xyz_dlg = None

        # 個別保存はセル完成のたびに行う（キャンセルしても済んだ分は残る）。
        # フォーマットは押した時点の設定で run 全体を通して固定する。
        cell_fmt = None
        if spec.get("save_cells"):
            fmt = self.cb_img_format.currentText()
            if fmt == NO_SAVE:
                self.append_log("Format = 保存しない のため個別画像は保存しません")
            else:
                cell_fmt = (fmt, *self._encode_params(fmt))
        ctx = {
            "spec": spec,
            "base": base,
            "params": dict(plan),
            # セルごとの ComfyUI グラフ（個別保存の画像に埋め込む。通常の
            # 生成画像と同じく、そのセルを再現できるワークフローになる）。
            "graphs": dict(jobs),
            "nx": len(values[0]), "ny": len(values[1]), "nz": len(values[2]),
            "total": total,
            "cell_fmt": cell_fmt,   # (fmt, ext, quality) | None
            "cells_saved": 0,
            "embed": self.chk_embed_meta.isChecked(),  # run 全体で固定
        }
        return {"ctx": ctx, "jobs": jobs}

    def _start_xyz_run(self, run: dict) -> None:
        """組み立て済みの XYZ 実行を開始する。"""
        from datetime import datetime
        ctx, jobs = run["ctx"], run["jobs"]
        # 積んだ後にマージ元のファイルが消えた場合は、この実行と待機中の
        # XYZ タスクを止める。
        missing = sorted({m for p in ctx["params"].values()
                          for m in self._params_missing_files(p)})
        if self._missing_files_error(missing, "XYZ プロット（マージモデル）",
                                     self._xyz_parent()):
            if self._xyz_queue:
                n = len(self._xyz_queue)
                self._xyz_queue.clear()
                self.append_log(f"停止したため待機中の XYZ タスク {n} 件を破棄しました")
                self._push_xyz_queue()
            return
        spec, base, total = ctx["spec"], ctx["base"], ctx["total"]
        # ファイル名のタイムスタンプは実際に開始した時刻。
        ctx["stamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._xyz_ctx = ctx
        self._xyz_last_spec = spec   # 連続モードの次回実行用
        self._xyz_last_ok = False
        self._last_seed = base.seed
        self._last_params = base
        self._last_graph = None
        self._last_gen_ok = False  # 連続モードに拾わせない
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage(f"XYZ プロット生成中… (0/{total})")
        self.append_log(
            f"XYZ プロット開始: {total} セル "
            f"({self._xyz_ctx['nx']}x{self._xyz_ctx['ny']}x"
            f"{self._xyz_ctx['nz']}) seed={base.seed}")
        self._push_xyz_running(True, total)

        self._xyz_thread = QThread(self)
        self._xyz_worker = _XyzWorker(self.backend, jobs)
        self._xyz_worker.moveToThread(self._xyz_thread)
        self._xyz_thread.started.connect(self._xyz_worker.run)
        self._xyz_worker.progress.connect(self._on_progress)
        self._xyz_worker.preview.connect(self._on_preview_frame)
        self._xyz_worker.timing.connect(self._on_gen_timing)
        self._xyz_worker.cell_done.connect(self._on_xyz_cell_done)
        self._xyz_worker.cell_image.connect(self._on_xyz_cell_image)
        self._xyz_worker.cell_error.connect(self._on_xyz_cell_error)
        self._xyz_worker.done.connect(self._on_xyz_done)
        self._xyz_worker.failed.connect(self._on_xyz_failed)
        self._xyz_worker.done.connect(self._xyz_thread.quit)
        self._xyz_worker.failed.connect(self._xyz_thread.quit)
        self._xyz_thread.finished.connect(self._cleanup_xyz_thread)
        self._xyz_thread.start()

    def _resolve_xyz_model_cell(self, p: GenParams, confs: dict,
                                locked: set) -> GenParams:
        """モデル軸のセルを解決する: マージ展開 + プリセット適用 + 系統整合。

        値が「マージモデル：<名前>」なら登録済みマージエントリのレシピに
        展開する。次に、そのモデルの系統に対応する表示モデル（プリセット）
        の Models + 設定値（``confs`` = メモリ上の記憶。アクティブな系統は
        現在の UI 値）をセルへ適用する — ただし軸で振っている ``locked``
        フィールドと seed（比較可能性のため全セル共通）は上書きしない。
        最後に TE / CLIP / VAE の系統整合を検証し、食い違いは補正する
        （整合が取れないと conditioning の次元不一致でバックエンドが
        "mat1 and mat2 shapes cannot be multiplied" を出す）。
        """
        from dataclasses import replace
        if p.diffusion.startswith(MERGE_PREFIX):
            name = p.diffusion[len(MERGE_PREFIX):]
            entry = next((e for e in self._merges if e["name"] == name), None)
            if entry is None:
                raise ValueError(
                    f"モデル軸のマージモデル「{name}」が見つかりません"
                    "（削除または名前変更されていませんか）")
            if self._merge_broken(entry):
                raise ValueError(
                    f"モデル軸のマージモデル「{name}」は削除済みのマージ"
                    "モデルを素材にしているため使用できません")
            p = replace(p, diffusion="", checkpoint=False,
                        merge_models=[(norm_source(n), float(w))
                                      for n, w in entry["models"]],
                        merge_quant=str(entry["quant"]),
                        merge_low_memory=bool(entry["low_memory"]),
                        merge_name=str(entry["name"]),
                        merge_loras=[(str(n), float(s))
                                     for n, s in entry.get("loras", [])],
                        merge_diffs=[(norm_source(a), norm_source(b), float(s))
                                     for a, b, s in entry.get("diffs", [])])
            fam = self._merge_family(entry)
        else:
            fam = self._diffusion_family(p.diffusion)
            # フルチェックポイント（VAE/CLIP 内蔵）はセルでも内蔵を使う。
            # TE/VAE の系統整列は不要（むしろ内蔵を上書きしてしまう）。
            if p.diffusion and modelinfo.is_checkpoint(
                    config.models_root() / "diffusion_models" / p.diffusion):
                return replace(p, checkpoint=True, vae="", te=[])
            # 非チェックポイントのセルは内蔵フラグを必ず落とす（基準 p が
            # チェックポイントでも、このセルは分割ロードで整列する）。
            p = replace(p, checkpoint=False)

        conf = confs.get(fam) if families.is_family(fam) else None
        if conf:
            p = self._apply_preset_conf_to_cell(p, conf, locked)
            note = ("preset", fam)
            if note not in self._xyz_align_logged:
                self._xyz_align_logged.add(note)
                self.append_log(
                    f"XYZ: {fam} 系セルには表示モデル {fam} の設定を適用します")

        family = families.get(fam)
        if family is None:
            return p
        if family.dual_te:
            # SDXL: clip_l + clip_g と専用 VAE。
            te_ok = (p.clip_type == family.clip_type and len(p.te) == 2
                     and all(self._te_family(t) == fam for t in p.te))
            vae_ok = self._vae_family(p.vae) in family.vae_families
            if te_ok and vae_ok:
                return p
            clip_l, clip_g = self._sdxl_te_pair()
            if clip_l is None or clip_g is None:
                raise ValueError(
                    "モデル軸に SDXL 系がありますが、SDXL 用 text encoder"
                    "（clip_l / clip_g）が text_encoders に見つかりません。")
            vae = p.vae if vae_ok else self._first_vae(family.vae_families)
            if vae is None:
                raise ValueError(
                    "モデル軸に SDXL 系がありますが、SDXL 用 VAE が vae に"
                    "見つかりません。")
            self._log_xyz_align(fam, f"{clip_l} + {clip_g}", family.clip_type,
                                vae)
            return replace(p, te=[clip_l, clip_g], clip_type=family.clip_type,
                           vae=vae)

        te_ok = (len(p.te) == 1 and self._te_family(p.te[0]) == fam)
        clip_ok = families.clip_ok(family, p.clip_type)
        vae_ok = self._vae_family(p.vae) in family.vae_families
        if te_ok and clip_ok and vae_ok:
            return p
        te = list(p.te)
        if not te_ok:
            for name in self._all_models.get("text_encoders", []):
                if self._te_family(name) == fam:
                    te = [name]
                    break
            else:
                raise ValueError(
                    f"モデル軸の {p.diffusion} は {fam} 系ですが、対応する "
                    "text encoder が見つかりません。「設定…」から"
                    "ダウンロードしてください。")
        clip = p.clip_type if clip_ok else family.clip_type
        vae = p.vae if vae_ok else self._first_vae(family.vae_families)
        if vae is None:
            raise ValueError(f"{fam} 用の VAE が見つかりません。")
        self._log_xyz_align(fam, te[0], clip, vae)
        return replace(p, te=te, clip_type=clip, vae=vae)

    def _apply_preset_conf_to_cell(self, p: GenParams, conf: dict,
                                   locked: set) -> GenParams:
        """Apply a preset's remembered Models/設定 values to one XYZ cell.

        モデル（diffusion / マージ指定）は軸の値、seed はベース値のまま。
        ``locked`` のフィールド（軸で振っているもの）も触らない。ここで
        入った値は続く系統整合チェックで検証される（古いファイル名等は
        そこで補正される）。
        """
        from dataclasses import replace
        kw = {}
        te = [str(conf.get("te1", ""))]
        if conf.get("dual_te") and str(conf.get("te2", "")).strip():
            te.append(str(conf.get("te2")).strip())
        te = [t for t in te if t]
        if te:
            kw["te"] = te
        if str(conf.get("clip_type", "")):
            kw["clip_type"] = str(conf["clip_type"])
        if str(conf.get("vae", "")):
            kw["vae"] = str(conf["vae"])
        try:
            fields = (("width", "width", int), ("height", "height", int),
                      ("steps", "steps", int), ("cfg", "cfg", float),
                      ("sampler", "sampler", str),
                      ("scheduler", "scheduler", str),
                      ("weight_dtype", "dtype", str),
                      ("hires_enabled", "hires_enabled", bool),
                      ("hires_scale", "hires_scale", float),
                      ("hires_denoise", "hires_denoise", float),
                      ("hires_steps", "hires_steps", int),
                      ("hires_method", "hires_method", str))
            for field, key, cast in fields:
                if field not in locked and key in conf:
                    kw[field] = cast(conf[key])
        except (TypeError, ValueError):
            pass  # 壊れた保存値: 適用できた分だけ使う
        return replace(p, **kw)

    def _log_xyz_align(self, fam: str, te_desc: str, clip: str,
                       vae: str) -> None:
        note = (fam, te_desc, clip, vae)
        if note not in self._xyz_align_logged:
            self._xyz_align_logged.add(note)
            self.append_log(
                f"XYZ: {fam} 系モデルには text encoder {te_desc} / "
                f"CLIP type {clip} / VAE {vae} を使用します")

    def _on_xyz_cell_done(self, done: int, total: int) -> None:
        self.status.showMessage(f"XYZ プロット生成中… ({done}/{total})")
        if self._xyz_dlg is not None:
            try:
                self._xyz_dlg.set_progress(done)
            except RuntimeError:
                self._xyz_dlg = None

    def _on_xyz_cell_image(self, idx: int, data: bytes) -> None:
        """個別保存 ON のとき、セルが出来た直後にその画像を保存する。"""
        ctx = self._xyz_ctx
        if not ctx or not ctx.get("cell_fmt") or not metadata.AVAILABLE:
            return
        fmt, ext, quality = ctx["cell_fmt"]
        path = (self.paths.output_dir
                / f"xyz_{ctx['stamp']}_c{idx:03d}.{ext}")
        cell_text, cell_extra = self._build_metadata(
            ctx.get("params", {}).get(idx))
        try:
            metadata.save_with_metadata(
                data, path, fmt, quality, cell_text, extra=cell_extra,
                comfy_prompt=ctx.get("graphs", {}).get(idx),
                embed=bool(ctx.get("embed", True)))
            ctx["cells_saved"] += 1
        except Exception as e:  # noqa: BLE001
            self.append_log(f"保存に失敗: {e}")

    def _on_xyz_cell_error(self, idx: int, msg: str) -> None:
        self.append_log(f"XYZ セル {idx} でエラー（グレーで継続）: {msg}")

    def _on_xyz_done(self, results: dict) -> None:
        ctx = self._xyz_ctx or {}
        spec = ctx.get("spec", {})
        nx, ny, nz = ctx.get("nx", 1), ctx.get("ny", 1), ctx.get("nz", 1)
        n_cells = nx * ny * nz
        ok = sum(1 for v in results.values() if v)
        self.status.showMessage(f"XYZ プロット完了（{ok}/{n_cells} セル）")
        self.append_log(f"XYZ プロット完了: {ok}/{n_cells} セル成功")
        if ctx.get("cell_fmt"):
            self.append_log(f"個別画像を {ctx.get('cells_saved', 0)} 枚"
                            "保存しました（各セル生成直後に保存）")
        self._xyz_last_ok = ok > 0  # 連続モードは成功時のみ続行
        # モデル軸でマージモデルを使ったならピン状態 ●/○ を最新化。
        if any(p.merge_models for p in ctx.get("params", {}).values()):
            self._sync_merge_states()
        # 比較グリッドの合成/保存はワーカースレッドで行う（大きなグリッド
        # では数秒かかり、UI スレッドで行うとフリーズして見えるため）。
        # 表示 OFF なら合成自体をスキップ（保存も表示が前提）。
        if not spec.get("show_grid", True):
            return
        cells: list = [None] * n_cells
        for idx, data in results.items():
            cells[idx] = data or None
        labels = [a["labels"] for a in spec.get("axes", [])] or [[""]] * 3
        save_path = None
        if spec.get("save_grid", True):
            if metadata.AVAILABLE:
                from datetime import datetime
                stamp = (ctx.get("stamp")
                         or datetime.now().strftime("%Y%m%d_%H%M%S"))
                # グリッドは本機能の成果物なので Format 設定に関わらず PNG。
                save_path = (self.paths.output_dir
                             / f"xyz_{stamp}_{self._last_seed}.png")
            else:
                self.append_log("警告: Pillow/piexif が無いため保存できません")
        # メタデータ文字列は UI の値を読むためここ（UIスレッド）で組み立てる。
        params_text, extra = self._build_metadata(ctx.get("base"))
        for name, a in zip(("x", "y", "z"), spec.get("axes", [])):
            if a["id"] != "none":
                extra[f"xyz_{name}_type"] = a["id"]
                extra[f"xyz_{name}_values"] = ", ".join(a["labels"])
        self._start_xyz_compose(
            cells, nx, ny, nz, labels, int(spec.get("margin", 0)),
            save_path, params_text, extra, bool(ctx.get("embed", True)))

    def _start_xyz_compose(self, cells, nx, ny, nz, labels, margin,
                           save_path, params_text, extra, embed) -> None:
        thread = QThread(self)
        worker = _XyzComposeWorker(cells, nx, ny, nz, labels, True, margin,
                                   save_path, params_text, extra, embed)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_xyz_grid_ready)
        worker.done.connect(thread.quit)
        # 参照を保持（GC 防止）。連続モードで次の実行と重なっても互いに独立。
        self._xyz_compose_jobs.append((thread, worker))
        thread.finished.connect(
            lambda t=thread, w=worker: self._xyz_compose_jobs.remove((t, w)))
        thread.start()

    def _on_xyz_grid_ready(self, res: dict) -> None:
        self.append_log(f"比較画像を合成しました（{res['w']}x{res['h']}px）")
        for line in res.get("log", []):
            self.append_log(line)
        self._last_images = [res["png"]]
        self._show_image(res["png"])

    def _on_xyz_failed(self, msg: str) -> None:
        self.status.showMessage("XYZ プロット失敗")
        self.append_log("エラー: " + msg)
        if "キャンセル" not in msg:
            QMessageBox.critical(self._xyz_parent(), "XYZ プロットエラー", msg)

    def _cleanup_xyz_thread(self) -> None:
        self._xyz_thread = None
        self._xyz_worker = None
        self._xyz_ctx = None
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._push_xyz_running(False)
        # 成功（またはスキップによる中断）で終わったら、待機タスクを最優先
        # で消費し、無ければ XYZ 連続モード（同じ設定で次の実行）へ。
        skip = self._xyz_skip
        self._xyz_skip = False
        proceed = getattr(self, "_xyz_last_ok", False) or skip
        if proceed and self._xyz_queue:
            run = self._xyz_queue.pop(0)
            self.append_log(
                f"待機中の XYZ プロットを開始します（残り {len(self._xyz_queue)} 件）")
            self._push_xyz_queue()
            QTimer.singleShot(0, lambda r=run: self._start_xyz_run(r))
            return
        if not proceed and self._xyz_queue:
            # キャンセル/エラーで停止したときは待機タスクも破棄する。
            n = len(self._xyz_queue)
            self._xyz_queue.clear()
            self.append_log(f"停止したため待機中の XYZ タスク {n} 件を破棄しました")
            self._push_xyz_queue()
        spec = getattr(self, "_xyz_last_spec", None)
        if not (proceed and spec is not None and self._xyz_dlg is not None):
            return
        try:
            chained = self._xyz_dlg.chk_continuous.isChecked()
        except RuntimeError:
            self._xyz_dlg = None
            return
        if chained:
            QTimer.singleShot(
                0, lambda: self._on_xyz_requested(spec, auto=True))

    def _on_xyz_cancel_requested(self) -> None:
        """XYZ ウィンドウのキャンセル/スキップボタン。

        XYZ の連続 ON、またはタスクが積まれているときは「スキップ」: 現在の
        実行だけ中断し、待機タスク/連続はそのまま次へ進む。どちらでもない
        ときだけ従来どおりのキャンセル。"""
        if not self._xyz_worker:
            return
        xyz_cont = False
        if self._xyz_dlg is not None:
            try:
                xyz_cont = self._xyz_dlg.chk_continuous.isChecked()
            except RuntimeError:
                self._xyz_dlg = None
        if xyz_cont or self._xyz_queue:
            self._xyz_skip = True
            self.append_log("スキップ: 現在の XYZ 実行を中断して次へ進みます")
        else:
            self.append_log("XYZ プロットのキャンセルを要求しました")
        self._xyz_worker.cancel()
