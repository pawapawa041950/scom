"""PySide6 main window: model selection, generation settings, preview."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QEvent, QRegularExpression, Qt, QThread, QTimer, QUrl,
)
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QImage, QPixmap,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import (config, families, settings, metadata, modelinfo, prompt_llm,
                prompt_presets)
from .. import lora as lora_meta
from . import ansi_log
from .widgets import FlowLayout, GrowingTextEdit, WideComboBox
from ..comfy_backend import ComfyBackend, Progress
from ..workflow import (
    CLIP_TYPES_DUAL, CLIP_TYPES_SINGLE, DEFAULT_NEGATIVE, SAMPLERS,
    SCHEDULERS, GenParams, build_graph, format_merge_source, is_cached_lora,
    lora_label, merge_source_file,
)
from .mw_common import (
    MAX_SEED,
    MERGE_COLOR,
    MERGE_PREFIX,
    MERGE_TOKEN,
    NO_SAVE,
)
from .mw_lora import LoraMixin
from .mw_merge import MergeMixin
from .mw_xyz import XyzMixin
from .workers import _GenWorker, _StartWorker, _XyzWorker


class MainWindow(MergeMixin, LoraMixin, XyzMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scom - 画像生成")
        self.resize(1180, 760)

        self.paths = config.AppPaths()
        self.backend = ComfyBackend(self.paths)
        self._start_thread: Optional[QThread] = None
        self._gen_thread: Optional[QThread] = None
        self._gen_worker: Optional[_GenWorker] = None
        self._last_images: list[bytes] = []
        self._last_seed: int = 0
        self._last_params: Optional[GenParams] = None
        self._last_graph: Optional[dict] = None
        self._last_gen_ok: bool = False
        # スキップ（連続 ON 中のキャンセル）: 現在の生成だけ中断して
        # ループは続行するためのフラグ。cleanup で消費される。
        self._gen_skip: bool = False
        self._xyz_skip: bool = False
        # 生成中に積まれた待機タスク（押した時点の GenParams スナップショット）。
        self._gen_queue: list[GenParams] = []
        # ComfyUI の起動状態: "starting" → "ready" | "failed"。
        # backend.is_running() はプロセスを立ち上げた直後から True になり、
        # 応答可能になったかは分からないので、生成系の入口はこちらで判定する。
        self._backend_state: str = "starting"

        # Persisted settings. prompt/negative are read as initial values only.
        self.settings, settings_error = settings.load(self.paths.settings_path)
        # Named merge entries: {"id", "name", "models", "quant", "low_memory"}.
        self._merges: list[dict] = self._load_merges()
        self._merge_seq = int(self.settings.get("merge_seq", 0) or 0)
        self._merge_seq = max([self._merge_seq]
                              + [int(e["id"]) for e in self._merges])
        # Entries known to be built in backend RAM this session (best effort;
        # corrected via execution_cached events whenever an entry is used).
        self._merge_built_ids: set[int] = set()
        self._last_merge_id: Optional[int] = None   # entry used by last gen
        # 差分 LoRA（RAM キャッシュ）: {"id","name","model_a","model_b","rank"}
        self._cached_loras: list[dict] = self._load_cached_loras()
        # バックエンドの RAM に作成済みの差分 LoRA（マージウィンドウの ●/○）。
        self._cached_lora_built: set[int] = set()
        self._cached_lora_seq = max(
            [int(self.settings.get("cached_lora_seq", 0) or 0)]
            + [int(e["id"]) for e in self._cached_loras])
        self._merge_was_cached = False              # node "4" was a cache hit
        self._merge_dlg = None  # non-modal MergeDialog (at most one)
        self._merge_running = False  # a merge-only run is in flight
        # Applied LoRAs: [{"name": str, "strength": float}, ...] in chain order.
        self._loras: list[dict] = []
        self._absorbing_lora_tags = False      # _absorb_lora_tags の再入防止
        self._lora_tag_warned: set[str] = set()  # 見つからなかった <lora:名前>
        self._lora_dlg = None  # non-modal LoraDialog (at most one)
        # LoRA カードのホバーで出すトリガーワードのポップアップ（遅延生成）。
        self._lora_popup = None
        self._lora_pop_timer = QTimer(self)
        self._lora_pop_timer.setSingleShot(True)
        self._lora_pop_timer.setInterval(220)   # 離脱後この時間で閉じる
        self._lora_pop_timer.timeout.connect(self._hide_lora_popup)
        # Per-preset (anima/krea2/qwen21/sdxl) memory of the Models + 設定 values;
        # filled from settings in _apply_settings, kept fresh in _do_save.
        self._preset_conf: dict = {}
        self._active_preset: str = families.DEFAULT_ID
        # XYZ plot: non-modal dialog + sequential-cell worker state.
        self._xyz_dlg = None
        self._xyz_thread: Optional[QThread] = None
        self._xyz_worker: Optional[_XyzWorker] = None
        self._xyz_ctx: Optional[dict] = None
        # XYZ 実行中に「実行」を押すと、その時点の設定（メイン画面の値 +
        # XYZ ウィンドウの軸）で組んだ実行を待機タスクとして積む。
        self._xyz_queue: list[dict] = []
        # 比較グリッド合成スレッド（完了後に非同期で走る; 複数保持可）。
        self._xyz_compose_jobs: list = []
        self._loading = True
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()
        self.refresh_models()       # fill combos before applying saved selection
        self._reload_prompt_presets(quiet=True)
        self._apply_settings()
        self._sync_builtin_for_diffusion()  # 復元した diffusion に内蔵チェックを追従
        self._loading = False
        self._connect_autosave()

        if settings_error:
            # Don't silently reset (and don't overwrite) a hand-broken file.
            self.append_log(f"settings.toml の読み込みエラー: {settings_error}")
            QMessageBox.warning(
                self, "settings.toml の読み込みに失敗",
                "settings.toml に文法エラーがあるため、既定値で起動します。\n"
                "ファイルを修正するまで自動保存は行いません。\n\n"
                f"エラー: {settings_error}",
            )
        else:
            # Ensure a settings.toml exists from the start (incl. initial prompts).
            self._do_save()
        self._settings_broken = bool(settings_error)

        self.start_backend()

    # ----- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left block: an untitled outer group bundles Models + 設定 (with the
        # 表示モデル selector one level above them); the prompt box spans the
        # full width underneath.
        left = QWidget()
        grid = QGridLayout(left)
        grid.addWidget(self._build_top_group(), 0, 0)
        grid.addWidget(self._build_prompt_box(), 1, 0)
        grid.setRowStretch(1, 1)  # prompt area takes the remaining height
        left.setMinimumWidth(660)

        # Right column, top to bottom: log, preview, one-line image box,
        # action buttons.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("バックエンドログ…")
        ansi_log.style_log(self.log_view)
        right_layout.addWidget(self.log_view, stretch=1)

        self.preview = QLabel("プレビュー")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(512, 512)
        self.preview.setStyleSheet(
            "QLabel { background:#1e1e1e; color:#888; border:1px solid #333; }"
        )
        right_layout.addWidget(self.preview, stretch=3)
        right_layout.addWidget(self._build_image_box())
        right_layout.addWidget(self._build_action_box())

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)  # preview/log column expands
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self.status.addPermanentWidget(self.lbl_gen_time)  # プログレスバーの左
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("バックエンドを起動中…")

    def _build_top_group(self) -> QWidget:
        """Untitled outer group bundling the Models and 設定 categories.

        表示モデル is a cross-category setting (it filters Models AND drives
        the defaults), so it is drawn like the group's TITLE: a widget row
        overlaid on the frame's top border (QGroupBox titles are text-only,
        so the row is a floating child centered on the border line, with an
        opaque background that interrupts the line the way a title does)."""
        # Preset: picking anima/krea2/sdxl filters the model dropdowns to
        # that family. The Models + 設定 values are remembered per preset
        # (settings.toml "preset_conf") and restored on switch.
        self.cb_preset = WideComboBox()
        for fam in families.FAMILIES:
            self.cb_preset.addItem(fam.id, fam.id)
        self.cb_preset.setToolTip(
            "モデル・VAE・Text encoder をその系統に絞り込みます。"
            "Models と設定カテゴリの内容は系統ごとに記憶され、"
            "切り替えると前回の状態が復元されます"
        )
        self.cb_preset.currentIndexChanged.connect(self._on_preset_changed)

        title = QWidget()
        th = QHBoxLayout(title)
        th.setContentsMargins(6, 0, 6, 0)
        th.addWidget(QLabel("表示モデル:"))
        th.addWidget(self.cb_preset)
        title.setAutoFillBackground(True)  # hide the border line behind it
        title_h = title.sizeHint().height()

        wrap = QWidget()
        outer = QVBoxLayout(wrap)
        # The title row overhangs the group box by half its height.
        outer.setContentsMargins(0, title_h // 2, 0, 0)
        box = QGroupBox()
        v = QVBoxLayout(box)
        m = v.contentsMargins()
        v.setContentsMargins(m.left(), title_h // 2 + 4, m.right(), m.bottom())
        boxes = QHBoxLayout()
        boxes.addWidget(self._build_model_box(), stretch=1)
        settings_col = QVBoxLayout()
        settings_col.addWidget(self._build_settings_box())
        settings_col.addWidget(self._build_hires_box())
        settings_col.addStretch(1)
        boxes.addLayout(settings_col, stretch=1)
        v.addLayout(boxes)
        outer.addWidget(box)

        title.setParent(wrap)
        title.adjustSize()
        title.move(14, 0)  # same left offset feel as the "Models" title
        title.raise_()
        return wrap

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Models")
        form = QFormLayout(box)

        self.cb_diffusion = WideComboBox()
        self.cb_vae = WideComboBox()
        self.cb_te1 = WideComboBox()
        self.cb_te2 = WideComboBox()
        self.chk_dual_te = QCheckBox("2つ目の text encoder を使用 (DualCLIPLoader)")
        self.chk_dual_te.toggled.connect(self._on_dual_toggled)

        self.cb_clip_type = WideComboBox()
        self.cb_clip_type.addItems(CLIP_TYPES_SINGLE)

        # フルチェックポイント（VAE/CLIP 内蔵）を選んだときだけ有効。ON で内蔵の
        # VAE/CLIP を使い、下の VAE / Text encoder 指定は無視する。
        self.chk_builtin = QCheckBox("VAE / CLIP はモデル内蔵を使用（フルモデル）")
        self.chk_builtin.setToolTip(
            "SDXL のフルチェックポイントなど VAE/CLIP を内蔵するモデルで有効。"
            "ON にすると内蔵の VAE/CLIP を使い、下の VAE / Text encoder / "
            "CLIP type の指定は無視されます。")
        self.chk_builtin.setEnabled(False)
        self.chk_builtin.toggled.connect(self._on_builtin_toggled)

        # Picking a model auto-aligns the CLIP type / text encoder to its family
        # (e.g. a krea2 diffusion needs CLIP type 'krea2' + a Qwen3-VL encoder).
        self.cb_diffusion.currentTextChanged.connect(self._on_diffusion_changed)

        merge_btn = QPushButton("マージ設定…")
        merge_btn.clicked.connect(self._open_merge_dialog)
        refresh = QPushButton("再スキャン")
        refresh.clicked.connect(self.refresh_models)
        manage = QPushButton("設定…")
        manage.clicked.connect(self.open_models_dialog)
        btn_row = QHBoxLayout()
        btn_row.addWidget(merge_btn, stretch=1)
        btn_row.addWidget(refresh, stretch=1)
        btn_row.addWidget(manage, stretch=1)
        btns = QWidget(); btns.setLayout(btn_row)
        btn_row.setContentsMargins(0, 0, 0, 0)

        form.addRow("Diffusion:", self.cb_diffusion)
        form.addRow("", self.chk_builtin)
        form.addRow("VAE:", self.cb_vae)
        form.addRow("Text encoder 1:", self.cb_te1)
        form.addRow("", self.chk_dual_te)
        form.addRow("Text encoder 2:", self.cb_te2)
        form.addRow("CLIP type:", self.cb_clip_type)
        # Single-widget row: spans the label column too, so the three buttons
        # get the group box's full width (their labels were getting clipped).
        form.addRow(btns)
        self.cb_te2.setEnabled(False)
        return box

    def _build_prompt_box(self) -> QGroupBox:
        box = QGroupBox("Prompt")
        layout = QVBoxLayout(box)
        self.txt_prompt = GrowingTextEdit(min_lines=4, max_lines=15)
        self.txt_prompt.setPlaceholderText("positive prompt…")
        self.txt_negative = GrowingTextEdit(min_lines=3, max_lines=8)
        self.txt_negative.setPlaceholderText("negative prompt…")
        self.txt_negative.setPlainText(DEFAULT_NEGATIVE)
        # プロンプトの LLM 整形（app/prompt_llm.py）。候補は models/llm 内の
        # ファイル。選ぶと生成時にポジティブ欄を LLM が書き直す
        # （欄の文はそのまま、整形後はログと画像メタデータに残る）。
        llm_row = QHBoxLayout()
        llm_row.addWidget(QLabel("LLMで整形:"))
        self.cb_prompt_llm = WideComboBox()
        self.cb_prompt_llm.setToolTip(
            "生成時にポジティブプロンプトを LLM に書き直させます（日本語で"
            "書いても英語のプロンプトに変換）。表示モデルに合わせて自然文 / "
            "タグ列の書き方を切り替えます。整形後の文はログと画像メタデータに"
            "記録され、プロンプト欄は書き換えません。\n"
            "候補: models/llm に置いた LLM（Qwen-Image 2.1 の PE モデルは"
            "「設定…」からダウンロードできます）")
        llm_row.addWidget(self.cb_prompt_llm, stretch=1)
        layout.addLayout(llm_row)
        layout.addWidget(QLabel("Positive"))
        layout.addWidget(self.txt_prompt)
        # LoRA（選択ボタン + 適用中チップ）。旧「LoRA」カテゴリをここへ移設。
        self.btn_lora = QPushButton("LoRA選択…")
        self.btn_lora.setToolTip(
            "LoRA の一覧（サムネイル・トリガーワード付き）を開いて"
            "適用する LoRA を選びます")
        self.btn_lora.clicked.connect(self._open_lora_dialog)
        lora_row = QWidget()
        self._lora_flow = FlowLayout(lora_row, hspacing=6, vspacing=4)
        self._lora_flow.setContentsMargins(0, 0, 0, 0)
        self._lora_flow.addWidget(self.btn_lora)
        layout.addWidget(lora_row)
        layout.addWidget(QLabel("Negative"))
        layout.addWidget(self.txt_negative)

        # Prompt presets: pick a named entry from prompts.csv and append its
        # prompt/negative to the fields with the pen button.
        self.cb_prompt_preset = WideComboBox()
        self.cb_prompt_preset.setToolTip(
            "prompts.csv のプリセット（1列目: 設定名、2列目: プロンプト、"
            "3列目: ネガティブプロンプト）")
        btn_apply = QPushButton("書込み")
        btn_apply.setToolTip("選択中のプリセットをプロンプト欄・ネガティブ欄に追記")
        btn_apply.clicked.connect(self._apply_prompt_preset)
        btn_edit = QPushButton("編集")
        btn_edit.setToolTip("設定ファイル (prompts.csv) を開いて編集")
        btn_edit.clicked.connect(self._open_prompt_csv)
        btn_reload = QPushButton("再読込み")
        btn_reload.setToolTip("設定ファイルを再読み込み")
        btn_reload.clicked.connect(self._reload_prompt_presets)
        layout.addWidget(QLabel("プロンプト入力 (prompts.csvの1個目の設定が起動時↑に設定されます)"))
        preset_row = QHBoxLayout()
        preset_row.addWidget(self.cb_prompt_preset, stretch=1)
        preset_row.addWidget(btn_apply)
        preset_row.addWidget(btn_edit)
        preset_row.addWidget(btn_reload)
        layout.addLayout(preset_row)

        layout.addStretch(1)
        # Shift+Enter in either prompt field starts generation.
        self.txt_prompt.installEventFilter(self)
        self.txt_negative.installEventFilter(self)
        # ユーザーが挿入済みハイライトを手で消したときも LoRA 窓の表示を追従。
        self.txt_prompt.textChanged.connect(self._push_lora_inserted)
        self.txt_negative.textChanged.connect(self._push_lora_inserted)
        # <lora:名前:強度> が貼られたら LoRA を適用して文字列を取り除く。
        self.txt_prompt.textChanged.connect(
            lambda: self._absorb_lora_tags(self.txt_prompt))
        self.txt_negative.textChanged.connect(
            lambda: self._absorb_lora_tags(self.txt_negative))
        return box

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("設定")
        grid = QGridLayout(box)

        self.sp_width = QSpinBox(); self.sp_width.setRange(64, 4096)
        self.sp_width.setSingleStep(64); self.sp_width.setValue(1024)
        self.sp_height = QSpinBox(); self.sp_height.setRange(64, 4096)
        self.sp_height.setSingleStep(64); self.sp_height.setValue(1024)
        self.sp_steps = QSpinBox(); self.sp_steps.setRange(1, 200)
        self.sp_steps.setValue(30)
        self.sp_cfg = QDoubleSpinBox(); self.sp_cfg.setRange(0.0, 30.0)
        self.sp_cfg.setSingleStep(0.5); self.sp_cfg.setValue(4.0)
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 16)
        self.sp_batch.setValue(1)

        self.cb_sampler = WideComboBox(); self.cb_sampler.addItems(SAMPLERS)
        self.cb_sampler.setCurrentText("er_sde")
        self.cb_scheduler = WideComboBox(); self.cb_scheduler.addItems(SCHEDULERS)
        self.cb_scheduler.setCurrentText("simple")

        # Seeds can exceed 32-bit (QSpinBox limit), so use a text field.
        # Seed は保存も生成後の書き戻しもしない。-1（起動時の初期値）なら
        # 生成のたびに内部でランダム値を引く。実際に使った値はログと画像
        # メタデータに残る。
        self.ed_seed = QLineEdit("-1")
        self.ed_seed.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"-1|\d{1,19}"))
        )
        self.ed_seed.setToolTip(
            "-1 = 生成ごとにランダム（欄は書き換わりません。使われた値は"
            "ログ・画像メタデータで確認できます）")

        self.cb_dtype = WideComboBox()
        self.cb_dtype.addItems(["default", "fp8_e4m3fn", "fp8_e5m2"])

        # サイズ行は他の行と違い列をまたいだ一体レイアウトにする:
        # 「サイズ [Width] ⇄ [Height]」。入れ替えボタンが両者の間に来て
        # 直感的になり、専用の列を作らないので他の設定行の幅にも影響しない
        # （スパン配置は各列の幅要求を変えない）。
        self.sp_width.setToolTip("Width（横）")
        self.sp_height.setToolTip("Height（縦）")
        self.btn_swap_size = QPushButton("⇄")
        self.btn_swap_size.setFlat(True)
        self.btn_swap_size.setFixedSize(22, 22)
        self.btn_swap_size.setStyleSheet(
            "QPushButton { border: 1px solid #666; border-radius: 3px; }"
            "QPushButton:hover { background: rgba(128,128,128,0.25); }")
        self.btn_swap_size.setToolTip("Width と Height を入れ替え")
        self.btn_swap_size.clicked.connect(self._swap_size)
        size_cell = QHBoxLayout()
        size_cell.setContentsMargins(0, 0, 0, 0)
        size_cell.setSpacing(4)
        size_cell.addWidget(QLabel("Width"))
        size_cell.addWidget(self.sp_width, stretch=1)
        size_cell.addWidget(self.btn_swap_size)
        size_cell.addWidget(self.sp_height, stretch=1)
        size_cell.addWidget(QLabel("Height"))

        r = 0
        grid.addLayout(size_cell, r, 0, 1, 4)   # 行全体（列0〜3）をスパン
        r += 1
        grid.addWidget(QLabel("Steps"), r, 0); grid.addWidget(self.sp_steps, r, 1)
        grid.addWidget(QLabel("CFG"), r, 2); grid.addWidget(self.sp_cfg, r, 3)
        r += 1
        grid.addWidget(QLabel("Sampler"), r, 0); grid.addWidget(self.cb_sampler, r, 1)
        grid.addWidget(QLabel("Scheduler"), r, 2); grid.addWidget(self.cb_scheduler, r, 3)
        r += 1
        grid.addWidget(QLabel("Seed"), r, 0); grid.addWidget(self.ed_seed, r, 1)
        grid.addWidget(QLabel("Batch"), r, 2); grid.addWidget(self.sp_batch, r, 3)
        r += 1
        grid.addWidget(QLabel("UNet dtype"), r, 0); grid.addWidget(self.cb_dtype, r, 1)
        return box

    def _build_hires_box(self) -> QGroupBox:
        """Hires fix (latent): チェック付きグループ。OFF なら中身ごと無効。
        1段目の latent を補間拡大し、2段目の KSampler で細部を描き直す。"""
        box = QGroupBox("Hires fix (latent)")
        box.setCheckable(True)
        box.setChecked(False)
        box.setToolTip(
            "生成した latent を拡大して同じプロンプトで再サンプリングし、"
            "高解像度化します（生成時間は約2倍になります）")
        self.grp_hires = box
        grid = QGridLayout(box)

        self.sp_hires_scale = QDoubleSpinBox()
        self.sp_hires_scale.setRange(1.05, 4.0)
        self.sp_hires_scale.setSingleStep(0.05)
        self.sp_hires_scale.setDecimals(2)
        self.sp_hires_scale.setValue(1.5)
        self.sp_hires_scale.setToolTip("最終解像度 = 生成サイズ × この倍率")

        self.sp_hires_denoise = QDoubleSpinBox()
        self.sp_hires_denoise.setRange(0.0, 1.0)
        self.sp_hires_denoise.setSingleStep(0.05)
        self.sp_hires_denoise.setDecimals(2)
        self.sp_hires_denoise.setValue(0.55)
        self.sp_hires_denoise.setToolTip(
            "2段目の描き直し量。latent 拡大では 0.4〜0.7 が目安"
            "（低いとボケ、高いと構図が変わる）")

        self.sp_hires_steps = QSpinBox()
        self.sp_hires_steps.setRange(0, 200)
        self.sp_hires_steps.setValue(0)
        self.sp_hires_steps.setSpecialValueText("メインと同じ")
        self.sp_hires_steps.setToolTip("2段目のステップ数（0 = メインと同じ）")

        self.cb_hires_method = WideComboBox()
        self.cb_hires_method.addItems(
            ["bislerp", "nearest-exact", "bilinear", "bicubic"])
        self.cb_hires_method.setToolTip("latent の補間方法")

        grid.addWidget(QLabel("倍率"), 0, 0)
        grid.addWidget(self.sp_hires_scale, 0, 1)
        grid.addWidget(QLabel("Denoise"), 0, 2)
        grid.addWidget(self.sp_hires_denoise, 0, 3)
        grid.addWidget(QLabel("Steps"), 1, 0)
        grid.addWidget(self.sp_hires_steps, 1, 1)
        grid.addWidget(QLabel("補間"), 1, 2)
        grid.addWidget(self.cb_hires_method, 1, 3)

        # タイトルに現在の倍率での出力解像度を表示（サイズ/倍率の変更に追従）。
        self.sp_hires_scale.valueChanged.connect(self._update_hires_title)
        self.sp_width.valueChanged.connect(self._update_hires_title)
        self.sp_height.valueChanged.connect(self._update_hires_title)
        self._update_hires_title()
        return box

    def _hires_out_size(self) -> tuple[int, int]:
        """現在の倍率での最終解像度。ComfyUI の LatentUpscaleBy は latent
        単位（1/8px）で丸めるため、それと同じ計算で表示する。"""
        scale = float(self.sp_hires_scale.value())
        w = round(self.sp_width.value() / 8 * scale) * 8
        h = round(self.sp_height.value() / 8 * scale) * 8
        return int(w), int(h)

    def _update_hires_title(self, *_a) -> None:
        w, h = self._hires_out_size()
        self.grp_hires.setTitle(f"Hires fix (latent)  →  {w}x{h}")

    def _build_action_box(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        # 連続 is a checkbox: two states with an unmistakable ON/OFF look
        # (a pressed QPushButton reads poorly). Same isChecked/setChecked API.
        self.btn_xyz = QPushButton("XYZ")
        self.btn_xyz.setToolTip(
            "XYZ プロット: 複数パラメータの値の組み合わせで生成し、"
            "1枚のグリッド画像にまとめます")
        self.btn_xyz.setMinimumHeight(40)
        self.btn_xyz.setMaximumWidth(56)
        self.btn_xyz.clicked.connect(self._open_xyz_dialog)
        self.btn_continuous = QCheckBox("連続")
        self.btn_continuous.setToolTip(
            "ONの間、生成が終わるたびに自動で次を生成します"
            "（ON中のキャンセルボタンは「スキップ」= 現在の生成だけ中断）")
        self.btn_continuous.setMinimumHeight(40)
        self.btn_generate = QPushButton("ComfyUI 準備中…")
        self.btn_generate.setEnabled(False)   # ComfyUI の準備完了で有効化
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_continuous.toggled.connect(self._update_cancel_button)
        # 生成 stays about twice キャンセル; Ignored size policy makes the
        # layout use the stretch ratios alone instead of the text size hints.
        for b in (self.btn_generate, self.btn_cancel):
            b.setMinimumHeight(40)
            sp = b.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Ignored)
            b.setSizePolicy(sp)
        row.addWidget(self.btn_xyz)
        row.addWidget(self.btn_continuous)
        row.addWidget(self.btn_generate, stretch=2)
        row.addWidget(self.btn_cancel, stretch=1)
        # Progress moved to the status bar to save vertical space here.
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setMaximumWidth(220)
        # 直近の純粋な推論時間（プログレスバーの左隣に常駐表示）。
        self.lbl_gen_time = QLabel("")
        self.lbl_gen_time.setToolTip(
            "直近の生成の推論時間（サンプリングのみ。モデル読み込み・"
            "テキストエンコード・VAEデコード等は含みません）")
        return w

    # ----- image output settings (below the preview) ----------------------
    def _build_image_box(self) -> QGroupBox:
        """One-line Image box (sits between the preview and the log)."""
        box = QGroupBox("Image")
        row = QHBoxLayout(box)

        self.cb_img_format = WideComboBox()
        self.cb_img_format.addItems(["png", "jpg", "webp", NO_SAVE])  # index 0/1/2/3

        # One compact quality control per format, swapped on format change.
        self.stack_quality = QStackedWidget()

        self.sp_png_compress = QSpinBox()
        self.sp_png_compress.setRange(0, 9); self.sp_png_compress.setValue(6)
        self.sp_png_compress.setToolTip("0 = 低圧縮/大きい, 9 = 高圧縮/小さい")
        self.sp_jpg_quality = QSpinBox()
        self.sp_jpg_quality.setRange(1, 100); self.sp_jpg_quality.setValue(92)
        self.sp_webp_quality = QSpinBox()
        self.sp_webp_quality.setRange(1, 100); self.sp_webp_quality.setValue(90)
        self.sp_webp_quality.setToolTip("100 = lossless（ロスレス）")

        for label, spin in (("Compression (0-9)", self.sp_png_compress),
                             ("Quality (1-100)", self.sp_jpg_quality),
                             ("Quality (1-100)", self.sp_webp_quality)):
            page = QWidget()
            pr = QHBoxLayout(page)
            pr.setContentsMargins(0, 0, 0, 0)
            pr.addWidget(QLabel(label))
            pr.addWidget(spin)
            self.stack_quality.addWidget(page)
        # "保存しない" page (index 3): no quality control.
        nosave_page = QWidget()
        nr = QHBoxLayout(nosave_page); nr.setContentsMargins(0, 0, 0, 0)
        nr.addWidget(QLabel("（output へ自動保存しません）"))
        self.stack_quality.addWidget(nosave_page)
        self.cb_img_format.currentIndexChanged.connect(self.stack_quality.setCurrentIndex)

        self.chk_embed_meta = QCheckBox("画像にメタ情報を埋め込む")
        self.chk_embed_meta.setChecked(True)
        self.chk_embed_meta.setToolTip(
            "ONで生成設定（プロンプト・seed 等）を画像に埋め込みます"
            "（PNG: parameters チャンク / JPEG・WEBP: EXIF）。"
            "OFFにするとメタ情報を一切書き込みません")

        row.addWidget(QLabel("Format"))
        row.addWidget(self.cb_img_format)
        row.addSpacing(16)
        row.addWidget(self.stack_quality)
        row.addSpacing(16)
        row.addWidget(self.chk_embed_meta)
        row.addStretch(1)
        return box

    # ----- model scanning --------------------------------------------------
    def refresh_models(self) -> None:
        config.ensure_model_dirs()
        # Keep the unfiltered lists; the preset decides what the combos show.
        self._all_models = {
            "diffusion_models": config.scan_models("diffusion_models"),
            "vae": config.scan_models("vae"),
            "text_encoders": config.scan_models("text_encoders"),
            "loras": config.scan_models("loras"),
        }
        self.append_log(
            f"モデル: diffusion_models={len(self._all_models['diffusion_models'])} "
            f"vae={len(self._all_models['vae'])} "
            f"text_encoders={len(self._all_models['text_encoders'])} "
            f"loras={len(self._all_models['loras'])}"
        )
        self._apply_preset_filter()
        self._refresh_llm_combo()

    def _refresh_llm_combo(self) -> None:
        """「LLMで整形」の候補を models/llm のファイルから作り直す。"""
        current = self.cb_prompt_llm.currentData()
        if current is None:
            current = self.settings.get("prompt_llm", "")
        files = prompt_llm.list_llm_files()
        self.cb_prompt_llm.blockSignals(True)
        self.cb_prompt_llm.clear()
        self.cb_prompt_llm.addItem("使わない", "")
        for f in files:
            self.cb_prompt_llm.addItem(f, f)
        idx = self.cb_prompt_llm.findData(current)
        self.cb_prompt_llm.setCurrentIndex(max(0, idx))
        self.cb_prompt_llm.blockSignals(False)

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[str], allow_empty: bool = False) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if allow_empty:
            combo.addItem("")
        combo.addItems(items)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def open_models_dialog(self) -> None:
        from .models_dialog import ModelsDialog
        dlg = ModelsDialog(
            self.paths, sage_enabled=bool(self.settings.get("sage_attention",
                                                            False)),
            ck_enabled=bool(self.settings.get("ck_attention", False)),
            parent=self)
        dlg.sage_toggled.connect(self._on_sage_setting_toggled)
        dlg.ck_toggled.connect(self._on_ck_setting_toggled)
        dlg.exec()
        self.refresh_models()

    def _on_sage_setting_toggled(self, enabled: bool) -> None:
        self.settings["sage_attention"] = bool(enabled)
        self._schedule_save()
        self.append_log(
            "SageAttention を{}にしました（アプリ再起動後に反映）".format(
                "有効" if enabled else "無効"))

    def _on_ck_setting_toggled(self, enabled: bool) -> None:
        self.settings["ck_attention"] = bool(enabled)
        self._schedule_save()
        self.append_log(
            "Comfy Kitchen INT8 attention を{}にしました（アプリ再起動後に反映）"
            .format("有効" if enabled else "無効"))

    def _on_dual_toggled(self, checked: bool) -> None:
        self.cb_te2.setEnabled(checked)
        current = self.cb_clip_type.currentText()
        self.cb_clip_type.clear()
        self.cb_clip_type.addItems(CLIP_TYPES_DUAL if checked else CLIP_TYPES_SINGLE)
        idx = self.cb_clip_type.findText(current)
        if idx >= 0:
            self.cb_clip_type.setCurrentIndex(idx)

    # ----- model family helpers (content-based, see app/modelinfo.py) ------
    def _diffusion_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family(
            "diffusion_models",
            config.models_root() / "diffusion_models" / relname)

    def _te_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family(
            "text_encoders", config.models_root() / "text_encoders" / relname)

    def _vae_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family("vae", config.models_root() / "vae" / relname)

    def _first_vae(self, families: tuple[str, ...]) -> Optional[str]:
        """First scanned VAE whose content family is in ``families``."""
        for name in self._all_models.get("vae", []):
            if self._vae_family(name) in families:
                return name
        return None

    def _sdxl_te_pair(self) -> tuple[Optional[str], Optional[str]]:
        """(clip_l, clip_g) filenames from the scanned text encoders."""
        root = config.models_root() / "text_encoders"
        clip_l = clip_g = None
        for name in self._all_models.get("text_encoders", []):
            if self._te_family(name) != modelinfo.SDXL:
                continue
            kind = modelinfo.sdxl_te_kind(root / name)
            if kind == "clip_l" and clip_l is None:
                clip_l = name
            elif kind == "clip_g" and clip_g is None:
                clip_g = name
        return clip_l, clip_g

    def _select_te_for_family(self, fams: tuple[str, ...]) -> bool:
        """Select the first text encoder whose content is one of ``fams``."""
        for i in range(self.cb_te1.count()):
            name = self.cb_te1.itemText(i)
            if name and self._te_family(name) in fams:
                self.cb_te1.setCurrentIndex(i)
                return True
        return False

    def _diffusion_is_checkpoint(self) -> bool:
        """選択中の diffusion が VAE/CLIP 内蔵のフルチェックポイントか。"""
        if self._merge_selected():
            return False
        name = self.cb_diffusion.currentText().strip()
        if not name:
            return False
        return modelinfo.is_checkpoint(
            config.models_root() / "diffusion_models" / name)

    def _use_builtin(self) -> bool:
        return self.chk_builtin.isEnabled() and self.chk_builtin.isChecked()

    def _apply_builtin_enabled(self) -> None:
        """内蔵利用中は VAE/TE/CLIP 指定を触れなくする（無視されるため）。"""
        builtin = self._use_builtin()
        for w in (self.cb_vae, self.cb_te1, self.cb_te2,
                  self.cb_clip_type, self.chk_dual_te):
            w.setEnabled(not builtin)
        if not builtin:
            self.cb_te2.setEnabled(self.chk_dual_te.isChecked())

    def _sync_builtin_for_diffusion(self) -> None:
        """diffusion 選択に合わせて内蔵チェックの有効/既定を更新する。フル
        チェックポイントなら有効化して既定ON（＝内蔵を使う）。"""
        is_ckpt = self._diffusion_is_checkpoint()
        self.chk_builtin.blockSignals(True)
        self.chk_builtin.setEnabled(is_ckpt)
        self.chk_builtin.setChecked(is_ckpt)
        self.chk_builtin.blockSignals(False)
        self._apply_builtin_enabled()

    def _on_builtin_toggled(self, _checked: bool) -> None:
        self._apply_builtin_enabled()
        self._schedule_save()

    def _on_diffusion_changed(self, name: str) -> None:
        """Auto-align CLIP type / text encoder to the selected model's family."""
        if self._loading:
            return  # honor saved settings on startup; only react to user picks
        # マージ/チェックポイントを含めて内蔵チェックの状態を先に反映する。
        self._sync_builtin_for_diffusion()
        if self._merge_selected():
            return  # the merge recipe decides family; nothing to auto-align
        if self._diffusion_is_checkpoint():
            return  # 内蔵 VAE/CLIP を使うので系統整列(clip_l/clip_g等)は不要
        fam = families.get(self._diffusion_family(name))
        if fam is not None:
            self._align_ui_to_family(fam)

    def _align_ui_to_family(self, fam: "families.Family") -> None:
        """Make the TE / CLIP type / VAE selection usable for ``fam``: only
        the parts that do not fit are changed (the user's other picks stay)."""
        if fam.dual_te:
            # SDXL: DualCLIPLoader (clip_l + clip_g) + its CLIP type + VAE.
            if (self.cb_clip_type.currentText() == fam.clip_type
                    and self.chk_dual_te.isChecked()):
                return
            self.chk_dual_te.setChecked(True)  # switches the CLIP list to DUAL
            idx = self.cb_clip_type.findText(fam.clip_type)
            if idx >= 0:
                self.cb_clip_type.setCurrentIndex(idx)
            clip_l, clip_g = self._sdxl_te_pair()
            if clip_l:
                self.cb_te1.setCurrentText(clip_l)
            if clip_g:
                self.cb_te2.setCurrentText(clip_g)
        else:
            if self.chk_dual_te.isChecked():
                self.chk_dual_te.setChecked(False)
            if not families.clip_ok(fam, self.cb_clip_type.currentText()):
                idx = self.cb_clip_type.findText(fam.clip_type)
                if idx >= 0:
                    self.cb_clip_type.setCurrentIndex(idx)
            if self._te_family(self.cb_te1.currentText()) not in fam.te_families:
                self._select_te_for_family(fam.te_families)
        if self._vae_family(self.cb_vae.currentText()) not in fam.vae_families:
            v = self._first_vae(fam.vae_families)
            if v:
                self.cb_vae.setCurrentText(v)

    def _family_config_warning(self, params: GenParams) -> Optional[str]:
        """Pre-flight check: translate the cryptic backend mismatch into a
        clear, actionable message before a run is even submitted (what each
        family checks is defined in app/families.py)."""
        # For a merge, the family is judged from the first source model.
        name = (merge_source_file(params.merge_models[0][0])
                if params.merge_models else params.diffusion)
        fam = families.get(self._diffusion_family(name))
        if fam is None or (fam.te_warning is None and fam.vae_warning is None):
            return None
        msgs = []
        if fam.strict_clip and params.clip_type != fam.clip_type:
            msgs.append(f"・CLIP type を '{fam.clip_type}' にしてください"
                        f"（現在: {params.clip_type}）")
        if fam.te_warning is not None:
            matching = [t for t in params.te
                        if self._te_family(t) in fam.te_families]
            if fam.dual_te:
                bad_te = len(params.te) < 2 or len(matching) < 2
            else:
                bad_te = not matching
            if bad_te:
                msgs.append(fam.te_warning)
        if (fam.vae_warning is not None
                and self._vae_family(params.vae) not in fam.vae_families):
            msgs.append(fam.vae_warning)
        if not msgs:
            return None
        return (f"選択中のモデルは {fam.title} です。次を直してください:\n\n"
                + "\n".join(msgs))

    def _config_warning(self, params: GenParams) -> Optional[str]:
        """Family-mismatch pre-flight: returns a message when the current
        model needs a different TE/CLIP/VAE configuration."""
        # フルチェックポイントで内蔵 VAE/CLIP を使う場合は検証不要。
        if params.checkpoint and not params.vae and not params.te:
            return None
        return self._family_config_warning(params)

    # ----- preset filtering (family judged from file content) -------------
    def _current_preset(self) -> str:
        return self.cb_preset.currentData() or families.DEFAULT_ID

    def _set_preset(self, token: str) -> None:
        i = self.cb_preset.findData(token)
        self.cb_preset.setCurrentIndex(i if i >= 0 else 0)

    def _filter_for_preset(self, kind: str, files: list[str], preset: str) -> list[str]:
        if preset == "all":
            return files
        # 共用ファイル（"shared" = anima/krea2 の Qwen-Image VAE、krea2 と
        # 判定される Qwen3-VL 4B TE）は、それを使える系統にだけ出す
        # （app/families.py の shared_vae / te_from）。
        fam = families.get(preset)
        allowed = set(fam.allowed(kind)) if fam else {preset}
        root = config.models_root() / kind
        return [f for f in files
                if modelinfo.family(kind, root / f) in allowed]

    def _apply_preset_filter(self) -> None:
        """Repopulate the model combos with only the current preset's files."""
        if not hasattr(self, "_all_models"):
            return
        preset = self._current_preset()
        prev_data = self.cb_diffusion.currentData()
        self._fill_combo(self.cb_diffusion, self._filter_for_preset(
            "diffusion_models", self._all_models["diffusion_models"], preset))
        # Merge entries go at the top of the list, filtered like regular
        # models (family judged from the first source model), with a green
        # tint to stand out.
        self.cb_diffusion.blockSignals(True)
        shown = [e for e in self._merges
                 if not self._merge_broken(e)
                 and (preset == "all"
                      or self._merge_family(e) in (preset, "shared"))]
        for i, e in enumerate(shown):
            self.cb_diffusion.insertItem(
                i, MERGE_PREFIX + e["name"], f"{MERGE_TOKEN}:{e['id']}")
            self.cb_diffusion.setItemData(
                i, QBrush(QColor(MERGE_COLOR)), Qt.ForegroundRole)
        if isinstance(prev_data, str) and prev_data.startswith(MERGE_TOKEN):
            idx = self.cb_diffusion.findData(prev_data)
            if idx >= 0:
                self.cb_diffusion.setCurrentIndex(idx)
        self.cb_diffusion.blockSignals(False)
        self._fill_combo(self.cb_vae, self._filter_for_preset(
            "vae", self._all_models["vae"], preset))
        te = self._filter_for_preset(
            "text_encoders", self._all_models["text_encoders"], preset)
        self._fill_combo(self.cb_te1, te)
        self._fill_combo(self.cb_te2, te, allow_empty=True)

    def _on_preset_changed(self, *_) -> None:
        preset = self._current_preset()
        if not self._loading:
            # 現在の Models + 設定の内容を旧プリセットの枠に退避する。
            # 絞り込み（_apply_preset_filter）はコンボの選択を書き換えるので、
            # 必ずその前に退避すること。
            old = getattr(self, "_active_preset", None)
            if old and old != preset:
                self._preset_conf[old] = self._collect_preset_conf()
            self._active_preset = preset
        self._apply_preset_filter()
        if self._loading:
            return  # startup restores exact saved picks, no auto-defaulting
        # 新プリセットの保存値（無ければ既定値）を復元する。
        stored = self._preset_conf.get(preset)
        prev_loading = self._loading
        self._loading = True  # 復元中の自動整合・自動保存を抑止
        try:
            if stored:
                self._apply_preset_conf(stored)
            else:
                self._select_preset_defaults(preset)
        finally:
            self._loading = prev_loading
        self._sync_builtin_for_diffusion()  # 新 diffusion に内蔵チェックを追従
        if not self.cb_diffusion.count():
            self.append_log(
                f"{preset} のモデルが見つかりません。"
                "「設定…」からダウンロードしてください。")
        self._schedule_save()

    # ----- per-preset Models/設定 memory -----------------------------------
    def _collect_preset_conf(self) -> dict:
        """Current Models + 設定 values (persisted per preset)."""
        diffusion = (self.cb_diffusion.currentData()
                     if self._merge_selected()
                     else self.cb_diffusion.currentText())
        return {
            "diffusion": str(diffusion),
            "vae": self.cb_vae.currentText(),
            "te1": self.cb_te1.currentText(),
            "te2": self.cb_te2.currentText(),
            "dual_te": self.chk_dual_te.isChecked(),
            "clip_type": self.cb_clip_type.currentText(),
            "width": self.sp_width.value(),
            "height": self.sp_height.value(),
            "steps": self.sp_steps.value(),
            "cfg": float(self.sp_cfg.value()),
            "batch": self.sp_batch.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "dtype": self.cb_dtype.currentText(),
            "hires_enabled": self.grp_hires.isChecked(),
            "hires_scale": float(self.sp_hires_scale.value()),
            "hires_denoise": float(self.sp_hires_denoise.value()),
            "hires_steps": self.sp_hires_steps.value(),
            "hires_method": self.cb_hires_method.currentText(),
        }

    def _apply_preset_conf(self, c: dict) -> None:
        """Restore Models + 設定 values saved for a preset (combos are
        already filtered to that preset; missing files are skipped)."""
        try:
            saved_diffusion = str(c.get("diffusion", ""))
            if saved_diffusion.startswith(MERGE_TOKEN):
                idx = self.cb_diffusion.findData(saved_diffusion)
                if idx >= 0:
                    self.cb_diffusion.setCurrentIndex(idx)
            elif saved_diffusion:
                self.cb_diffusion.setCurrentText(saved_diffusion)
            self.cb_vae.setCurrentText(str(c.get("vae", "")))
            self.cb_te1.setCurrentText(str(c.get("te1", "")))
            self.cb_te2.setCurrentText(str(c.get("te2", "")))
            self.chk_dual_te.setChecked(bool(c.get("dual_te", False)))
            self._on_dual_toggled(self.chk_dual_te.isChecked())
            self.cb_clip_type.setCurrentText(
                str(c.get("clip_type", "stable_diffusion")))
            self.sp_width.setValue(int(c.get("width", self.sp_width.value())))
            self.sp_height.setValue(int(c.get("height", self.sp_height.value())))
            self.sp_steps.setValue(int(c.get("steps", self.sp_steps.value())))
            self.sp_cfg.setValue(float(c.get("cfg", self.sp_cfg.value())))
            self.sp_batch.setValue(int(c.get("batch", self.sp_batch.value())))
            self.cb_sampler.setCurrentText(str(c.get("sampler", "er_sde")))
            self.cb_scheduler.setCurrentText(str(c.get("scheduler", "simple")))
            self.ed_seed.setText(str(c.get("seed", "-1")))
            self.cb_dtype.setCurrentText(str(c.get("dtype", "default")))
            self.grp_hires.setChecked(bool(c.get("hires_enabled", False)))
            self.sp_hires_scale.setValue(float(c.get("hires_scale", 1.5)))
            self.sp_hires_denoise.setValue(
                float(c.get("hires_denoise", 0.55)))
            self.sp_hires_steps.setValue(int(c.get("hires_steps", 0)))
            self.cb_hires_method.setCurrentText(
                str(c.get("hires_method", "bislerp")))
        except (TypeError, ValueError):
            pass  # 壊れた保存値は途中まで適用（以後の保存で正される）

    def _select_preset_defaults(self, preset: str) -> None:
        """Pick sensible model/vae/te/CLIP defaults for the chosen family.

        The combos are already content-filtered to this family, so the text
        encoder and VAE simply default to the first entry; the diffusion model
        prefers a turbo/base variant by name when several are present."""
        fam = families.get(preset) or families.get(families.DEFAULT_ID)
        if fam.dual_te:
            # SDXL は dual TE (clip_l + clip_g) + その CLIP type。
            self.chk_dual_te.setChecked(True)
            self._select_preferred(self.cb_diffusion, fam.prefer)
            if self.cb_vae.count():
                self.cb_vae.setCurrentIndex(0)
            clip_l, clip_g = self._sdxl_te_pair()
            if clip_l:
                self.cb_te1.setCurrentText(clip_l)
            if clip_g:
                self.cb_te2.setCurrentText(clip_g)
        else:
            self.chk_dual_te.setChecked(False)  # single CLIPLoader
            self._select_preferred(self.cb_diffusion, fam.prefer)
            if self.cb_vae.count():
                self.cb_vae.setCurrentIndex(0)
            self._select_preferred(self.cb_te1, fam.te_prefer)
        idx = self.cb_clip_type.findText(fam.clip_type)
        if idx >= 0:
            self.cb_clip_type.setCurrentIndex(idx)
        # 初めて開いた系統の生成設定（保存値があればそちらが優先される）。
        d = fam.defaults_for(self.cb_diffusion.currentText())
        if "steps" in d:
            self.sp_steps.setValue(int(d["steps"]))
        if "cfg" in d:
            self.sp_cfg.setValue(float(d["cfg"]))
        if "sampler" in d:
            self.cb_sampler.setCurrentText(str(d["sampler"]))
        if "scheduler" in d:
            self.cb_scheduler.setCurrentText(str(d["scheduler"]))

    def _select_preferred(self, combo: QComboBox, prefer: tuple[str, ...]) -> None:
        """First item containing ``prefer[0]``, else ``prefer[1]`` ... ,
        else the first item."""
        for s in prefer:
            for i in range(combo.count()):
                if s in combo.itemText(i).lower():
                    combo.setCurrentIndex(i)
                    return
        if combo.count():
            combo.setCurrentIndex(0)

    # ----- settings persistence -------------------------------------------
    def _apply_settings(self) -> None:
        """Apply loaded settings to the widgets (called once, at startup)."""
        s = self.settings
        # Preset first: it filters the model dropdowns before we restore picks.
        # 旧バージョンの「すべて」("all") は保存されていたモデルの系統に移行。
        preset = str(s.get("preset", ""))
        if not families.is_family(preset):
            saved_diffusion = str(s.get("diffusion", ""))
            if saved_diffusion.startswith(MERGE_TOKEN):
                try:
                    entry = self._merge_entry_by_id(
                        int(saved_diffusion.split(":", 1)[1]))
                except (IndexError, ValueError):
                    entry = None
                fam = self._merge_family(entry) if entry else "unknown"
            else:
                fam = self._diffusion_family(saved_diffusion)
            preset = fam if families.is_family(fam) else families.DEFAULT_ID
        self._set_preset(preset)
        # Merge selections are stored as their data token ("__merge__:<id>");
        # they are restorable because unbuilt entries auto-merge at generation.
        saved_diffusion = str(s.get("diffusion", ""))
        if saved_diffusion.startswith(MERGE_TOKEN):
            idx = self.cb_diffusion.findData(saved_diffusion)
            if idx >= 0:
                self.cb_diffusion.setCurrentIndex(idx)
        else:
            self.cb_diffusion.setCurrentText(saved_diffusion)
        self.cb_vae.setCurrentText(str(s.get("vae", "")))
        self.cb_te1.setCurrentText(str(s.get("te1", "")))
        self.cb_te2.setCurrentText(str(s.get("te2", "")))
        self.chk_dual_te.setChecked(bool(s.get("dual_te", False)))
        self._on_dual_toggled(self.chk_dual_te.isChecked())  # sync clip list
        self.cb_clip_type.setCurrentText(str(s.get("clip_type", "stable_diffusion")))
        # prompt/negative: startup values come from the first prompts.csv
        # entry (already loaded); without one the constructor defaults stay.
        presets = getattr(self, "_prompt_presets", [])
        if presets:
            _name, prompt, negative = presets[0]
            self.txt_prompt.setPlainText(prompt)
            self.txt_negative.setPlainText(negative)
        self.sp_width.setValue(int(s.get("width", 1024)))
        self.sp_height.setValue(int(s.get("height", 1024)))
        self.sp_steps.setValue(int(s.get("steps", 30)))
        self.sp_cfg.setValue(float(s.get("cfg", 4.0)))
        self.sp_batch.setValue(int(s.get("batch", 1)))
        self.cb_sampler.setCurrentText(str(s.get("sampler", "er_sde")))
        self.cb_scheduler.setCurrentText(str(s.get("scheduler", "simple")))
        self.ed_seed.setText(str(s.get("seed", "-1")))
        self.cb_dtype.setCurrentText(str(s.get("dtype", "default")))
        idx = self.cb_prompt_llm.findData(str(s.get("prompt_llm", "")))
        self.cb_prompt_llm.setCurrentIndex(max(0, idx))
        # Hires fix (latent)
        self.grp_hires.setChecked(bool(s.get("hires_enabled", False)))
        self.sp_hires_scale.setValue(float(s.get("hires_scale", 1.5)))
        self.sp_hires_denoise.setValue(float(s.get("hires_denoise", 0.55)))
        self.sp_hires_steps.setValue(int(s.get("hires_steps", 0)))
        self.cb_hires_method.setCurrentText(
            str(s.get("hires_method", "bislerp")))
        # image output
        self.cb_img_format.setCurrentText(str(s.get("image_format", "png")))
        self.stack_quality.setCurrentIndex(self.cb_img_format.currentIndex())
        self.sp_png_compress.setValue(int(s.get("png_compress", 6)))
        self.sp_jpg_quality.setValue(int(s.get("jpg_quality", 92)))
        self.sp_webp_quality.setValue(int(s.get("webp_quality", 90)))
        self.chk_embed_meta.setChecked(bool(s.get("embed_metadata", True)))
        # Applied LoRAs は永続化しない（毎回まっさらで起動する仕様）。
        self._loras = []
        self._rebuild_lora_rows()
        # Per-preset Models/設定 memory (settings "preset_conf" JSON).
        try:
            pc = json.loads(str(s.get("preset_conf", "{}")))
            self._preset_conf = pc if isinstance(pc, dict) else {}
        except (ValueError, TypeError):
            self._preset_conf = {}
        self._active_preset = self._current_preset()

    def _connect_autosave(self) -> None:
        """Save whenever any persisted control changes."""
        for combo in (self.cb_diffusion, self.cb_vae, self.cb_te1, self.cb_te2,
                      self.cb_clip_type, self.cb_sampler, self.cb_scheduler,
                      self.cb_dtype, self.cb_img_format, self.cb_hires_method):
            combo.currentTextChanged.connect(self._schedule_save)
        for spin in (self.sp_width, self.sp_height, self.sp_steps, self.sp_batch,
                     self.sp_png_compress, self.sp_jpg_quality, self.sp_webp_quality,
                     self.sp_hires_steps):
            spin.valueChanged.connect(self._schedule_save)
        self.sp_cfg.valueChanged.connect(self._schedule_save)
        self.sp_hires_scale.valueChanged.connect(self._schedule_save)
        self.sp_hires_denoise.valueChanged.connect(self._schedule_save)
        self.grp_hires.toggled.connect(self._schedule_save)
        self.chk_dual_te.toggled.connect(self._schedule_save)
        self.chk_embed_meta.toggled.connect(self._schedule_save)
        self.ed_seed.textChanged.connect(self._schedule_save)
        self.cb_prompt_llm.currentIndexChanged.connect(self._schedule_save)

    def _schedule_save(self, *args) -> None:
        if self._loading:
            return
        self._save_timer.start()  # debounce rapid changes (e.g. spinbox drag)

    def _do_save(self) -> None:
        if getattr(self, "_settings_broken", False):
            return  # don't overwrite a hand-broken settings.toml
        # Persist everything except prompt/negative, which keep their file values.
        # Merge selections persist as their data token, not the display text.
        diffusion = (self.cb_diffusion.currentData()
                     if self._merge_selected()
                     else self.cb_diffusion.currentText())
        # Keep the active preset's slot in sync with the widgets.
        self._preset_conf[self._current_preset()] = self._collect_preset_conf()
        data = {
            "preset": self._current_preset(),
            "preset_conf": json.dumps(self._preset_conf, ensure_ascii=False),
            "diffusion": str(diffusion),
            "vae": self.cb_vae.currentText(),
            "te1": self.cb_te1.currentText(),
            "te2": self.cb_te2.currentText(),
            "dual_te": self.chk_dual_te.isChecked(),
            "clip_type": self.cb_clip_type.currentText(),
            "merges": json.dumps(
                [{"id": e["id"], "name": e["name"],
                  "models": [[n, w] for n, w in e["models"]],
                  "quant": e["quant"], "low_memory": e["low_memory"],
                  "loras": [[n, s] for n, s in e.get("loras", [])],
                  "diffs": [[a, b, s] for a, b, s in e.get("diffs", [])]}
                 for e in self._merges], ensure_ascii=False),
            "merge_seq": int(self._merge_seq),
            "cached_loras": json.dumps(
                [{"id": e["id"], "name": e["name"], "model_a": e["model_a"],
                  "model_b": e["model_b"], "rank": e["rank"]}
                 for e in self._cached_loras], ensure_ascii=False),
            "cached_lora_seq": int(self._cached_lora_seq),
            "width": self.sp_width.value(),
            "height": self.sp_height.value(),
            "steps": self.sp_steps.value(),
            "cfg": float(self.sp_cfg.value()),
            "batch": self.sp_batch.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "dtype": self.cb_dtype.currentText(),
            "hires_enabled": self.grp_hires.isChecked(),
            "hires_scale": float(self.sp_hires_scale.value()),
            "hires_denoise": float(self.sp_hires_denoise.value()),
            "hires_steps": self.sp_hires_steps.value(),
            "hires_method": self.cb_hires_method.currentText(),
            "prompt_llm": str(self.cb_prompt_llm.currentData() or ""),
            "image_format": self.cb_img_format.currentText(),
            "embed_metadata": self.chk_embed_meta.isChecked(),
            "png_compress": self.sp_png_compress.value(),
            "jpg_quality": self.sp_jpg_quality.value(),
            "webp_quality": self.sp_webp_quality.value(),
        }
        self.settings.update(data)
        try:
            settings.save(self.paths.settings_path, self.settings)
        except OSError as e:
            self.append_log(f"設定の保存に失敗: {e}")

    # ----- backend start ---------------------------------------------------
    def start_backend(self) -> None:
        # SageAttention の設定を起動フラグへ反映（未導入なら backend 側で無視）。
        self.backend.use_sage_attention = bool(
            self.settings.get("sage_attention", False))
        self.backend.use_ck_attention = bool(
            self.settings.get("ck_attention", False))
        self._start_thread = QThread(self)
        worker = _StartWorker(self.backend)
        worker.moveToThread(self._start_thread)
        self._start_thread.started.connect(worker.run)
        worker.log.connect(self.append_log)
        worker.done.connect(self._on_backend_ready)
        worker.failed.connect(self._on_backend_failed)
        worker.done.connect(self._start_thread.quit)
        worker.failed.connect(self._start_thread.quit)
        self._start_worker = worker  # keep ref
        self._start_thread.start()

    def _backend_ready(self) -> bool:
        """ComfyUI が起動し、リクエストを受け付けられる状態か。"""
        return self._backend_state == "ready" and self.backend.is_running()

    def _on_backend_ready(self) -> None:
        self._backend_state = "ready"
        self.status.showMessage(f"バックエンド準備完了: {self.backend.base_url}")
        self.append_log("バックエンド準備完了")
        if self._gen_thread is None and self._xyz_thread is None:
            self.btn_generate.setEnabled(True)
        self._update_generate_button()

    def _on_backend_failed(self, msg: str) -> None:
        self._backend_state = "failed"
        self._update_generate_button()
        self.status.showMessage("バックエンドの起動に失敗")
        self.append_log("エラー: " + msg)
        QMessageBox.critical(self, "バックエンドエラー", msg)

    # ----- generation ------------------------------------------------------
    def _seed_value(self) -> int:
        try:
            return int(self.ed_seed.text().strip())
        except ValueError:
            return -1

    def _collect_params(self) -> GenParams:
        te = [self.cb_te1.currentText().strip()]
        if self.chk_dual_te.isChecked() and self.cb_te2.currentText().strip():
            te.append(self.cb_te2.currentText().strip())

        # -1 は生成のたびに内部でランダム値へ解決する（欄は書き換えない。
        # 使った値はログ・画像メタデータに残る）。
        seed = self._seed_value()
        if seed < 0:
            seed = random.randint(0, MAX_SEED)

        entry = self._selected_merge_entry()
        self._last_merge_id = int(entry["id"]) if entry else None
        # フルチェックポイントで内蔵 ON のときは VAE/TE を空にして内蔵を使う。
        checkpoint = entry is None and self._diffusion_is_checkpoint()
        use_builtin = checkpoint and self.chk_builtin.isChecked()
        vae = "" if use_builtin else self.cb_vae.currentText().strip()
        te_list = [] if use_builtin else [t for t in te if t]
        return GenParams(
            diffusion="" if entry else self.cb_diffusion.currentText().strip(),
            checkpoint=checkpoint,
            merge_models=list(entry["models"]) if entry else [],
            merge_quant=entry["quant"] if entry else "",
            merge_low_memory=entry["low_memory"] if entry else False,
            merge_name=str(entry["name"]) if entry else "",
            merge_loras=list(entry.get("loras", [])) if entry else [],
            merge_diffs=list(entry.get("diffs", [])) if entry else [],
            vae=vae,
            te=te_list,
            clip_type=self.cb_clip_type.currentText(),
            loras=[(self._lora_param(e["name"]), float(e["strength"]))
                   for e in self._loras],
            prompt=self.txt_prompt.toPlainText(),
            negative=self.txt_negative.toPlainText(),
            width=self.sp_width.value(),
            height=self.sp_height.value(),
            steps=self.sp_steps.value(),
            cfg=self.sp_cfg.value(),
            sampler=self.cb_sampler.currentText(),
            scheduler=self.cb_scheduler.currentText(),
            seed=seed,
            batch_size=self.sp_batch.value(),
            weight_dtype=self.cb_dtype.currentText(),
            hires_enabled=self.grp_hires.isChecked(),
            hires_scale=float(self.sp_hires_scale.value()),
            hires_denoise=float(self.sp_hires_denoise.value()),
            hires_steps=self.sp_hires_steps.value(),
            hires_method=self.cb_hires_method.currentText(),
            llm_file=str(self.cb_prompt_llm.currentData() or ""),
            llm_style=prompt_llm.effective_style(
                self._current_preset(),
                str(self.cb_prompt_llm.currentData() or "")),
        )

    # ----- prompt presets (prompts.csv) -------------------------------------
    def _reload_prompt_presets(self, *_args, quiet: bool = False) -> None:
        try:
            # Created on first run so the sample preset seeds the prompt
            # fields at startup (the first entry is the startup prompt).
            prompt_presets.ensure_file(self.paths.prompts_path)
            self._prompt_presets = prompt_presets.load(self.paths.prompts_path)
        except (OSError, ValueError) as e:
            self.append_log(f"prompts.csv の読み込みエラー: {e}")
            return
        current = self.cb_prompt_preset.currentText()
        self.cb_prompt_preset.blockSignals(True)
        self.cb_prompt_preset.clear()
        self.cb_prompt_preset.addItems([n for n, _p, _n in self._prompt_presets])
        idx = self.cb_prompt_preset.findText(current)
        if idx >= 0:
            self.cb_prompt_preset.setCurrentIndex(idx)
        self.cb_prompt_preset.blockSignals(False)
        if not quiet:
            self.append_log(
                f"プロンプトプリセットを {len(self._prompt_presets)} 件読み込みました")

    def _apply_prompt_preset(self) -> None:
        i = self.cb_prompt_preset.currentIndex()
        if i < 0 or i >= len(getattr(self, "_prompt_presets", [])):
            return
        _name, prompt, negative = self._prompt_presets[i]
        self._append_prompt_text(self.txt_prompt, prompt)
        self._append_prompt_text(self.txt_negative, negative)

    @staticmethod
    def _append_prompt_text(edit, text: str) -> None:
        """Append preset text, comma-separated after any existing content."""
        if not text:
            return
        current = edit.toPlainText()
        if current.strip():
            sep = "" if current.rstrip().endswith(",") else ","
            edit.setPlainText(current.rstrip() + sep + " " + text)
        else:
            edit.setPlainText(text)

    def _open_prompt_csv(self) -> None:
        path = self.paths.prompts_path
        try:
            prompt_presets.ensure_file(path)
        except OSError as e:
            self.append_log(f"prompts.csv を作成できませんでした: {e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt signature)
        # Shift+Enter in a prompt field triggers generation.
        if (obj in (self.txt_prompt, self.txt_negative)
                and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Return, Qt.Key_Enter)
                and event.modifiers() & Qt.ShiftModifier):
            self.on_generate()
            return True
        # LoRA チップのホバーでトリガーワードのポップアップを開閉する。
        name = getattr(obj, "_lora_name", None)
        if name is not None:
            if event.type() == QEvent.Enter:
                self._show_lora_popup(name, getattr(obj, "_lora_chip", obj))
            elif event.type() == QEvent.Leave:
                self._lora_pop_timer.start()   # 猶予後に閉じる（保持判定つき）
        return super().eventFilter(obj, event)

    def on_generate(self) -> None:
        """生成ボタン / Shift+Enter。アイドルなら即開始、生成中なら現在の
        UI設定のスナップショットをタスクとして積む（完了後に順次消費）。"""
        if not self._backend_ready():
            # ボタンは準備完了まで無効。Shift+Enter 経由の呼び出しもここで止める。
            QMessageBox.warning(self, "未準備", "ComfyUI の準備がまだ完了していません。")
            return
        if self._xyz_thread is not None:
            return  # XYZ 実行中はタスク積み対象外（ボタンも無効）
        if self._merge_running:
            QMessageBox.information(
                self, "マージ実行中", "マージの完了後に生成してください。")
            return
        if self._merge_selected() and self._selected_merge_entry() is None:
            QMessageBox.warning(self, "マージ未設定",
                                "選択中のマージモデルが見つかりません。")
            self._open_merge_dialog()
            return
        sel = self._selected_merge_entry()
        if sel is not None and self._merge_broken(sel):
            QMessageBox.warning(
                self, "使用できないマージモデル",
                f"「{sel['name']}」は削除済みのマージモデルを素材にしている"
                "ため使用できません。マージウィンドウで素材を変更してください。")
            self._open_merge_dialog()
            return
        try:
            params = self._collect_params()
            warn = self._config_warning(params)
            if warn:
                QMessageBox.warning(self, "モデル設定を確認してください", warn)
                return
            build_graph(params)   # 積む場合も入力不足はこの場で検出する
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        if self._missing_files_error(self._params_missing_files(params),
                                     "画像生成（マージモデル）"):
            return
        if self._gen_thread is not None:
            # 生成中: タスクに積む（押した時点の設定・seed を保持）。
            self._gen_queue.append(params)
            self.append_log(
                f"生成をタスクに積みました（待機 {len(self._gen_queue)} 件, "
                f"seed={params.seed}）")
            self._update_generate_button()
            return
        self._start_generation(params)

    def _start_generation(self, params: GenParams) -> None:
        """1件の生成を開始する（params は確定済みのスナップショット）。"""
        try:
            graph = build_graph(params)
        except ValueError as e:
            # 積んだ後に前提が消えた場合（マージ削除等）のみ起こり得る。
            QMessageBox.warning(self, "入力不足", str(e))
            self._update_generate_button()
            return
        # 積んだ後にマージ元のファイルが消えた場合も、生成を止めて待機
        # タスクも破棄する（キャンセルやエラー停止と同じ扱い）。
        if self._missing_files_error(self._params_missing_files(params),
                                     "画像生成（マージモデル）"):
            if self._gen_queue:
                n = len(self._gen_queue)
                self._gen_queue.clear()
                self.append_log(f"停止したため待機中のタスク {n} 件を破棄しました")
            self._update_generate_button()
            return
        self._merge_was_cached = False
        self._last_seed = params.seed
        self._last_params = params
        self._last_graph = graph
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage("生成中…")
        self.append_log(f"生成 seed={params.seed} {params.width}x{params.height}")

        self._gen_thread = QThread(self)
        self._gen_worker = _GenWorker(self.backend, graph, params=params,
                                      user_data=self.paths.user_data)
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.status.connect(self._on_gen_status)
        self._gen_worker.rewritten.connect(self._on_prompt_rewritten)
        self._gen_worker.prepared.connect(self._on_gen_prepared)
        self._gen_worker.progress.connect(self._on_progress)
        self._gen_worker.preview.connect(self._on_preview_frame)
        self._gen_worker.cached.connect(self._on_cached_nodes)
        self._gen_worker.timing.connect(self._on_gen_timing)
        self._gen_worker.done.connect(self._on_gen_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()
        self._update_generate_button()

    def _update_generate_button(self, *_a) -> None:
        """生成中はボタンを「生成をタスクに積む (待機数)」表示にする。
        ComfyUI の準備が済むまでは状態表示（ボタン自体は無効のまま）。"""
        if self._backend_state != "ready":
            failed = self._backend_state == "failed"
            self.btn_generate.setText(
                "ComfyUI 起動失敗" if failed else "ComfyUI 準備中…")
            self.btn_generate.setToolTip(
                "ComfyUI の起動に失敗したため生成できません（ログを確認してください）"
                if failed else "ComfyUI の起動完了後に生成できるようになります")
        elif self._gen_thread is not None:
            self.btn_generate.setText(
                f"生成をタスクに積む ({len(self._gen_queue)})")
            self.btn_generate.setToolTip(
                "現在の設定・プロンプトのスナップショットを待機タスクとして"
                "積みます。現在の生成が終わると順番に実行されます")
        else:
            self.btn_generate.setText("生成")
            self.btn_generate.setToolTip("")
        # 待機数でキャンセル/スキップの表示が変わるので合わせて更新する。
        self._update_cancel_button()

    def on_cancel(self) -> None:
        """メイン画面のキャンセル/スキップボタン。

        連続 ON のとき、またはタスクが積まれているときは「スキップ」:
        現在の生成だけ中断し、待機タスク/連続はそのまま次へ進む。どちらでも
        ないときだけ従来どおりのキャンセル。XYZ 実行中に押された場合は XYZ を
        完全に停止する（XYZ のスキップは XYZ ウィンドウのボタンで行う）。"""
        if self._gen_worker:
            if self.btn_continuous.isChecked() or self._gen_queue:
                self._gen_skip = True
                self.append_log("スキップ: 現在の生成を中断して次へ進みます")
            else:
                self.append_log("キャンセルを要求しました")
            self._gen_worker.cancel()
        if self._xyz_worker:
            # メイン側からのキャンセルは XYZ ループ終了の合図。
            if self._xyz_dlg is not None:
                try:
                    self._xyz_dlg.chk_continuous.setChecked(False)
                except RuntimeError:
                    self._xyz_dlg = None
            self._xyz_worker.cancel()
            self.append_log("XYZ プロットのキャンセルを要求しました")

    def _update_cancel_button(self, *_a) -> None:
        """連続 ON かタスクが積まれているときは「スキップ」表示にする。

        どちらも「押しても待機タスクは残り、次へ進む」動作なのでラベルを
        揃える。待機タスクが無く連続 OFF のときだけ本当のキャンセル。"""
        queued = len(self._gen_queue)
        skip = self.btn_continuous.isChecked() or queued > 0
        self.btn_cancel.setText("スキップ" if skip else "キャンセル")
        if not skip:
            self.btn_cancel.setToolTip("")
        elif queued:
            self.btn_cancel.setToolTip(
                "現在の生成を中断して次の待機タスクに進みます"
                f"（待機 {queued} 件）")
        else:
            self.btn_cancel.setToolTip(
                "現在の生成を中断して次の生成に進みます（連続は続行）")

    def _on_gen_status(self, msg: str) -> None:
        self.status.showMessage(msg)
        self.append_log(msg)

    def _on_prompt_rewritten(self, original: str, text: str,
                             cached: bool) -> None:
        note = "（キャッシュ）" if cached else ""
        self.append_log(f"LLM 整形後のプロンプト{note}: {text}")

    def _on_gen_prepared(self, params: GenParams, graph: dict) -> None:
        # 整形後のプロンプトで組み直したグラフ（保存メタデータに使う）。
        self._last_params = params
        self._last_graph = graph

    def _on_progress(self, p: Progress) -> None:
        if p.maximum:
            self.progress.setMaximum(p.maximum)
            self.progress.setValue(p.value)
            self.progress.setFormat(f"{p.note} {p.value}/{p.maximum}")

    def _on_gen_timing(self, secs: float) -> None:
        """純粋な推論時間（サンプリングのみ）をログとステータスバーへ。"""
        self.lbl_gen_time.setText(f"推論 {secs:.2f} 秒")
        self.append_log(f"推論時間: {secs:.2f} 秒")

    def _swap_size(self) -> None:
        w, h = self.sp_width.value(), self.sp_height.value()
        self.sp_width.setValue(h)
        self.sp_height.setValue(w)

    def _on_preview_frame(self, data: bytes) -> None:
        # Live latent preview during sampling; does not replace the final image.
        self._show_image(data)

    def _on_gen_done(self, images: list) -> None:
        self.status.showMessage(f"完了（{len(images)} 枚）")
        self.append_log(f"完了: {len(images)} 枚")
        self._last_gen_ok = True
        self._last_images = images
        # A merge generation guarantees the merged model is (now) in backend
        # RAM — auto-built, reused from the pin cache, or a plain cache hit.
        # Sync ●/○ from the backend's pin cache (the source of truth).
        if (self._last_params is not None and self._last_params.merge_models
                and self._last_merge_id is not None):
            self._merge_built_ids.add(self._last_merge_id)
            self._sync_merge_states()
        if images:
            self._show_image(images[0])
            self._save_outputs(images)

    def _on_cached_nodes(self, nodes: list) -> None:
        # Node "4" is the merge node in every merge graph; seeing it in the
        # execution_cached list means the merged model came straight from RAM.
        if "4" in nodes:
            self._merge_was_cached = True

    def _save_outputs(self, images: list) -> list:
        """Save each generated image to output/ with embedded metadata."""
        from datetime import datetime
        fmt = self.cb_img_format.currentText()
        if fmt == NO_SAVE:
            self.append_log("画像は保存しません（Format = 保存しない）")
            return []
        if not metadata.AVAILABLE:
            self.append_log("警告: Pillow/piexif が無いため保存できません")
            return []
        out = self.paths.output_dir
        ext, quality = self._encode_params(fmt)
        params_text, extra = self._build_metadata()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []
        embed = self.chk_embed_meta.isChecked()
        for i, data in enumerate(images):
            path = out / f"scom_{stamp}_{self._last_seed}_{i}.{ext}"
            try:
                metadata.save_with_metadata(
                    data, path, fmt, quality, params_text,
                    extra=extra, comfy_prompt=self._last_graph, embed=embed,
                )
                saved.append(path)
            except Exception as e:  # noqa: BLE001
                self.append_log(f"保存に失敗: {e}")
        if saved:
            note = "メタデータ付き" if embed else "メタデータなし"
            self.append_log(f"{out} に {len(saved)} 枚保存しました（{note}）")
        else:
            self.append_log("警告: 画像を保存できませんでした")
        return saved

    def _encode_params(self, fmt: str) -> tuple[str, int]:
        """Return (extension, quality). For PNG, quality is the compress level."""
        if fmt == "jpg":
            return "jpg", self.sp_jpg_quality.value()
        if fmt == "webp":
            return "webp", self.sp_webp_quality.value()
        return "png", self.sp_png_compress.value()  # 0..9 compress level

    def _build_metadata(self, params: Optional[GenParams] = None) -> tuple[str, dict]:
        """Build the parameters string + structured dict from the last run
        (or from ``params`` when given, e.g. per-cell XYZ metadata)."""
        p = params if params is not None else self._last_params
        if p is None:
            return "", {}
        if p.merge_models:
            # キャッシュ上のマージモデルは一覧の項目名だけを書く（マージ元の
            # モデル名・比率は残さない）。名前の無い場合だけレシピで代用。
            # 再現用の完全な構成は埋め込みの ComfyUI グラフ側に残る。
            model_name = p.merge_name or format_merge_source({
                "models": p.merge_models, "diffs": p.merge_diffs,
                "loras": p.merge_loras, "quant": p.merge_quant})
        else:
            model_name = p.diffusion
        # webui 互換: メタデータ上のプロンプトには適用中 LoRA を
        # <lora:名前:強度> タグとして末尾に入れ込む（生成に使う実際の
        # プロンプトには含まれない — 適用は LoraLoader ノードで行う）。
        prompt_meta = p.prompt
        if p.loras:
            tags = " ".join(f"<lora:{lora_label(n)}:{w:g}>"
                            for n, w in p.loras)
            body = prompt_meta.rstrip()
            prompt_meta = (body + ", " + tags) if body.strip() else tags
        meta = {
            "prompt": prompt_meta,
            "negative": p.negative,
            "steps": p.steps,
            "sampler": p.sampler,
            "scheduler": p.scheduler,
            "cfg": p.cfg,
            "seed": p.seed,
            "width": p.width,
            "height": p.height,
            "model": model_name,
            "vae": p.vae,
            "text_encoder": ", ".join(p.te),
            "clip_type": p.clip_type,
            "batch": p.batch_size,
            "dtype": p.weight_dtype,
        }
        if p.hires_enabled:
            # webui 互換フィールド名（Hires upscale / Hires steps / Denoising
            # strength）に latent 方式の情報を足す。
            meta["hires_upscale"] = f"{p.hires_scale:g}"
            meta["hires_steps"] = (int(p.hires_steps) if p.hires_steps > 0
                                   else p.steps)
            meta["denoising_strength"] = f"{p.hires_denoise:g}"
            meta["hires_method"] = f"Latent ({p.hires_method})"
        if p.loras:
            def _lora_meta_name(n) -> str:
                # キャッシュ上の差分 LoRA は登録名だけ（差分元は書かない）。
                if is_cached_lora(n):
                    return str(n["cached_lora"])
                return str(n)
            meta["loras"] = ", ".join(f"{_lora_meta_name(n)}:{w:g}"
                                      for n, w in p.loras)
            # webui の "Lora hashes" フィールド（AutoV2 = SHA256 先頭10桁）。
            # ハッシュは LoRA ブラウザが計算・キャッシュ済みのものだけ使う
            # （ここで数GBのハッシュ計算を始めない）。
            cache = lora_meta.LoraCache(self.paths.user_data / "lora_cache")
            hashes = []
            for n, _w in p.loras:
                if is_cached_lora(n):
                    continue   # ファイルが無い（ハッシュも無い）
                e = cache.lookup(n, config.models_root() / "loras" / n)
                if e and e.get("sha256"):
                    hashes.append(f"{Path(n).stem}: {e['sha256'][:10]}")
            if hashes:
                meta["lora_hashes"] = ", ".join(hashes)
        if p.llm_file and p.prompt_original:
            meta["prompt_llm"] = p.llm_file
            meta["prompt_original"] = p.prompt_original
        params_text = metadata.build_parameters(meta)
        return params_text, {"app": "scom", **meta}

    def _on_gen_failed(self, msg: str) -> None:
        self.status.showMessage("生成失敗")
        self.append_log("エラー: " + msg)
        self._last_gen_ok = False
        if "キャンセル" not in msg:
            QMessageBox.critical(self, "生成エラー", msg)

    def _cleanup_gen_thread(self) -> None:
        self._gen_thread = None
        self._gen_worker = None
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        skip = self._gen_skip
        self._gen_skip = False
        proceed = self._last_gen_ok or skip
        if proceed and self._gen_queue:
            # 待機タスクを最優先で消費（連続 ON でもタスクが先）。
            params = self._gen_queue.pop(0)
            self.append_log(
                f"待機タスクを開始します（残り {len(self._gen_queue)} 件）")
            self._update_generate_button()
            QTimer.singleShot(0, lambda p=params: self._start_generation(p))
            return
        if proceed and self.btn_continuous.isChecked():
            # Continuous mode: start the next with the current UI values.
            self._update_generate_button()
            QTimer.singleShot(0, self.on_generate)
            return
        if self._gen_queue:
            # キャンセル/エラーで停止したときは待機タスクも破棄する。
            n = len(self._gen_queue)
            self._gen_queue.clear()
            self.append_log(f"停止したため待機中のタスク {n} 件を破棄しました")
        self._update_generate_button()

    def _show_image(self, data: bytes) -> None:
        img = QImage.fromData(data)
        if img.isNull():
            return
        pix = QPixmap.fromImage(img)
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ----- misc ------------------------------------------------------------
    def append_log(self, text: str) -> None:
        ansi_log.append_ansi(self.log_view, text)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        if self._last_images:
            self._show_image(self._last_images[0])

    def closeEvent(self, event) -> None:  # noqa: N802
        # The merge/XYZ/LoRA windows are parentless (so they can go behind
        # us); close them explicitly or the app would keep running after the
        # main window.
        for dlg in (self._merge_dlg, self._xyz_dlg, self._lora_dlg):
            if dlg is not None:
                try:
                    dlg.close()
                except RuntimeError:
                    pass
        self.append_log("バックエンドを終了中…")
        if self._gen_worker:
            self._gen_worker.cancel()
        if self._xyz_worker:
            self._xyz_worker.cancel()
        self.backend.stop()
        super().closeEvent(event)
