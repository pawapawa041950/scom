"""PySide6 main window: model selection, generation settings, preview."""
from __future__ import annotations

import json
import random
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRegularExpression, QTimer, QEvent,
)
from PySide6.QtCore import QUrl
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QImage, QPixmap,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget, QCheckBox,
)

from .. import config, settings, metadata, modelinfo, prompt_presets, xyz
from . import ansi_log
from .widgets import GrowingTextEdit, WideComboBox
from ..comfy_backend import ComfyBackend, BackendError, Progress
from ..workflow import (
    GenParams, build_graph, build_merge_graph, merge_pin_key, merge_recipe,
    SAMPLERS, SCHEDULERS, CLIP_TYPES_SINGLE, CLIP_TYPES_DUAL, DEFAULT_NEGATIVE,
)

MAX_SEED = 2**63 - 1

# Image format option that disables saving generated images to disk.
NO_SAVE = "保存しない"

# Merge entries in the diffusion combo: shown at the top of the list with a
# colored prefix; the item data is "__merge__:<id>".
MERGE_PREFIX = "マージモデル："
MERGE_TOKEN = "__merge__"
MERGE_COLOR = "#1a7f37"  # green tint for merge entries


class _StartWorker(QObject):
    """Starts the ComfyUI backend off the UI thread."""
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend):
        super().__init__()
        self.backend = backend

    def run(self) -> None:
        try:
            self.backend.start(log=self.log.emit)
            self.done.emit()
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _GenWorker(QObject):
    progress = Signal(Progress)
    preview = Signal(bytes)
    cached = Signal(list)  # node ids served from the backend output cache
    done = Signal(list)    # list[bytes]
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend, graph: dict):
        super().__init__()
        self.backend = backend
        self.graph = graph
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            images = self.backend.generate(
                self.graph,
                on_progress=self.progress.emit,
                on_preview=self.preview.emit,
                cancel=lambda: self._cancel,
                on_cached=self.cached.emit,
            )
            self.done.emit(images)
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _XyzWorker(QObject):
    """Runs the XYZ plot cells sequentially off the UI thread.

    セル単位のエラーは placeholder (None) にして続行するが、最初のセルで
    失敗した場合は設定不備とみなして全体を中止する。
    """
    progress = Signal(Progress)
    preview = Signal(bytes)
    cell_done = Signal(int, int)    # finished count, total
    cell_image = Signal(int, bytes)  # grid index, png (直後の逐次保存用)
    cell_error = Signal(int, str)   # grid index, message
    done = Signal(object)           # {grid_index: png bytes | None}
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend, jobs: list):
        super().__init__()
        self.backend = backend
        self.jobs = jobs  # list[(grid_index, graph)]
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        results: dict[int, Optional[bytes]] = {}
        try:
            for n, (idx, graph) in enumerate(self.jobs):
                if self._cancel:
                    raise BackendError("生成をキャンセルしました")
                try:
                    images = self.backend.generate(
                        graph,
                        on_progress=self.progress.emit,
                        on_preview=self.preview.emit,
                        cancel=lambda: self._cancel,
                    )
                except BackendError as e:
                    if self._cancel or n == 0:
                        raise
                    results[idx] = None
                    self.cell_error.emit(idx, str(e))
                else:
                    results[idx] = images[0] if images else None
                    if results[idx]:
                        self.cell_image.emit(idx, results[idx])
                self.cell_done.emit(n + 1, len(self.jobs))
            self.done.emit(results)
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class MainWindow(QMainWindow):
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
        self._merge_was_cached = False              # node "4" was a cache hit
        self._merge_dlg = None  # non-modal MergeDialog (at most one)
        # XYZ plot: non-modal dialog + sequential-cell worker state.
        self._xyz_dlg = None
        self._xyz_thread: Optional[QThread] = None
        self._xyz_worker: Optional[_XyzWorker] = None
        self._xyz_ctx: Optional[dict] = None
        self._loading = True
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()
        self.refresh_models()       # fill combos before applying saved selection
        self._reload_prompt_presets(quiet=True)
        self._apply_settings()
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

        # Left block, two columns: 1 = preset/models, 2 = settings/image,
        # with the prompt box spanning both columns underneath.
        left = QWidget()
        grid = QGridLayout(left)
        col1 = QVBoxLayout()
        col1.addWidget(self._build_model_box())
        col1.addStretch(1)
        col2 = QVBoxLayout()
        col2.addWidget(self._build_settings_box())
        col2.addWidget(self._build_image_box())
        col2.addStretch(1)
        grid.addLayout(col1, 0, 0)
        grid.addLayout(col2, 0, 1)
        grid.addWidget(self._build_prompt_box(), 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)  # prompt area takes the remaining height
        left.setMinimumWidth(660)

        # Right column: action buttons above the preview, log below
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._build_action_box())
        self.preview = QLabel("プレビュー")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(512, 512)
        self.preview.setStyleSheet(
            "QLabel { background:#1e1e1e; color:#888; border:1px solid #333; }"
        )
        right_layout.addWidget(self.preview, stretch=3)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("バックエンドログ…")
        ansi_log.style_log(self.log_view)
        right_layout.addWidget(self.log_view, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)  # preview/log column expands
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("バックエンドを起動中…")

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Models")
        form = QFormLayout(box)

        # Preset: picking anima/krea2 filters the model dropdowns to that
        # family and auto-selects its model/vae/te/CLIP.
        self.cb_preset = WideComboBox()
        self.cb_preset.addItem("anima", "anima")
        self.cb_preset.addItem("krea2", "krea2")
        self.cb_preset.addItem("すべて", "all")
        self.cb_preset.setCurrentIndex(2)  # default: すべて (show everything)
        self.cb_preset.setToolTip(
            "anima / krea2 を選ぶと、その構成に合うモデル・VAE・"
            "Text encoder・CLIP type に絞り込み＆自動選択します"
        )
        self.cb_preset.currentIndexChanged.connect(self._on_preset_changed)

        self.cb_diffusion = WideComboBox()
        self.cb_vae = WideComboBox()
        self.cb_te1 = WideComboBox()
        self.cb_te2 = WideComboBox()
        self.chk_dual_te = QCheckBox("2つ目の text encoder を使用 (DualCLIPLoader)")
        self.chk_dual_te.toggled.connect(self._on_dual_toggled)

        self.cb_clip_type = WideComboBox()
        self.cb_clip_type.addItems(CLIP_TYPES_SINGLE)

        # Picking a model auto-aligns the CLIP type / text encoder to its family
        # (e.g. a krea2 diffusion needs CLIP type 'krea2' + a Qwen3-VL encoder).
        self.cb_diffusion.currentTextChanged.connect(self._on_diffusion_changed)

        merge_btn = QPushButton("マージ設定…")
        merge_btn.clicked.connect(self._open_merge_dialog)
        refresh = QPushButton("再スキャン")
        refresh.clicked.connect(self.refresh_models)
        manage = QPushButton("モデル管理…")
        manage.clicked.connect(self.open_models_dialog)
        btn_row = QHBoxLayout()
        btn_row.addWidget(merge_btn, stretch=1)
        btn_row.addWidget(refresh, stretch=1)
        btn_row.addWidget(manage, stretch=1)
        btns = QWidget(); btns.setLayout(btn_row)
        btn_row.setContentsMargins(0, 0, 0, 0)

        form.addRow("表示モデル:", self.cb_preset)
        form.addRow("Diffusion:", self.cb_diffusion)
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
        self.txt_prompt = GrowingTextEdit(min_lines=4)
        self.txt_prompt.setPlaceholderText("positive prompt…")
        self.txt_negative = GrowingTextEdit(min_lines=3)
        self.txt_negative.setPlaceholderText("negative prompt…")
        self.txt_negative.setPlainText(DEFAULT_NEGATIVE)
        layout.addWidget(QLabel("Positive"))
        layout.addWidget(self.txt_prompt)
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
        self.ed_seed = QLineEdit("-1")
        self.ed_seed.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"-1|\d{1,19}"))
        )
        self.ed_seed.setToolTip("-1 = 毎回ランダム")
        self.chk_randomize = QCheckBox("生成ごとに seed をランダム化")
        self.chk_randomize.setChecked(True)

        self.cb_dtype = WideComboBox()
        self.cb_dtype.addItems(["default", "fp8_e4m3fn", "fp8_e5m2"])

        r = 0
        grid.addWidget(QLabel("Width"), r, 0); grid.addWidget(self.sp_width, r, 1)
        grid.addWidget(QLabel("Height"), r, 2); grid.addWidget(self.sp_height, r, 3)
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
        grid.addWidget(self.chk_randomize, r, 2, 1, 2)
        return box

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
        self.btn_continuous.setToolTip("ONの間、生成が終わるたびに自動で次を生成します")
        self.btn_continuous.setMinimumHeight(40)
        self.btn_generate = QPushButton("生成")
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
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
        return w

    # ----- image output settings (below the preview) ----------------------
    def _build_image_box(self) -> QGroupBox:
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

        row.addWidget(QLabel("Format"))
        row.addWidget(self.cb_img_format)
        row.addSpacing(16)
        row.addWidget(self.stack_quality)
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
        }
        self.append_log(
            f"モデル: diffusion_models={len(self._all_models['diffusion_models'])} "
            f"vae={len(self._all_models['vae'])} "
            f"text_encoders={len(self._all_models['text_encoders'])}"
        )
        self._apply_preset_filter()

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
        dlg = ModelsDialog(self.paths, self)
        dlg.exec()
        self.refresh_models()

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

    def _select_te_for_family(self, fam: str) -> bool:
        """Select the first text encoder whose content matches ``fam``."""
        for i in range(self.cb_te1.count()):
            name = self.cb_te1.itemText(i)
            if name and self._te_family(name) == fam:
                self.cb_te1.setCurrentIndex(i)
                return True
        return False

    def _on_diffusion_changed(self, name: str) -> None:
        """Auto-align CLIP type / text encoder to the selected model's family."""
        if self._merge_selected():
            return  # the merge recipe decides family; nothing to auto-align
        if self._loading:
            return  # honor saved settings on startup; only react to user picks
        if self._diffusion_family(name) == "krea2" and not self.chk_dual_te.isChecked():
            # Krea-2 requires CLIP type 'krea2' and a Qwen3-VL text encoder.
            idx = self.cb_clip_type.findText("krea2")
            if idx >= 0:
                self.cb_clip_type.setCurrentIndex(idx)
            self._select_te_for_family("krea2")

    def _krea2_config_warning(self, params: GenParams) -> Optional[str]:
        """Pre-flight check: translate the cryptic backend mismatch into a
        clear, actionable message before a Krea-2 run is even submitted."""
        # For a merge, the family is judged from the first source model.
        name = params.merge_models[0][0] if params.merge_models else params.diffusion
        if self._diffusion_family(name) != "krea2":
            return None
        msgs = []
        if params.clip_type != "krea2":
            msgs.append(f"・CLIP type を 'krea2' にしてください（現在: {params.clip_type}）")
        if not any(self._te_family(t) == "krea2" for t in params.te):
            msgs.append("・Text encoder に Krea-2 用（Qwen3-VL）を選択してください"
                        "（未取得ならモデル管理からダウンロード）")
        if not msgs:
            return None
        return ("選択中のモデルは Krea-2 です。次を直してください:\n\n"
                + "\n".join(msgs))

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
                    "models": [(str(n), float(w)) for n, w in e["models"]],
                    "quant": str(e.get("quant", "")),
                    "low_memory": bool(e.get("low_memory", False)),
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
                 "low_memory": bool(self.settings.get("merge_low_memory", False))}]

    def _merge_family(self, entry: dict) -> str:
        """Family of a merge entry, judged from its first source model
        (same convention as _krea2_config_warning)."""
        if not entry.get("models"):
            return "unknown"
        return self._diffusion_family(entry["models"][0][0])

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
            self._merge_dlg.set_entries(self._merges, self._merge_built_ids)
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
            config.models_root() / "diffusion_models", None)
        dlg.merge_requested.connect(self._on_merge_requested)
        dlg.save_requested.connect(self._on_save_requested)
        dlg.delete_requested.connect(self._on_merge_delete)
        dlg.rename_requested.connect(self._on_merge_rename)
        dlg.free_memory_requested.connect(self._on_free_memory)
        self._merge_dlg = dlg
        self._push_merge_state()
        dlg.show()

    def _on_merge_requested(self, entries, quant: str, low_memory: bool) -> None:
        """マージ button: register a new entry and build it in backend RAM."""
        models = [(str(n), float(w)) for n, w in entries]
        try:
            graph = build_merge_graph(models, quant, low_memory)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        self._merge_seq += 1
        entry = {"id": self._merge_seq, "name": f"マージモデル{self._merge_seq}",
                 "models": models, "quant": str(quant),
                 "low_memory": bool(low_memory)}
        self._merges.append(entry)
        self._schedule_save()
        self._apply_preset_filter()  # the new entry appears in the dropdown
        self._push_merge_state()
        if self._merge_dlg is not None:
            self._merge_dlg.select_entry(entry["id"])
        self._run_merge(graph, saving=False, entry_id=entry["id"])

    def _on_save_requested(self, entries, quant: str, low_memory: bool,
                           filename: str) -> None:
        models = [(str(n), float(w)) for n, w in entries]
        try:
            graph = build_merge_graph(models, quant, low_memory,
                                      save_to=filename)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        self._run_merge(graph, saving=True, entry_id=None)

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
                    merge_recipe(entry["models"]), entry["quant"],
                    entry["low_memory"])
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

    def _on_merge_rename(self, entry_id: int, name: str) -> None:
        e = self._merge_entry_by_id(entry_id)
        if e is None or not name.strip():
            return
        e["name"] = name.strip()
        self._apply_preset_filter()  # dropdown labels follow the new name
        self._schedule_save()
        self._push_merge_state()

    def _on_free_memory(self) -> None:
        if not self.backend.is_running():
            QMessageBox.warning(self, "未準備", "バックエンドが起動していません。")
            return
        try:
            self.backend.release_all_merges()
            self.backend.free_memory()
        except OSError as e:
            self.append_log(f"メモリ解放に失敗: {e}")
            return
        self._merge_built_ids.clear()
        self._push_merge_state()
        self.append_log("バックエンドのキャッシュとモデルをすべて解放しました"
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
            if merge_pin_key(e["models"], e["quant"], e["low_memory"])
            in pinned}
        self._push_merge_state()

    def _run_merge(self, graph: dict, saving: bool,
                   entry_id: Optional[int]) -> None:
        """Run a merge-only prompt (build in RAM / save to file) off-thread."""
        if not self.backend.is_running():
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
        self._merge_run_ctx = (saving, entry_id)
        self._gen_worker.done.connect(self._on_merge_worker_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()

    def _on_merge_worker_done(self, _images: list) -> None:
        saving, entry_id = getattr(self, "_merge_run_ctx", (False, None))
        self._on_merge_done(saving, entry_id)

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
        choices = ([MERGE_PREFIX + e["name"] for e in self._merges]
                   + self._all_models.get("diffusion_models", []))
        dlg = XyzDialog(choices, state, None)
        dlg.run_requested.connect(self._on_xyz_requested)
        dlg.cancel_requested.connect(self.on_cancel)
        self._xyz_dlg = dlg
        if self._xyz_thread is not None and self._xyz_ctx is not None:
            dlg.set_running(True, int(self._xyz_ctx.get("total", 0)))
        dlg.show()

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
        except RuntimeError:
            self._xyz_dlg = None

    def _on_xyz_requested(self, spec: dict, auto: bool = False) -> None:
        """Start an XYZ run. ``auto`` marks a continuous-mode chained run
        (skips the many-cells confirmation so the loop keeps going)."""
        if not self.backend.is_running():
            QMessageBox.warning(self._xyz_parent(), "未準備",
                                "バックエンドがまだ起動していません。")
            return
        if self._gen_thread is not None or self._xyz_thread is not None:
            QMessageBox.information(self._xyz_parent(), "実行中",
                                    "生成の完了後に実行してください。")
            return
        try:
            base = self._collect_params()
            base.batch_size = 1  # 1セル = 1枚
            axes = [xyz.axis_by_id(a["id"]) for a in spec["axes"]]
            values = [a["values"] for a in spec["axes"]]
            has_model_axis = any(a.id == "model" for a in axes)
            if not has_model_axis:
                # モデル軸があるときは全セルでモデルが差し替わるので、ベース
                # 選択に対する krea2 チェックは意味を持たない（整合は下で
                # セルごとに取る）。
                warn = self._krea2_config_warning(base)
                if warn:
                    QMessageBox.warning(self._xyz_parent(),
                                        "Krea-2 の設定を確認してください", warn)
                    return
            plan = xyz.plan_cells(base, axes, values)
            if has_model_axis:
                self._xyz_align_logged = set()
                plan = [(idx, self._resolve_xyz_model_cell(p))
                        for idx, p in plan]
            jobs = [(idx, build_graph(p)) for idx, p in plan]
        except ValueError as e:
            QMessageBox.warning(self._xyz_parent(), "入力エラー", str(e))
            return
        total = len(jobs)
        if total > 64 and not auto:
            res = QMessageBox.question(
                self._xyz_parent(), "確認",
                f"{total} 枚の画像を生成します。実行しますか？")
            if res != QMessageBox.Yes:
                return
        # ウィンドウの入力状態を保存（次回開いたとき復元される）。
        if self._xyz_dlg is not None:
            try:
                self.settings["xyz"] = json.dumps(
                    self._xyz_dlg.state(), ensure_ascii=False)
                self._schedule_save()
            except RuntimeError:
                self._xyz_dlg = None

        from datetime import datetime
        # 個別保存はセル完成のたびに行う（キャンセルしても済んだ分は残る）。
        # フォーマットは開始時点の設定で run 全体を通して固定する。
        cell_fmt = None
        if spec.get("save_cells"):
            fmt = self.cb_img_format.currentText()
            if fmt == NO_SAVE:
                self.append_log("Format = 保存しない のため個別画像は保存しません")
            else:
                cell_fmt = (fmt, *self._encode_params(fmt))
        self._xyz_ctx = {
            "spec": spec,
            "base": base,
            "params": dict(plan),
            "nx": len(values[0]), "ny": len(values[1]), "nz": len(values[2]),
            "total": total,
            "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "cell_fmt": cell_fmt,   # (fmt, ext, quality) | None
            "cells_saved": 0,
        }
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
        self._xyz_worker.cell_done.connect(self._on_xyz_cell_done)
        self._xyz_worker.cell_image.connect(self._on_xyz_cell_image)
        self._xyz_worker.cell_error.connect(self._on_xyz_cell_error)
        self._xyz_worker.done.connect(self._on_xyz_done)
        self._xyz_worker.failed.connect(self._on_xyz_failed)
        self._xyz_worker.done.connect(self._xyz_thread.quit)
        self._xyz_worker.failed.connect(self._xyz_thread.quit)
        self._xyz_thread.finished.connect(self._cleanup_xyz_thread)
        self._xyz_thread.start()

    def _resolve_xyz_model_cell(self, p: GenParams) -> GenParams:
        """モデル軸のセルを解決する: マージトークンの展開 + 系統整合。

        値が「マージモデル：<名前>」なら登録済みマージエントリのレシピに
        展開する。その後、モデルの系統に合う TE / CLIP type に揃える
        （メイン画面でモデルを選んだときの自動整合と同じ規約:
        krea2 -> CLIP type 'krea2' + Qwen3-VL 系 TE、anima ->
        'stable_diffusion' + Qwen3 0.6B 系 TE）。系統が判定できないモデルは
        ベース設定のまま。整合が取れないと conditioning の次元不一致で
        バックエンドが "mat1 and mat2 shapes cannot be multiplied" を出す。
        """
        from dataclasses import replace
        if p.diffusion.startswith(MERGE_PREFIX):
            name = p.diffusion[len(MERGE_PREFIX):]
            entry = next((e for e in self._merges if e["name"] == name), None)
            if entry is None:
                raise ValueError(
                    f"モデル軸のマージモデル「{name}」が見つかりません"
                    "（削除または名前変更されていませんか）")
            p = replace(p, diffusion="",
                        merge_models=[(str(n), float(w))
                                      for n, w in entry["models"]],
                        merge_quant=str(entry["quant"]),
                        merge_low_memory=bool(entry["low_memory"]))
            fam = self._merge_family(entry)
        else:
            fam = self._diffusion_family(p.diffusion)
        if fam not in ("anima", "krea2"):
            return p
        te_ok = bool(p.te) and self._te_family(p.te[0]) == fam
        clip_ok = (p.clip_type == "krea2") == (fam == "krea2")
        if te_ok and clip_ok:
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
                    "text encoder が見つかりません。「モデル管理…」から"
                    "ダウンロードしてください。")
        clip = p.clip_type if clip_ok else (
            "krea2" if fam == "krea2" else "stable_diffusion")
        note = (fam, te[0], clip)
        if note not in self._xyz_align_logged:
            self._xyz_align_logged.add(note)
            self.append_log(
                f"XYZ: {fam} 系モデルには text encoder {te[0]} / "
                f"CLIP type {clip} を使用します")
        return replace(p, te=te, clip_type=clip)

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
                data, path, fmt, quality, cell_text, extra=cell_extra)
            ctx["cells_saved"] += 1
        except Exception as e:  # noqa: BLE001
            self.append_log(f"保存に失敗: {e}")

    def _on_xyz_cell_error(self, idx: int, msg: str) -> None:
        self.append_log(f"XYZ セル {idx} でエラー（グレーで継続）: {msg}")

    def _on_xyz_done(self, results: dict) -> None:
        ctx = self._xyz_ctx or {}
        spec = ctx.get("spec", {})
        nx, ny, nz = ctx.get("nx", 1), ctx.get("ny", 1), ctx.get("nz", 1)
        cells: list = [None] * (nx * ny * nz)
        for idx, data in results.items():
            cells[idx] = QImage.fromData(data) if data else None
        labels = [a["labels"] for a in spec.get("axes", [])] or [[""]] * 3
        grid = xyz.compose_grid(
            cells, nx, ny, nz, labels[0], labels[1], labels[2],
            draw_legend=bool(spec.get("legend", True)),
            margin=int(spec.get("margin", 0)))
        png = xyz.qimage_png_bytes(grid)
        ok = sum(1 for v in results.values() if v)
        self.status.showMessage(f"XYZ プロット完了（{ok}/{len(cells)} セル）")
        self.append_log(f"XYZ プロット完了: {ok}/{len(cells)} セル成功 "
                        f"（グリッド {grid.width()}x{grid.height()}px）")
        self._xyz_last_ok = ok > 0  # 連続モードは成功時のみ続行
        self._last_images = [png]
        self._show_image(png)
        self._save_xyz_outputs(png, results)
        # モデル軸でマージモデルを使ったならピン状態 ●/○ を最新化。
        if any(p.merge_models for p in ctx.get("params", {}).values()):
            self._sync_merge_states()
        if self.chk_randomize.isChecked():
            self._set_seed(-1)

    def _save_xyz_outputs(self, grid_png: bytes, results: dict) -> None:
        """Save the composed grid（個別セルは生成直後に保存済み）."""
        if not metadata.AVAILABLE:
            self.append_log("警告: Pillow/piexif が無いため保存できません")
            return
        ctx = self._xyz_ctx or {}
        spec = ctx.get("spec", {})
        base = ctx.get("base")
        from datetime import datetime
        stamp = ctx.get("stamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        # グリッド画像は本機能の成果物なので Format 設定に関わらず PNG で保存。
        params_text, extra = self._build_metadata(base)
        for name, a in zip(("x", "y", "z"), spec.get("axes", [])):
            if a["id"] != "none":
                extra[f"xyz_{name}_type"] = a["id"]
                extra[f"xyz_{name}_values"] = ", ".join(a["labels"])
        path = self.paths.output_dir / f"xyz_{stamp}_{self._last_seed}.png"
        try:
            metadata.save_with_metadata(
                grid_png, path, "png", 6, params_text, extra=extra)
            self.append_log(f"{path} に保存しました（メタデータ付き）")
        except Exception as e:  # noqa: BLE001
            self.append_log(f"グリッド画像の保存に失敗: {e}")
        if ctx.get("cell_fmt"):
            self.append_log(f"個別画像を {ctx.get('cells_saved', 0)} 枚"
                            "保存しました（各セル生成直後に保存）")

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
        # XYZ 連続モード: 成功して終わり、ウィンドウの連続がONなら同じ設定で
        # 次の実行を開始する（seed はメイン画面のランダム化設定に従う）。
        spec = getattr(self, "_xyz_last_spec", None)
        if not (getattr(self, "_xyz_last_ok", False) and spec is not None
                and self._xyz_dlg is not None):
            return
        try:
            chained = self._xyz_dlg.chk_continuous.isChecked()
        except RuntimeError:
            self._xyz_dlg = None
            return
        if chained:
            QTimer.singleShot(
                0, lambda: self._on_xyz_requested(spec, auto=True))

    # ----- preset filtering (family judged from file content) -------------
    def _current_preset(self) -> str:
        return self.cb_preset.currentData() or "all"

    def _set_preset(self, token: str) -> None:
        i = self.cb_preset.findData(token)
        self.cb_preset.setCurrentIndex(i if i >= 0 else self.cb_preset.findData("all"))

    def _filter_for_preset(self, kind: str, files: list[str], preset: str) -> list[str]:
        if preset == "all":
            return files
        root = config.models_root() / kind
        return [f for f in files
                if modelinfo.family(kind, root / f) in (preset, "shared")]

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
                 if preset == "all" or self._merge_family(e) in (preset, "shared")]
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
        self._apply_preset_filter()
        if self._loading:
            return  # startup restores exact saved picks, no auto-defaulting
        preset = self._current_preset()
        if preset in ("anima", "krea2"):
            self._select_preset_defaults(preset)
            if not self.cb_diffusion.count():
                self.append_log(
                    f"{preset} のモデルが見つかりません。"
                    "「モデル管理…」からダウンロードしてください。")
        self._schedule_save()

    def _select_preset_defaults(self, preset: str) -> None:
        """Pick sensible model/vae/te/CLIP defaults for the chosen family.

        The combos are already content-filtered to this family, so the text
        encoder and VAE simply default to the first entry; the diffusion model
        prefers a turbo/base variant by name when several are present."""
        self.chk_dual_te.setChecked(False)  # both families use a single CLIPLoader
        prefer = ("base",) if preset == "anima" else ("turbo",)
        self._select_preferred(self.cb_diffusion, prefer)
        if self.cb_vae.count():
            self.cb_vae.setCurrentIndex(0)
        if self.cb_te1.count():
            self.cb_te1.setCurrentIndex(0)
        clip = "krea2" if preset == "krea2" else "stable_diffusion"
        idx = self.cb_clip_type.findText(clip)
        if idx >= 0:
            self.cb_clip_type.setCurrentIndex(idx)

    def _select_preferred(self, combo: QComboBox, prefer: tuple[str, ...]) -> None:
        for i in range(combo.count()):
            if any(s in combo.itemText(i).lower() for s in prefer):
                combo.setCurrentIndex(i)
                return
        if combo.count():
            combo.setCurrentIndex(0)

    # ----- settings persistence -------------------------------------------
    def _apply_settings(self) -> None:
        """Apply loaded settings to the widgets (called once, at startup)."""
        s = self.settings
        # Preset first: it filters the model dropdowns before we restore picks.
        self._set_preset(str(s.get("preset", "all")))
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
        self.chk_randomize.setChecked(bool(s.get("randomize", True)))
        self.cb_dtype.setCurrentText(str(s.get("dtype", "default")))
        # image output
        self.cb_img_format.setCurrentText(str(s.get("image_format", "png")))
        self.stack_quality.setCurrentIndex(self.cb_img_format.currentIndex())
        self.sp_png_compress.setValue(int(s.get("png_compress", 6)))
        self.sp_jpg_quality.setValue(int(s.get("jpg_quality", 92)))
        self.sp_webp_quality.setValue(int(s.get("webp_quality", 90)))

    def _connect_autosave(self) -> None:
        """Save whenever any persisted control changes."""
        for combo in (self.cb_diffusion, self.cb_vae, self.cb_te1, self.cb_te2,
                      self.cb_clip_type, self.cb_sampler, self.cb_scheduler,
                      self.cb_dtype, self.cb_img_format):
            combo.currentTextChanged.connect(self._schedule_save)
        for spin in (self.sp_width, self.sp_height, self.sp_steps, self.sp_batch,
                     self.sp_png_compress, self.sp_jpg_quality, self.sp_webp_quality):
            spin.valueChanged.connect(self._schedule_save)
        self.sp_cfg.valueChanged.connect(self._schedule_save)
        self.chk_dual_te.toggled.connect(self._schedule_save)
        self.chk_randomize.toggled.connect(self._schedule_save)
        self.ed_seed.textChanged.connect(self._schedule_save)

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
        data = {
            "preset": self._current_preset(),
            "diffusion": str(diffusion),
            "vae": self.cb_vae.currentText(),
            "te1": self.cb_te1.currentText(),
            "te2": self.cb_te2.currentText(),
            "dual_te": self.chk_dual_te.isChecked(),
            "clip_type": self.cb_clip_type.currentText(),
            "merges": json.dumps(
                [{"id": e["id"], "name": e["name"],
                  "models": [[n, w] for n, w in e["models"]],
                  "quant": e["quant"], "low_memory": e["low_memory"]}
                 for e in self._merges], ensure_ascii=False),
            "merge_seq": int(self._merge_seq),
            "width": self.sp_width.value(),
            "height": self.sp_height.value(),
            "steps": self.sp_steps.value(),
            "cfg": float(self.sp_cfg.value()),
            "batch": self.sp_batch.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "randomize": self.chk_randomize.isChecked(),
            "dtype": self.cb_dtype.currentText(),
            "image_format": self.cb_img_format.currentText(),
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

    def _on_backend_ready(self) -> None:
        self.status.showMessage(f"バックエンド準備完了: {self.backend.base_url}")
        self.append_log("バックエンド準備完了")

    def _on_backend_failed(self, msg: str) -> None:
        self.status.showMessage("バックエンドの起動に失敗")
        self.append_log("エラー: " + msg)
        QMessageBox.critical(self, "バックエンドエラー", msg)

    # ----- generation ------------------------------------------------------
    def _seed_value(self) -> int:
        try:
            return int(self.ed_seed.text().strip())
        except ValueError:
            return -1

    def _set_seed(self, v: int) -> None:
        self.ed_seed.setText(str(v))

    def _collect_params(self) -> GenParams:
        te = [self.cb_te1.currentText().strip()]
        if self.chk_dual_te.isChecked() and self.cb_te2.currentText().strip():
            te.append(self.cb_te2.currentText().strip())

        seed = self._seed_value()
        if seed < 0:
            seed = random.randint(0, MAX_SEED)
            self._set_seed(seed)

        entry = self._selected_merge_entry()
        self._last_merge_id = int(entry["id"]) if entry else None
        return GenParams(
            diffusion="" if entry else self.cb_diffusion.currentText().strip(),
            merge_models=list(entry["models"]) if entry else [],
            merge_quant=entry["quant"] if entry else "",
            merge_low_memory=entry["low_memory"] if entry else False,
            vae=self.cb_vae.currentText().strip(),
            te=[t for t in te if t],
            clip_type=self.cb_clip_type.currentText(),
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
        return super().eventFilter(obj, event)

    def on_generate(self) -> None:
        if not self.backend.is_running():
            QMessageBox.warning(self, "未準備", "バックエンドがまだ起動していません。")
            return
        if self._gen_thread is not None or self._xyz_thread is not None:
            return
        if self._merge_selected() and self._selected_merge_entry() is None:
            QMessageBox.warning(self, "マージ未設定",
                                "選択中のマージモデルが見つかりません。")
            self._open_merge_dialog()
            return
        self._merge_was_cached = False
        try:
            params = self._collect_params()
            warn = self._krea2_config_warning(params)
            if warn:
                QMessageBox.warning(self, "Krea-2 の設定を確認してください", warn)
                return
            graph = build_graph(params)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return

        self._last_seed = params.seed
        self._last_params = params
        self._last_graph = graph
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage("生成中…")
        self.append_log(f"生成 seed={params.seed} {params.width}x{params.height}")

        self._gen_thread = QThread(self)
        self._gen_worker = _GenWorker(self.backend, graph)
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.progress.connect(self._on_progress)
        self._gen_worker.preview.connect(self._on_preview_frame)
        self._gen_worker.cached.connect(self._on_cached_nodes)
        self._gen_worker.done.connect(self._on_gen_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()

    def on_cancel(self) -> None:
        # Cancelling also stops continuous mode so the loop clearly ends.
        self.btn_continuous.setChecked(False)
        if self._gen_worker:
            self._gen_worker.cancel()
            self.append_log("キャンセルを要求しました")
        if self._xyz_dlg is not None:
            # XYZ の連続モードもここで止める（キャンセル = ループ終了の合図）。
            try:
                self._xyz_dlg.chk_continuous.setChecked(False)
            except RuntimeError:
                self._xyz_dlg = None
        if self._xyz_worker:
            self._xyz_worker.cancel()
            self.append_log("XYZ プロットのキャンセルを要求しました")

    def _on_progress(self, p: Progress) -> None:
        if p.maximum:
            self.progress.setMaximum(p.maximum)
            self.progress.setValue(p.value)
            self.progress.setFormat(f"{p.note} {p.value}/{p.maximum}")

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
        if self.chk_randomize.isChecked():
            self._set_seed(-1)

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
        for i, data in enumerate(images):
            path = out / f"scom_{stamp}_{self._last_seed}_{i}.{ext}"
            try:
                metadata.save_with_metadata(
                    data, path, fmt, quality, params_text,
                    extra=extra, comfy_prompt=self._last_graph,
                )
                saved.append(path)
            except Exception as e:  # noqa: BLE001
                self.append_log(f"保存に失敗: {e}")
        if saved:
            self.append_log(f"{out} に {len(saved)} 枚保存しました（メタデータ付き）")
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
            # Record the merge recipe so the image stays reproducible.
            model_name = ("merge(" + ", ".join(
                f"{n}:{w:g}" for n, w in p.merge_models) + ")")
            if p.merge_quant:
                model_name += f" {p.merge_quant}"
        else:
            model_name = p.diffusion
        meta = {
            "prompt": p.prompt,
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
        # Continuous mode: as soon as a successful run finishes, start the next.
        if self.btn_continuous.isChecked() and self._last_gen_ok:
            QTimer.singleShot(0, self.on_generate)

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
        # The merge/XYZ windows are parentless (so they can go behind us);
        # close them explicitly or the app would keep running after the main
        # window.
        for dlg in (self._merge_dlg, self._xyz_dlg):
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
