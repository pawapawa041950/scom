"""PySide6 main window: model selection, generation settings, preview."""
from __future__ import annotations

import random
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRegularExpression, QTimer, QEvent,
)
from PySide6.QtGui import QImage, QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget, QCheckBox,
)

from .. import config, settings, metadata, modelinfo
from . import ansi_log
from .widgets import GrowingTextEdit
from ..comfy_backend import ComfyBackend, BackendError, Progress
from ..workflow import (
    GenParams, build_graph, SAMPLERS, SCHEDULERS,
    CLIP_TYPES_SINGLE, CLIP_TYPES_DUAL, DEFAULT_NEGATIVE,
)

MAX_SEED = 2**63 - 1

# Image format option that disables saving generated images to disk.
NO_SAVE = "保存しない"


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
    done = Signal(list)  # list[bytes]
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
            )
            self.done.emit(images)
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
        self._loading = True
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()
        self.refresh_models()       # fill combos before applying saved selection
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

        # Left column: models, settings, image, action
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_preset_box())
        left_layout.addWidget(self._build_model_box())
        left_layout.addWidget(self._build_settings_box())
        left_layout.addWidget(self._build_image_box())
        left_layout.addWidget(self._build_action_box())
        left_layout.addStretch(1)
        left.setMinimumWidth(380)
        left.setMaximumWidth(480)

        # Middle column: prompt settings
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.addWidget(self._build_prompt_box())
        middle.setMinimumWidth(280)

        # Right column: preview + log
        right = QWidget()
        right_layout = QVBoxLayout(right)
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
        splitter.addWidget(middle)
        splitter.addWidget(right)
        splitter.setStretchFactor(2, 1)  # preview/log column expands
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("バックエンドを起動中…")

    def _build_preset_box(self) -> QWidget:
        """Preset selector above the Models box. Picking anima/krea2 filters the
        model dropdowns to that family and auto-selects its model/vae/te/CLIP."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("anima", "anima")
        self.cb_preset.addItem("krea2", "krea2")
        self.cb_preset.addItem("すべて", "all")
        self.cb_preset.setCurrentIndex(2)  # default: すべて (show everything)
        self.cb_preset.setToolTip(
            "anima / krea2 を選ぶと、その構成に合うモデル・VAE・"
            "Text encoder・CLIP type に絞り込み＆自動選択します"
        )
        self.cb_preset.currentIndexChanged.connect(self._on_preset_changed)
        lbl = QLabel("Preset")
        f = lbl.font(); f.setBold(True); lbl.setFont(f)
        row.addWidget(lbl)
        row.addWidget(self.cb_preset, stretch=1)
        return w

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Models")
        form = QFormLayout(box)

        self.cb_diffusion = QComboBox()
        self.cb_vae = QComboBox()
        self.cb_te1 = QComboBox()
        self.cb_te2 = QComboBox()
        self.chk_dual_te = QCheckBox("2つ目の text encoder を使用 (DualCLIPLoader)")
        self.chk_dual_te.toggled.connect(self._on_dual_toggled)

        self.cb_clip_type = QComboBox()
        self.cb_clip_type.addItems(CLIP_TYPES_SINGLE)

        # Picking a model auto-aligns the CLIP type / text encoder to its family
        # (e.g. a krea2 diffusion needs CLIP type 'krea2' + a Qwen3-VL encoder).
        self.cb_diffusion.currentTextChanged.connect(self._on_diffusion_changed)

        refresh = QPushButton("再スキャン")
        refresh.clicked.connect(self.refresh_models)
        manage = QPushButton("モデル管理…")
        manage.clicked.connect(self.open_models_dialog)
        btn_row = QHBoxLayout()
        btn_row.addWidget(refresh)
        btn_row.addWidget(manage)
        btns = QWidget(); btns.setLayout(btn_row)
        btn_row.setContentsMargins(0, 0, 0, 0)

        form.addRow("Diffusion:", self.cb_diffusion)
        form.addRow("VAE:", self.cb_vae)
        form.addRow("Text encoder 1:", self.cb_te1)
        form.addRow("", self.chk_dual_te)
        form.addRow("Text encoder 2:", self.cb_te2)
        form.addRow("CLIP type:", self.cb_clip_type)
        form.addRow("", btns)
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

        self.cb_sampler = QComboBox(); self.cb_sampler.addItems(SAMPLERS)
        self.cb_sampler.setCurrentText("er_sde")
        self.cb_scheduler = QComboBox(); self.cb_scheduler.addItems(SCHEDULERS)
        self.cb_scheduler.setCurrentText("simple")

        # Seeds can exceed 32-bit (QSpinBox limit), so use a text field.
        self.ed_seed = QLineEdit("-1")
        self.ed_seed.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"-1|\d{1,19}"))
        )
        self.ed_seed.setToolTip("-1 = 毎回ランダム")
        self.chk_randomize = QCheckBox("生成ごとに seed をランダム化")
        self.chk_randomize.setChecked(True)

        self.cb_dtype = QComboBox()
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
        self.btn_continuous = QPushButton("連続")
        self.btn_continuous.setCheckable(True)
        self.btn_continuous.setMinimumHeight(40)
        self.btn_continuous.setToolTip("ONの間、生成が終わるたびに自動で次を生成します")
        self.btn_generate = QPushButton("生成")
        self.btn_generate.setMinimumHeight(40)
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        row.addWidget(self.btn_continuous, stretch=1)
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

        self.cb_img_format = QComboBox()
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
        if self._diffusion_family(params.diffusion) != "krea2":
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
        self._fill_combo(self.cb_diffusion, self._filter_for_preset(
            "diffusion_models", self._all_models["diffusion_models"], preset))
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
        self.cb_diffusion.setCurrentText(str(s.get("diffusion", "")))
        self.cb_vae.setCurrentText(str(s.get("vae", "")))
        self.cb_te1.setCurrentText(str(s.get("te1", "")))
        self.cb_te2.setCurrentText(str(s.get("te2", "")))
        self.chk_dual_te.setChecked(bool(s.get("dual_te", False)))
        self._on_dual_toggled(self.chk_dual_te.isChecked())  # sync clip list
        self.cb_clip_type.setCurrentText(str(s.get("clip_type", "stable_diffusion")))
        # prompt/negative: initial values only
        self.txt_prompt.setPlainText(str(s.get("prompt", "")))
        self.txt_negative.setPlainText(str(s.get("negative", DEFAULT_NEGATIVE)))
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
        data = {
            "preset": self._current_preset(),
            "diffusion": self.cb_diffusion.currentText(),
            "vae": self.cb_vae.currentText(),
            "te1": self.cb_te1.currentText(),
            "te2": self.cb_te2.currentText(),
            "dual_te": self.chk_dual_te.isChecked(),
            "clip_type": self.cb_clip_type.currentText(),
            "prompt": self.settings.get("prompt", ""),
            "negative": self.settings.get("negative", DEFAULT_NEGATIVE),
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

        return GenParams(
            diffusion=self.cb_diffusion.currentText().strip(),
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
        if self._gen_thread is not None:
            return
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
        if images:
            self._show_image(images[0])
            self._save_outputs(images)
        if self.chk_randomize.isChecked():
            self._set_seed(-1)

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

    def _build_metadata(self) -> tuple[str, dict]:
        """Build the parameters string + structured dict from the last run."""
        p = self._last_params
        if p is None:
            return "", {}
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
            "model": p.diffusion,
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
        self.append_log("バックエンドを終了中…")
        if self._gen_worker:
            self._gen_worker.cancel()
        self.backend.stop()
        super().closeEvent(event)
