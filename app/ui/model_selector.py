"""Reusable model-selection widget shared by the first-run setup and the model
manager, so the two screens don't drift apart.

It builds the manifest-driven checkbox list plus the quick-select preset
buttons. With ``with_progress=True`` it also adds a per-file ProgressRow under
each entry (used by the model manager while downloading); the setup screen uses
it purely for selection (``with_progress=False``).
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from .. import config
from ..bootstrap import models as models_mod
from .widgets import ModelRow

# Quick-select presets: which files a button ticks for a given setup.
PRESET_ANIMA = [
    "anima-base-v1.0.safetensors",
    "qwen_image_vae.safetensors",
    "qwen_3_06b_base.safetensors",
]
PRESET_KREA2_FP8 = [
    "krea2_turbo_mxfp8.safetensors",
    "qwen_image_vae.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors",
]
QUICK_PRESETS = [
    ("anima 必須モデル", PRESET_ANIMA),
    ("krea2 必須モデル fp8 (24GB VRAM目安)", PRESET_KREA2_FP8),
]


def fmt_size(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n} B"


class ModelSelector(QWidget):
    """Checkbox list of manifest models + quick-select buttons.

    ``checks``/``rows``/``manifest`` are exposed so an embedding dialog can
    drive downloads and per-file progress.
    """

    def __init__(self, paths: config.AppPaths, with_progress: bool = False,
                 parent=None):
        super().__init__(parent)
        self._paths = paths
        self._with_progress = with_progress
        self.manifest = models_mod.load_manifest(paths)
        self.checks: dict[str, QCheckBox] = {}
        self.rows: dict[str, ModelRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        quick = QGroupBox("クイック選択")
        quick_layout = QVBoxLayout(quick)
        quick_layout.addWidget(QLabel(
            "チェックした一式をまとめてダウンロード対象にします"))
        self._quick: list[tuple[QCheckBox, list]] = []
        for label, fileset in QUICK_PRESETS:
            qcb = QCheckBox(label)
            qcb.toggled.connect(
                lambda checked, s=fileset: self._on_quick_toggled(s, checked))
            quick_layout.addWidget(qcb)
            self._quick.append((qcb, fileset))
        layout.addWidget(quick)

        box = QGroupBox("Models")
        self._box_layout = QVBoxLayout(box)
        self._populate()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        layout.addWidget(scroll, stretch=1)

    def _is_present(self, m) -> bool:
        p = models_mod.target_path(self._paths, m)
        return p.exists() and (not m.size or p.stat().st_size == m.size)

    def _populate(self) -> None:
        for m in self.manifest:
            present = self._is_present(m)
            tag = "必須" if m.required else "任意"
            label = f"{m.filename}   [{m.kind}, {fmt_size(m.size)}, {tag}]"
            # With progress, the checkbox and its bar share one line (ModelRow);
            # otherwise it's just a checkbox.
            if self._with_progress:
                # Status column shows "ダウンロード済み"; don't repeat it in the
                # checkbox text (it would show twice when the window is wide).
                row = ModelRow(label)
                cb = row.check
                if present:
                    cb.setEnabled(False)
                    row.set_skipped("ダウンロード済み")
                self.rows[m.filename] = row
                self._box_layout.addWidget(row)
            else:
                # No status column here, so mark the checkbox text instead.
                cb = QCheckBox(label)
                if present:
                    cb.setEnabled(False)
                    cb.setText(label + "  ✓ ダウンロード済み")
                # nothing pre-selected; user opts in
            self.checks[m.filename] = cb
            if not self._with_progress:
                self._box_layout.addWidget(cb)

    def _on_quick_toggled(self, files: list, checked: bool) -> None:
        """Union (OR) selection: checking a preset ticks its files; unchecking
        unticks only the files no other still-checked preset needs — so the
        shared VAE survives until both presets are off.
        """
        if checked:
            for fn in files:
                cb = self.checks.get(fn)
                if cb is not None and cb.isEnabled():
                    cb.setChecked(True)
            return
        keep = set()
        for qcb, f in self._quick:
            if qcb.isChecked():
                keep |= set(f)
        for fn in files:
            if fn in keep:
                continue
            cb = self.checks.get(fn)
            if cb is not None and cb.isEnabled():
                cb.setChecked(False)

    def selected_filenames(self) -> list:
        """Filenames the user ticked (excludes already-downloaded ones)."""
        return [fn for fn, cb in self.checks.items()
                if cb.isChecked() and cb.isEnabled()]
