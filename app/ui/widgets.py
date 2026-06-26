"""Small reusable UI widgets."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QProgressBar, QTextEdit, QWidget,
)


class GrowingTextEdit(QTextEdit):
    """A plain-text edit that grows its height to fit its content.

    The vertical scrollbar is disabled — the widget resizes instead, so the
    full prompt is always visible without scrolling.
    """

    def __init__(self, parent=None, min_lines: int = 3):
        super().__init__(parent)
        self._min_lines = min_lines
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        # The document lays itself out at the viewport width automatically; this
        # signal fires when content OR width changes. Setting a fixed height
        # does not change the document width, so there is no feedback loop.
        self.document().documentLayout().documentSizeChanged.connect(self._fit)
        self._fit()

    def _fit(self, *args) -> None:
        doc_h = self.document().documentLayout().documentSize().height()
        min_h = self.fontMetrics().lineSpacing() * self._min_lines
        m = self.contentsMargins()
        height = int(max(doc_h, min_h)) + m.top() + m.bottom() \
            + 2 * self.frameWidth() + 4
        if height != self.height():
            self.setFixedHeight(height)


def _make_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 1000)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(16)
    return bar


def _make_status(width: int = 150) -> QLabel:
    status = QLabel("")
    status.setFixedWidth(width)
    status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return status


class _StatusMixin:
    """Status-label state styling. Host must create ``self.status``. Keeps the
    text/color for done/skipped/error/pending/running in one place."""

    def set_running(self, fraction: float | None = None, detail: str = "") -> None:
        self.status.setText(detail or "…")
        self.status.setStyleSheet("")

    def set_done(self, detail: str = "完了") -> None:
        self.status.setText(detail)
        self.status.setStyleSheet("color:#3a3;")

    def set_skipped(self, detail: str = "スキップ") -> None:
        self.status.setText(detail)
        self.status.setStyleSheet("color:#888;")

    def set_error(self, detail: str = "エラー") -> None:
        self.status.setText(detail)
        self.status.setStyleSheet("color:#c33;")

    def set_pending(self, detail: str = "待機中") -> None:
        self.status.setText(detail)
        self.status.setStyleSheet("color:#888;")


class _BarStatusMixin(_StatusMixin):
    """Adds a progress bar driven alongside the status label. Host must also
    create ``self.bar``."""

    def set_running(self, fraction: float | None = None, detail: str = "") -> None:
        if fraction is None:
            self.bar.setRange(0, 0)  # indeterminate / busy animation
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        super().set_running(fraction, detail)

    def set_done(self, detail: str = "完了") -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000)
        super().set_done(detail)

    def set_skipped(self, detail: str = "スキップ") -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000)
        super().set_skipped(detail)

    def set_error(self, detail: str = "エラー") -> None:
        self.bar.setRange(0, 1000)
        super().set_error(detail)

    def set_pending(self, detail: str = "待機中") -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        super().set_pending(detail)


class ProgressRow(_BarStatusMixin, QWidget):
    """One labelled progress bar + status, used for per-component progress."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.name = QLabel(title)
        self.name.setFixedWidth(210)
        self.name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.bar = _make_bar()
        self.status = _make_status()

        layout.addWidget(self.name)
        layout.addWidget(self.bar, stretch=1)
        layout.addWidget(self.status)

    def set_title(self, title: str) -> None:
        self.name.setText(title)


class ModelRow(_StatusMixin, QWidget):
    """One model entry on a single line: a checkbox carrying the model name /
    metadata, with a status column on the right (download figures, 完了, etc.).
    No progress bar — the status text alone reports progress."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.check = QCheckBox(text)
        self.status = _make_status(200)  # room for "13.21 GB / 13.21 GB"

        layout.addWidget(self.check, stretch=1)
        layout.addWidget(self.status)
