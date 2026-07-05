"""Small reusable UI widgets."""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QProgressBar, QProxyStyle,
    QStyle, QTextEdit, QWidget,
)


class WideComboBox(QComboBox):
    """A combo box whose popup grows to fit its longest item.

    The closed control stays at its layout width (long names are elided
    there), but the dropdown list widens so every item is fully readable.
    """

    def showPopup(self) -> None:
        view = self.view()
        fm = view.fontMetrics()
        widest = max((fm.horizontalAdvance(self.itemText(i))
                      for i in range(self.count())), default=0)
        # room for text margins + scrollbar; never narrower than the combo
        needed = widest + view.verticalScrollBar().sizeHint().width() + 24
        screen = self.screen()
        if screen is not None:
            needed = min(needed, int(screen.availableGeometry().width() * 0.8))
        view.setMinimumWidth(max(needed, self.width()))
        super().showPopup()


class CompactSpinStyle(QProxyStyle):
    """Stack a spin box's up/down buttons vertically in a narrow column.

    The Windows 11 style places the two buttons side by side, which squeezes
    the number display; only the subcontrol rects are overridden so the
    native rendering (theme colors, chevron arrows, hover) is kept.
    """
    BTN_W = 16

    def subControlRect(self, cc, opt, sc, widget=None):
        if cc == QStyle.CC_SpinBox:
            r = opt.rect
            if sc == QStyle.SC_SpinBoxUp:
                return QRect(r.right() - self.BTN_W, r.top(),
                             self.BTN_W, r.height() // 2)
            if sc == QStyle.SC_SpinBoxDown:
                return QRect(r.right() - self.BTN_W,
                             r.top() + r.height() // 2,
                             self.BTN_W, r.height() - r.height() // 2)
            if sc == QStyle.SC_SpinBoxEditField:
                return QRect(r.left() + 4, r.top(),
                             r.width() - self.BTN_W - 8, r.height())
        return super().subControlRect(cc, opt, sc, widget)


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
