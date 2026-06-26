"""Application entry point."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QDialog

from . import config
from .bootstrap.setup import FirstRunSetup
from .ui.main_window import MainWindow
from .ui.setup_dialog import SetupDialog


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("scom")

    paths = config.AppPaths()
    setup = FirstRunSetup(paths)
    if not setup.is_complete():
        dlg = SetupDialog(setup)
        if dlg.exec() != QDialog.Accepted:
            # User cancelled or setup failed; nothing to run yet.
            return 0

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
