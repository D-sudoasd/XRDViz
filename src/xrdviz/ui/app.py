from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    from xrdviz.ui.main_window import MainWindow

    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    window.resize(1280, 760)
    window.show()
    return app.exec()
