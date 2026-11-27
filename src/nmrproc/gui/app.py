"""Application entry point: ``python -m nmrproc`` or the ``nmrproc`` command."""
from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    os.environ.setdefault("QT_API", "pyside6")  # make matplotlib use PySide6
    from PySide6 import QtWidgets

    from nmrproc.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    window = MainWindow()
    window.show()
    if len(argv) > 1:  # optional: a data folder or a saved session given on the command line
        if argv[1].lower().endswith(".nmrproc"):
            window.open_session(argv[1])
        else:
            window.load_path(argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
