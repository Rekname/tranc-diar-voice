import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import core.quiet  # noqa: F401
from ui.qt_ui import create_ui


def main():
    app, window = create_ui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
