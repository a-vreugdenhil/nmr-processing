import os

# Qt tests run without a display (works on Windows, macOS and Linux CI machines).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
