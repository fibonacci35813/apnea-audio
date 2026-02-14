"""Streamlit Cloud entry point — delegates to the package UI."""

import sys
from pathlib import Path

# Ensure the src directory is on the import path so that
# ``apnea_screen`` can be imported as a top-level package.
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Now launch the real app
from apnea_screen.app import main  # noqa: E402

main()
