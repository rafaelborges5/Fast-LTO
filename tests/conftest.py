"""Pytest bootstrap: make the ``src`` layout importable as top-level packages.

Mirrors the ``sys.path.insert(0, .../src)`` pattern used throughout src/, since
the project has no installed package (no ``pip install -e .``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
