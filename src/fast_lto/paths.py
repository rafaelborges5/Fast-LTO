"""Data-root resolution for pipeline inputs and outputs.

Tracks, solutions, and plots hang off one root: ``$FAST_LTO_DATA`` if set,
else the source checkout containing this package, else ``cwd``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Explicit data-root override.
DATA_ROOT_ENV_VAR = "FAST_LTO_DATA"


def find_source_checkout(start: Optional[Path] = None) -> Optional[Path]:
    """Nearest ancestor of ``start`` with ``pyproject.toml`` and ``src/fast_lto``."""
    start = Path(start or Path(__file__)).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "fast_lto").is_dir():
            return candidate
    return None


def default_data_root() -> Path:
    """Data root when the caller does not set one.

    Order: ``$FAST_LTO_DATA``, source checkout, then ``cwd``.
    """
    from_env = os.environ.get(DATA_ROOT_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()

    checkout = find_source_checkout()
    if checkout is not None:
        return checkout

    return Path.cwd()
