"""Where the pipeline reads its inputs and writes its outputs.

Everything hangs off one data root: ``data/tracks``, ``data/solutions``,
``ocp_plots``. In a source checkout that is the checkout; installed from a
wheel there is no checkout to find, so it is the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Overrides the search below with an explicit directory.
DATA_ROOT_ENV_VAR = "FAST_LTO_DATA"


def find_source_checkout(start: Optional[Path] = None) -> Optional[Path]:
    """The Fast-LTO source tree containing ``start``, or None.

    Deliberately stricter than "the nearest directory with a pyproject.toml":
    it also requires ``src/fast_lto``, so it identifies *this* project rather
    than whatever project happens to own the virtualenv the package is
    installed into. That distinction is the whole point -- it is what tells a
    checkout apart from an installation.
    """
    start = Path(start or Path(__file__)).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "fast_lto").is_dir():
            return candidate
    return None


def default_data_root() -> Path:
    """The root to use when the caller named none.

    In order:

    1. ``$FAST_LTO_DATA``, for pointing runs at a scratch disk or a shared
       results directory;
    2. the source checkout this module lives in, so working from a clone
       behaves exactly as it always has -- outputs land in the repo, wherever
       you happen to run from;
    3. the current directory, which is the only sane answer for an installed
       package: results belong to whatever you are working on, not to
       ``site-packages``.
    """
    from_env = os.environ.get(DATA_ROOT_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()

    checkout = find_source_checkout()
    if checkout is not None:
        return checkout

    return Path.cwd()
