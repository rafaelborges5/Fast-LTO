"""Compile CasADi functions to a shared library, cached by source hash.

The cache key is a hash of everything that goes into the library — generated
C, extra sources, flags — so any change to the model, its parameters, ``ds``
or the event produces a new library. Nothing is ever invalidated by hand.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import casadi as ca

#: -O1 compiles faster than -O3, and the stage functions ran at the same speed.
DEFAULT_FLAGS = ("-O1",)

CASADI_DIR = Path(ca.__file__).resolve().parent


class CompiledLibrary(NamedTuple):
    path: Path
    compiled: bool  # False on a cache hit
    seconds: float


def cache_dir() -> Path:
    """``$FAST_LTO_CACHE``, else ``~/.cache/fast_lto``."""
    env = os.environ.get("FAST_LTO_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "fast_lto"


def find_compiler() -> str:
    """``$CC``, else the first of ``cc`` / ``gcc`` / ``clang`` on the path."""
    candidates = [os.environ.get("CC")] + ["cc", "gcc", "clang"]
    for cand in candidates:
        if cand and shutil.which(cand):
            return cand
    raise RuntimeError(
        "derivatives='compiled' needs a C compiler (set $CC, or install gcc/clang). "
        "Use derivatives='symbolic' to run without one."
    )


def compile_library(
    name: str,
    functions: Sequence[ca.Function],
    extra_sources: Optional[Dict[str, str]] = None,
    link_fatrop: bool = False,
    flags: Sequence[str] = DEFAULT_FLAGS,
) -> CompiledLibrary:
    """Generate ``functions`` (plus ``extra_sources``) into one cached library."""
    cg = ca.CodeGenerator(f"{name}.c", {"with_header": False})
    for f in functions:
        cg.add(f)
    sources = {f"{name}.c": cg.dump(), **(extra_sources or {})}

    compiler = find_compiler()
    digest = hashlib.sha1()
    for fname in sorted(sources):
        digest.update(fname.encode())
        digest.update(sources[fname].encode())
    digest.update(" ".join([compiler, *flags, str(link_fatrop), ca.__version__]).encode())
    key = digest.hexdigest()[:16]

    root = cache_dir()
    lib = root / f"{name}_{key}.so"
    if lib.exists():
        return CompiledLibrary(lib, False, 0.0)

    t0 = time.perf_counter()
    build = root / f"{name}_{key}"
    build.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for fname, text in sources.items():
        (build / fname).write_text(text)
        paths.append(str(build / fname))

    cmd = [compiler, "-shared", "-fPIC", *flags, *paths]
    if link_fatrop:
        cmd += [
            f"-I{CASADI_DIR / 'include'}",
            f"-L{CASADI_DIR}",
            "-lfatrop",
            "-lblasfeo",
            f"-Wl,-rpath,{CASADI_DIR}",
        ]
    cmd += ["-lm", "-o", f"{lib}.tmp"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Compiling {name} failed:\n{proc.stderr[-2000:]}")
    os.replace(f"{lib}.tmp", lib)
    shutil.rmtree(build, ignore_errors=True)
    return CompiledLibrary(lib, True, time.perf_counter() - t0)
