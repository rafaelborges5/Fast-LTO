"""The shipped examples must keep running.

``experiments/`` decayed into three modules that no longer import, because
nothing ever executed them: a refactor moved functions out from under them and
no signal fired. Examples are documentation people are invited to run, so they
earn a test rather than the same fate -- one that actually calls the entry
point, and one that checks the property the example exists to demonstrate.

The example is loaded by path, not imported as a package: ``examples/`` is
deliberately not importable library code, and giving it an ``__init__.py`` just
to test it would undo the point of moving it out of ``src/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{name}.py"
    assert path.is_file(), f"{path} is missing"

    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses and pickling can resolve the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frenet() -> ModuleType:
    return _load_example("frenet_frame")


@pytest.fixture(scope="module")
def ellipse_track(tmp_path_factory):
    """A small closed track to exercise the frame on."""
    from fast_lto.splines.spline_fitter import fit_and_discretize
    from fast_lto.tracks.ellipse import generate_ellipse_track

    csv_path = tmp_path_factory.mktemp("frenet_track") / "ellipse.csv"
    generate_ellipse_track(output_csv=csv_path)
    return fit_and_discretize(csv_path, ds_m=1.0, continuity="C2")


def test_frenet_round_trip_recovers_the_offset(frenet, ellipse_track) -> None:
    """xy -> (s, d) -> xy is the identity, which is what makes the frame usable.

    Tolerance is loose because ``project_xy_to_frenet`` snaps to the nearest
    centerline *segment* -- a chord, not the arc -- so a discretized track
    carries a sagitta error of order ds^2 * kappa / 8. It is a property test of
    the mapping, not of the discretization.
    """
    processor = frenet.TrackProcessor(ellipse_track)

    for s in np.linspace(0.0, ellipse_track.total_length_m, 17, endpoint=False):
        for d in (-1.0, -0.25, 0.0, 0.25, 1.0):
            xy = processor.frenet_to_xy(float(s), float(d))
            back = processor.project_xy_to_frenet(xy)

            assert back.d == pytest.approx(d, abs=0.05), f"offset drifted at s={s:.1f}, d={d}"
            # Arc length wraps, so compare on the circle rather than the line.
            gap = abs(back.s - s) % ellipse_track.total_length_m
            gap = min(gap, ellipse_track.total_length_m - gap)
            assert gap < 0.5, f"arc length drifted at s={s:.1f}, d={d}: got {back.s:.2f}"


def test_frenet_reports_curvature_of_the_station_it_snapped_to(frenet, ellipse_track) -> None:
    """The projection carries kappa, which is what D_kappa = 1 - kappa*d needs."""
    processor = frenet.TrackProcessor(ellipse_track)

    projected = processor.project_xy_to_frenet(processor.frenet_to_xy(0.0, 0.0))
    nearest = float(ellipse_track.curvatures[projected.idx])

    assert projected.kappa_s == pytest.approx(nearest, abs=1e-3)


def test_frenet_example_runs_end_to_end(frenet, tmp_path, monkeypatch) -> None:
    """The entry point works headless, from any working directory.

    Guards the two ways an example rots: a stale relative path (it used to
    compute a repo root by counting parents, which the move to examples/ broke)
    and an import that a later refactor removes.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    out = tmp_path / "frenet.png"
    frenet.main(["--out", str(out), "--ds", "8.0"])

    assert out.is_file() and out.stat().st_size > 0
