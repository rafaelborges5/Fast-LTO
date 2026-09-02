"""Golden-solution regression tests for the full OCP.

These are the safety net for refactoring: they solve a real (small) lap for each
vehicle model and compare the answer against values recorded in
``tests/data/golden/``.  Anything that silently changes the physics, the
constraint set, the objective or the spline geometry moves these numbers well
outside the tolerance, even though every other test in the suite still passes.

Only solver-independent quantities are compared.  Solve time, iteration count
and time-per-node are deliberately excluded: they are properties of the machine
and the IPOPT build, not of the trajectory.

The solve is bit-identical run to run on one machine; ``RTOL`` exists to absorb
a different BLAS/IPOPT build, and is still ~3 orders of magnitude tighter than
any real behaviour change.

To re-record after an intentional change::

    FAST_LTO_UPDATE_GOLDEN=1 pytest tests/test_golden_solutions.py

then read the diff on the golden file carefully — it is the changelog of what
your change did to the trajectory, and it belongs in the commit.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from fast_lto.pipeline import PipelineConfig, run_pipeline

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden" / "trackdrive_ellipse.json"
UPDATE = os.environ.get("FAST_LTO_UPDATE_GOLDEN") == "1"

MODELS = ["point_mass", "dynamic_bicycle", "four_wheel"]

# Track/solve settings for the golden lap. Coarse on purpose: the point is to
# pin behaviour, not to produce a good racing line, and the whole sweep has to
# stay cheap enough to run in the default suite.
TRACK = {
    "track_type": "ellipse",
    "ds_m": 1.5,
    "continuity": "C2",
    "mode": "trackdrive",
    "integrator_name": "euler",
}

RTOL = 1e-4


def _solve(model_name: str, repo_root: Path) -> Dict[str, Any]:
    """Solve the golden lap for one model under an isolated repo root."""
    config = PipelineConfig(
        track_id="ellipse",
        track_type=TRACK["track_type"],
        generate_track=True,
        repo_root=repo_root,
        ds_m=TRACK["ds_m"],
        continuity=TRACK["continuity"],
        mode=TRACK["mode"],
        model_name=model_name,
        integrator_name=TRACK["integrator_name"],
        # A seed store would make the result depend on test ordering.
        warm_start="off",
        export_trajectory=False,
        plot_results=False,
        show_plots=False,
        solver_verbose=False,
    )
    with redirect_stdout(io.StringIO()):
        results = run_pipeline(config, start_from="track", end_at="ocp")
    return json.loads(Path(results["ocp"]).read_text())


def _metrics(solution: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a solution to the quantities a refactor must not change."""
    profiling = solution["profiling"]
    speed = np.asarray(solution["v"] if "v" in solution else solution["v_long"], dtype=float)
    d = np.asarray(solution["d"], dtype=float)
    arc = np.asarray(solution["arc_lengths"], dtype=float)

    return {
        "return_status": profiling["return_status"],
        "N": int(profiling["N"]),
        "obj_val": float(solution["obj_val"]),
        "lap_time_s": float(profiling["lap_time_s"]),
        "reg_term": float(profiling["reg_term"]),
        "v_min": float(speed.min()),
        "v_max": float(speed.max()),
        "abs_d_mean": float(np.abs(d).mean()),
        "abs_d_max": float(np.abs(d).max()),
        "track_length_m": float(arc[-1]),
    }


@pytest.fixture(scope="module")
def solved(tmp_path_factory) -> Dict[str, Dict[str, Any]]:
    """Solve every model once for the whole module (~6 s total)."""
    root = tmp_path_factory.mktemp("golden_repo")
    return {name: _metrics(_solve(name, root / name)) for name in MODELS}


def test_golden_file_is_current(solved) -> None:
    """Record the goldens when asked, otherwise assert the file matches.

    Runs first so a re-record writes the file before the per-model tests read
    it, and fails loudly afterwards so an accidental update can never pass CI.
    """
    if UPDATE:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "track": TRACK,
            "models": {name: solved[name] for name in MODELS},
        }
        GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        pytest.fail(
            f"Rewrote {GOLDEN_PATH.name} from this run. Review the diff, commit it, "
            "and re-run without FAST_LTO_UPDATE_GOLDEN."
        )

    assert GOLDEN_PATH.is_file(), (
        f"{GOLDEN_PATH} missing. Create it with "
        "FAST_LTO_UPDATE_GOLDEN=1 pytest tests/test_golden_solutions.py"
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    assert (
        golden["track"] == TRACK
    ), "Golden was recorded with different track settings; re-record it."
    assert sorted(golden["models"]) == sorted(MODELS)


@pytest.mark.parametrize("model_name", MODELS)
def test_solution_matches_golden(model_name: str, solved) -> None:
    if UPDATE:
        pytest.skip("re-recording goldens")

    golden = json.loads(GOLDEN_PATH.read_text())["models"][model_name]
    actual = solved[model_name]

    assert actual["return_status"] == golden["return_status"]
    assert actual["N"] == golden["N"]

    for key in (
        "obj_val",
        "lap_time_s",
        "reg_term",
        "v_min",
        "v_max",
        "abs_d_mean",
        "abs_d_max",
        "track_length_m",
    ):
        assert actual[key] == pytest.approx(golden[key], rel=RTOL), (
            f"{model_name}.{key} drifted: {actual[key]!r} vs golden {golden[key]!r}. "
            "If the change was intentional, re-record with FAST_LTO_UPDATE_GOLDEN=1."
        )


@pytest.mark.parametrize("model_name", MODELS)
def test_solution_is_physically_sane(model_name: str, solved) -> None:
    """Cheap invariants that hold regardless of the recorded numbers."""
    actual = solved[model_name]
    assert actual["return_status"] == "Solve_Succeeded"
    assert actual["v_min"] > 0.0, "the car must never stop on a trackdrive lap"
    assert actual["v_max"] >= actual["v_min"]
    assert actual["lap_time_s"] > 0.0
    # The ellipse is ~75 m; staying inside it is a corridor-constraint check.
    assert actual["abs_d_max"] < 5.0
