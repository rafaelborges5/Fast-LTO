"""Golden-solution regression tests for the full OCP.

These are the safety net for refactoring: they solve a real (small) problem for
each vehicle model in each event mode and compare the answer against values
recorded in ``tests/data/golden/``.  Anything that silently changes the physics,
the constraint set, the objective or the spline geometry moves these numbers
well outside the tolerance, even though every other test in the suite still
passes.

One scenario per event mode, because the three modes exercise genuinely
different machinery: trackdrive closes the loop, autox extends the track past a
timing gate and pins a standing start, and skidpad builds its track from a cone
map and weights the objective from a timed mask.  A refactor that only ever ran
the trackdrive scenario would not notice breaking either of the others.

Only solver-independent quantities are compared.  Solve time, iteration count
and time-per-node are deliberately excluded: they are properties of the machine
and the IPOPT build, not of the trajectory.

The solve is bit-identical run to run on one machine; ``RTOL`` exists to absorb
a different BLAS/IPOPT build, and is still ~3 orders of magnitude tighter than
any real behaviour change.

The full sweep is ~50 s, most of it the four-wheel skidpad solve.  It runs by
default, because a safety net you have to opt into is not a safety net.  For a
faster inner loop while iterating locally::

    pytest -m "not slow"

To re-record after an intentional change::

    FAST_LTO_UPDATE_GOLDEN=1 pytest tests/test_golden_solutions.py

then read the diff on the golden files carefully — it is the changelog of what
your change did to the trajectory, and it belongs in the commit.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Tuple

import numpy as np
import pytest

from fast_lto.config import build_pipeline_config
from fast_lto.pipeline import run_pipeline

GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden"
REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATE = os.environ.get("FAST_LTO_UPDATE_GOLDEN") == "1"

MODELS = ("point_mass", "dynamic_bicycle", "four_wheel")

RTOL = 1e-4


@dataclass(frozen=True)
class Scenario:
    """One recorded problem: what to solve, and what to record about it."""

    key: str

    #: The settings that define the problem.  Recorded into the golden file and
    #: compared on load, so a golden can never be read back against a different
    #: question than the one it answers.
    settings: Dict[str, Any]

    #: Extra pipeline arguments that do *not* define the problem — absolute
    #: paths to input data, which are machine-specific and must stay out of the
    #: recorded file.
    inputs: Dict[str, Any] = field(default_factory=dict)

    #: Profiling keys worth pinning for this mode on top of the shared metrics.
    extra_metrics: Tuple[str, ...] = ()

    #: Models whose solve is slow enough to be worth an opt-out marker.
    slow_models: FrozenSet[str] = frozenset()

    @property
    def golden_path(self) -> Path:
        return GOLDEN_DIR / f"{self.key}.json"


SCENARIOS = {
    scenario.key: scenario
    for scenario in (
        Scenario(
            key="trackdrive_ellipse",
            settings={
                "track_id": "ellipse",
                "track_type": "ellipse",
                "ds_m": 1.5,
                "continuity": "C2",
                "mode": "trackdrive",
                "integrator_name": "euler",
            },
        ),
        Scenario(
            key="autox_ellipse",
            settings={
                "track_id": "ellipse",
                "track_type": "ellipse",
                "ds_m": 1.5,
                "continuity": "C2",
                "mode": "autox",
                "integrator_name": "euler",
                # Short on purpose: enough run-off to reach the timing gate and
                # brake into, without doubling the node count.
                "autox": {"extension_m": 10.0},
            },
            extra_metrics=("autox_lap_time_s",),
        ),
        Scenario(
            key="skidpad",
            settings={
                "track_id": "skidpad",
                "track_type": "skidpad",
                "ds_m": 1.5,
                "mode": "skidpad",
                "integrator_name": "euler",
            },
            inputs={
                "skidpad": {
                    "map_csv": str(REPO_ROOT / "data/tracks/skidpad/skidpad_map.csv"),
                    "reference_csv": str(REPO_ROOT / "data/tracks/skidpad/skidpad_reference.csv"),
                }
            },
            extra_metrics=("skidpad_score_s", "pure_timed_time_s"),
            slow_models=frozenset({"four_wheel"}),
        ),
    )
}

CASES = [(scenario.key, model) for scenario in SCENARIOS.values() for model in MODELS]


def _case_id(case: Tuple[str, str]) -> str:
    return f"{case[0]}-{case[1]}"


def _case_params():
    """Parametrisation carrying each scenario's own slow markers."""
    for scenario_key, model in CASES:
        marks = []
        if model in SCENARIOS[scenario_key].slow_models:
            marks.append(pytest.mark.slow)
        yield pytest.param(scenario_key, model, id=f"{scenario_key}-{model}", marks=marks)


def _solve(scenario: Scenario, model_name: str, repo_root: Path) -> Dict[str, Any]:
    """Solve one scenario for one model under an isolated repo root."""
    # Through build_pipeline_config so a scenario's event block is validated and
    # built the same way a config file's is, rather than by a second code path.
    clash = set(scenario.settings) & set(scenario.inputs)
    assert not clash, f"{scenario.key} splits {sorted(clash)} across settings and inputs"

    config = build_pipeline_config(
        {
            **scenario.settings,
            **scenario.inputs,
            "repo_root": repo_root,
            "model_name": model_name,
            "generate_track": True,
            # A seed store would make the result depend on test ordering.
            "warm_start": "off",
            "export_trajectory": False,
            "plot_results": False,
            "show_plots": False,
            "solver_verbose": False,
        }
    )
    with redirect_stdout(io.StringIO()):
        results = run_pipeline(config, start_from="track", end_at="ocp")
    return json.loads(Path(results["ocp"]).read_text())


def _metrics(scenario: Scenario, solution: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a solution to the quantities a refactor must not change."""
    profiling = solution["profiling"]
    speed = np.asarray(solution["v"] if "v" in solution else solution["v_long"], dtype=float)
    d = np.asarray(solution["d"], dtype=float)
    arc = np.asarray(solution["arc_lengths"], dtype=float)

    metrics = {
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
    for key in scenario.extra_metrics:
        value = profiling.get(key)
        assert value is not None, f"{scenario.key} recorded no {key!r}; the mode changed"
        metrics[key] = float(value)
    return metrics


@pytest.fixture(scope="session")
def solve_case(tmp_path_factory):
    """Solve a (scenario, model) pair once per session, on demand.

    Lazy rather than eager so that deselecting the slow cases actually skips
    their solves instead of paying for them in a fixture nobody reads.
    """
    root = tmp_path_factory.mktemp("golden_repo")
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _get(scenario_key: str, model_name: str) -> Dict[str, Any]:
        key = (scenario_key, model_name)
        if key not in cache:
            scenario = SCENARIOS[scenario_key]
            solution = _solve(scenario, model_name, root / f"{scenario_key}_{model_name}")
            cache[key] = _metrics(scenario, solution)
        return cache[key]

    return _get


@pytest.mark.skipif(not UPDATE, reason="set FAST_LTO_UPDATE_GOLDEN=1 to re-record")
def test_rerecord_goldens(solve_case) -> None:
    """Rewrite every golden file, then fail loudly.

    Failing on success is deliberate: a re-record must never be mistaken for a
    passing run, and the diff has to be read before it is committed.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS.values():
        payload = {
            "settings": scenario.settings,
            "models": {model: solve_case(scenario.key, model) for model in MODELS},
        }
        scenario.golden_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    pytest.fail(
        f"Rewrote {len(SCENARIOS)} golden files. Review the diff, commit it, "
        "and re-run without FAST_LTO_UPDATE_GOLDEN."
    )


@pytest.mark.parametrize("scenario_key", sorted(SCENARIOS))
def test_golden_file_answers_this_question(scenario_key: str) -> None:
    """The recorded file must describe the problem we are about to solve."""
    if UPDATE:
        pytest.skip("re-recording goldens")

    scenario = SCENARIOS[scenario_key]
    assert scenario.golden_path.is_file(), (
        f"{scenario.golden_path} missing. Create it with "
        "FAST_LTO_UPDATE_GOLDEN=1 pytest tests/test_golden_solutions.py"
    )

    golden = json.loads(scenario.golden_path.read_text())
    assert golden["settings"] == scenario.settings, (
        f"{scenario.key} golden was recorded with different settings; re-record it. "
        f"recorded {golden['settings']!r} vs current {scenario.settings!r}"
    )
    assert sorted(golden["models"]) == sorted(MODELS)


@pytest.mark.parametrize(("scenario_key", "model_name"), list(_case_params()))
def test_solution_matches_golden(scenario_key: str, model_name: str, solve_case) -> None:
    if UPDATE:
        pytest.skip("re-recording goldens")

    scenario = SCENARIOS[scenario_key]
    golden = json.loads(scenario.golden_path.read_text())["models"][model_name]
    actual = solve_case(scenario_key, model_name)

    assert actual["return_status"] == golden["return_status"]
    assert actual["N"] == golden["N"]

    numeric = [key for key in golden if key not in ("return_status", "N")]
    assert numeric, "golden recorded nothing comparable"

    for key in numeric:
        assert actual[key] == pytest.approx(golden[key], rel=RTOL), (
            f"{scenario_key}/{model_name}.{key} drifted: "
            f"{actual[key]!r} vs golden {golden[key]!r}. "
            "If the change was intentional, re-record with FAST_LTO_UPDATE_GOLDEN=1."
        )


@pytest.mark.parametrize(("scenario_key", "model_name"), list(_case_params()))
def test_solution_is_physically_sane(scenario_key: str, model_name: str, solve_case) -> None:
    """Cheap invariants that hold regardless of the recorded numbers."""
    if UPDATE:
        pytest.skip("re-recording goldens")

    actual = solve_case(scenario_key, model_name)

    assert actual["return_status"] == "Solve_Succeeded"
    assert actual["v_min"] > 0.0, "every model bounds speed away from zero"
    assert actual["v_max"] >= actual["v_min"]
    assert actual["lap_time_s"] > 0.0
    # Both shipped tracks are a few metres wide; staying inside is a
    # corridor-constraint check.
    assert actual["abs_d_max"] < 5.0


def test_autox_timed_lap_is_shorter_than_the_full_maneuver(solve_case) -> None:
    """The autox gate-to-gate time must exclude the run-off it brakes into.

    Pins the thing autox exists to compute: ``lap_time_s`` covers the whole
    horizon including the post-finish extension, while ``autox_lap_time_s`` is
    measured between the two timing-gate crossings.
    """
    if UPDATE:
        pytest.skip("re-recording goldens")

    for model in MODELS:
        actual = solve_case("autox_ellipse", model)
        assert 0.0 < actual["autox_lap_time_s"] < actual["lap_time_s"], (
            f"{model}: autox_lap_time_s={actual['autox_lap_time_s']!r} is not a "
            f"strict subset of lap_time_s={actual['lap_time_s']!r}"
        )


# ---------------------------------------------------------------------------
# Cross-model parameter agreement
# ---------------------------------------------------------------------------


SHARED_GEOMETRY_KEYS = ["m", "g", "lf", "lr"]


def test_models_agree_on_the_car_they_describe() -> None:
    """Every model's defaults must describe the same physical car.

    They did not: ``dynamic_bicycle`` had ``lf``/``lr`` reversed relative to
    ``four_wheel`` and every shipped config, so building it without a config
    gave a car with 55% front weight instead of 45%. Nothing caught it, because
    the configs set both values and papered over the defaults.
    """
    from fast_lto.pipeline import _make_model

    defaults = {name: _make_model(name).get_default_params() for name in MODELS}

    for key in SHARED_GEOMETRY_KEYS:
        values = {name: params[key] for name, params in defaults.items() if key in params}
        assert len(set(values.values())) == 1, f"models disagree on {key!r}: {values}"
