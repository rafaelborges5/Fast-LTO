"""Which steps a run executes, and what it does about the ones it skips.

``start_from`` and ``end_at`` bound the run, and anything before ``start_from``
has to exist on disk already.

These drive the plan directly rather than solving, so they are cheap and say
something about control flow; the golden suite covers the physics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from fast_lto import pipeline
from fast_lto.pipeline import STEP_ORDER, PipelineConfig, run_pipeline


def _config(tmp_path: Path, **kwargs) -> PipelineConfig:
    return PipelineConfig(repo_root=tmp_path, track_id="t", **kwargs)


def _names(config: PipelineConfig) -> List[str]:
    return [step.name for step in pipeline._plan(config)]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_road_course_plans_track_spline_bounds(tmp_path: Path) -> None:
    assert _names(_config(tmp_path, mode="trackdrive")) == list(STEP_ORDER)


def test_skidpad_reaches_the_same_track_by_one_step(tmp_path: Path) -> None:
    """Its path overlaps itself, so it cannot go through spline and bounds."""
    names = _names(_config(tmp_path, mode="skidpad"))

    assert "spline" not in names
    assert "track" not in names
    assert names == ["bounds", "ocp", "export", "plot"]


def test_skidpad_by_track_type_alone_plans_the_same_way(tmp_path: Path) -> None:
    assert _names(_config(tmp_path, track_type="skidpad")) == _names(
        _config(tmp_path, mode="skidpad")
    )


@pytest.mark.parametrize(
    ("flag", "dropped"),
    [("export_trajectory", "export"), ("plot_results", "plot"), ("compute_bounds", "bounds")],
)
def test_a_disabled_step_is_not_planned(tmp_path: Path, flag: str, dropped: str) -> None:
    config = _config(tmp_path, **{flag: False})
    executed = _executed(config)
    assert dropped not in executed


# ---------------------------------------------------------------------------
# Bounds of the run
# ---------------------------------------------------------------------------


def _run_stubbed(config: PipelineConfig, **kwargs):
    """Run the plan with every step stubbed.

    Returns ``(steps that ran, the result paths)``. Stubbing keeps these tests
    about control flow: no spline is fitted and no OCP is solved, so they stay
    in milliseconds and fail for exactly one reason.
    """
    ran: List[str] = []

    def stub(step_name: str):
        def _run(cfg: PipelineConfig, results: Dict[str, Path]) -> Path:
            ran.append(step_name)
            artifact = _ARTIFACTS[step_name](cfg)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if not artifact.suffix:
                artifact.mkdir(parents=True, exist_ok=True)
            else:
                artifact.write_text("stub")
            return artifact

        return _run

    original = pipeline._plan

    def patched(cfg: PipelineConfig):
        steps = original(cfg)
        return [pipeline._Step(s.name, stub(s.name), s.artifact, s.enabled) for s in steps]

    pipeline._plan = patched
    try:
        results = run_pipeline(config, **kwargs)
    finally:
        pipeline._plan = original
    return ran, results


def _executed(config: PipelineConfig, **kwargs) -> List[str]:
    """Just the steps that ran."""
    return _run_stubbed(config, **kwargs)[0]


_ARTIFACTS = {
    "track": lambda c: c.track_csv_path,
    "spline": lambda c: c.discretized_track_path,
    "bounds": lambda c: c.track_with_widths_path,
    "ocp": lambda c: c.solution_path,
    "export": lambda c: c.output_trajectories_dir / "export.csv",
    "plot": lambda c: c.plots_dir / "plots",
}


def test_end_at_stops_the_run(tmp_path: Path) -> None:
    assert _executed(_config(tmp_path), end_at="bounds") == ["track", "spline", "bounds"]


def test_start_from_skips_the_earlier_steps(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _executed(config, end_at="bounds")  # lay down the artifacts

    assert _executed(config, start_from="bounds", end_at="bounds") == ["bounds"]


def test_skipped_steps_are_still_reported(tmp_path: Path) -> None:
    """The caller gets every path, whether or not this run produced it."""
    config = _config(tmp_path)
    _executed(config, end_at="bounds")

    ran, results = _run_stubbed(config, start_from="bounds", end_at="bounds")

    assert ran == ["bounds"], "only the requested step should run"
    assert set(results) == {"track", "spline", "bounds"}
    assert results["spline"] == config.discretized_track_path


def test_a_missing_earlier_artifact_names_the_step_to_rerun(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="Run with start_from='track'"):
        run_pipeline(config, start_from="spline", end_at="spline")


def test_skidpad_honours_start_from(tmp_path: Path) -> None:
    """It used to accept end_at and quietly ignore start_from, so a
    ``--start-from ocp`` skidpad run rebuilt the track it was told to reuse."""
    config = _config(tmp_path, mode="skidpad")
    _executed(config, end_at="bounds")

    assert _executed(config, start_from="ocp", end_at="ocp") == ["ocp"]


# ---------------------------------------------------------------------------
# Bad requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"start_from": "nope"}, {"end_at": "nope"}])
def test_an_unknown_step_is_rejected(tmp_path: Path, kwargs) -> None:
    with pytest.raises(ValueError, match="Must be one of"):
        run_pipeline(_config(tmp_path), **kwargs)


def test_end_before_start_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="comes before"):
        run_pipeline(_config(tmp_path), start_from="ocp", end_at="spline")
