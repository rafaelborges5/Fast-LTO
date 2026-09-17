"""The shared profiling panel, and the per-model rows each module feeds it.

The panel looks up constraint activity by string key and renders a missing one
as ``0.0%`` -- right for a model that has no such constraint, but it makes a
typo in a row key invisible. These tie each module's rows to the keys its own
activity computer produces.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from fast_lto.visualization import (  # noqa: E402
    ocp_plots,
    ocp_plots_dynamic_bicycle,
    ocp_plots_four_wheel,
)
from fast_lto.visualization.summary_panel import plot_profiling_panel  # noqa: E402

MODULES = {
    "point_mass": ocp_plots,
    "dynamic_bicycle": ocp_plots_dynamic_bicycle,
    "four_wheel": ocp_plots_four_wheel,
}

# The activity dict each module's computer builds, by the keys it writes.
PRODUCED_KEYS = {
    "point_mass": {
        "track_bounds_active",
        "friction_circle_active",
        "a_long_bounds_active",
        "a_lat_bounds_active",
        "v_bounds_active",
    },
    "dynamic_bicycle": {
        "track_bounds_active",
        "gg_active",
        "a_long_bounds_active",
        "delta_bounds_active",
        "v_bounds_active",
    },
    "four_wheel": {
        "track_bounds_active",
        "friction_fl_active",
        "friction_fr_active",
        "friction_rr_active",
        "friction_rl_active",
        "force_bounds_active",
        "speed_bounds_active",
    },
}


class _RecordingAxes:
    """Just enough of a matplotlib Axes to capture the rendered text."""

    def __init__(self) -> None:
        self.text_calls: List[str] = []

    def axis(self, *args, **kwargs) -> None:
        pass

    def text(self, x, y, s, **kwargs) -> None:
        self.text_calls.append(s)

    @property
    def transAxes(self):
        return None

    @property
    def rendered(self) -> str:
        return "\n".join(self.text_calls)


def _profiling() -> Dict:
    return {
        "N": 50,
        "ds_m": 1.5,
        "solve_time_s": 1.0,
        "iter_count": 10,
        "return_status": "Solve_Succeeded",
        "time_per_point_ms": 2.0,
        "time_per_iter_ms": 3.0,
        "lap_time_s": 7.5,
        "reg_term": 2.0,
        "reg_term_relative": 0.25,
    }


@pytest.mark.parametrize("model_name", sorted(MODULES))
def test_every_row_key_is_one_the_model_actually_produces(model_name: str) -> None:
    rows = MODULES[model_name].ACTIVITY_ROWS
    unknown = {key for _, key in rows} - PRODUCED_KEYS[model_name]
    assert not unknown, (
        f"{model_name} rows reference keys its activity computer never sets: "
        f"{sorted(unknown)} — these would silently render as 0.0%"
    )


@pytest.mark.parametrize("model_name", sorted(MODULES))
def test_rows_are_unique_and_non_empty(model_name: str) -> None:
    rows = MODULES[model_name].ACTIVITY_ROWS
    assert rows, f"{model_name} renders no constraint rows"
    keys = [key for _, key in rows]
    assert len(keys) == len(set(keys)), f"{model_name} repeats a row key"


@pytest.mark.parametrize("model_name", sorted(MODULES))
def test_panel_renders_a_row_per_constraint(model_name: str) -> None:
    rows = MODULES[model_name].ACTIVITY_ROWS
    activity = {key: 0.25 for _, key in rows}

    ax = _RecordingAxes()
    plot_profiling_panel(_profiling(), activity, ax, activity_rows=rows)

    rendered = ax.rendered
    for label, _ in rows:
        assert label in rendered
    assert rendered.count("25.0%") == len(rows)
    assert "Solve_Succeeded" in rendered


def test_missing_profiling_says_so_instead_of_rendering_an_empty_panel() -> None:
    ax = _RecordingAxes()
    plot_profiling_panel(None, None, ax, activity_rows=ocp_plots.ACTIVITY_ROWS)
    assert "No profiling data available." in ax.rendered


def test_missing_activity_is_labelled_not_rendered_as_zeros() -> None:
    """Absent data and genuinely-inactive constraints must not look the same."""
    ax = _RecordingAxes()
    plot_profiling_panel(_profiling(), None, ax, activity_rows=ocp_plots.ACTIVITY_ROWS)
    assert "(no constraint activity data)" in ax.rendered
    assert "0.0%" not in ax.rendered
