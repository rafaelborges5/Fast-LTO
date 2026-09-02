"""Contract tests for the exported controller-reference CSV.

The CSV in ``export/trajectory.py`` is the interface the car's trajectory
tracker consumes, so its shape is a contract: the column list, one row per
discretization node, and finite values in every numeric cell.  A refactor that
reorders or renames a column, or lets a NaN through, breaks the controller
without breaking any solver test.

Complements ``test_autox_leadin.py``, which checks one derived quantity
(lead-in curvature) rather than the file contract.
"""

from __future__ import annotations

import csv
import io
import math
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

import pytest

from fast_lto.export.trajectory import CSV_COLUMNS
from fast_lto.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def exported_csv(tmp_path_factory) -> List[Dict[str, str]]:
    """Solve the cheapest real lap and export it, once for the module."""
    root = tmp_path_factory.mktemp("export_repo")
    config = PipelineConfig(
        track_id="ellipse",
        track_type="ellipse",
        generate_track=True,
        repo_root=root,
        ds_m=1.5,
        model_name="point_mass",
        mode="trackdrive",
        warm_start="off",
        export_trajectory=True,
        plot_results=False,
        show_plots=False,
    )
    with redirect_stdout(io.StringIO()):
        results = run_pipeline(config, start_from="track", end_at="export")

    with Path(results["export"]).open(newline="") as f:
        return list(csv.DictReader(f))


def test_header_is_the_declared_column_list(exported_csv) -> None:
    assert list(exported_csv[0].keys()) == CSV_COLUMNS


def test_one_row_per_node(exported_csv) -> None:
    # The golden lap discretizes the ellipse into 50 nodes.
    assert len(exported_csv) == 50


def test_every_cell_is_finite(exported_csv) -> None:
    """No NaN or inf reaches the controller."""
    bad = [
        (i, col, row[col])
        for i, row in enumerate(exported_csv)
        for col in CSV_COLUMNS
        if not math.isfinite(float(row[col]))
    ]
    assert not bad, f"non-finite values exported: {bad[:5]}"


def test_arc_progress_and_time_increase(exported_csv) -> None:
    """Both are monotonic along the lap, or the tracker cannot follow them."""
    progress = [float(r["arc_progress"]) for r in exported_csv]
    time = [float(r["time"]) for r in exported_csv]

    assert progress == sorted(progress), "arc_progress must be non-decreasing"
    assert time == sorted(time), "time must be non-decreasing"
    assert time[0] == pytest.approx(0.0), "the reference must start at t=0"


def test_speed_is_positive(exported_csv) -> None:
    for i, row in enumerate(exported_csv):
        assert float(row["velocity_long_ref"]) > 0.0, f"row {i} has non-positive speed"


def test_boundaries_are_signed_room_around_the_trajectory(exported_csv) -> None:
    """``boundary_left``/``boundary_right`` are room left over, not half-widths.

    The exporter writes ``boundary_left = w_left - d`` and
    ``boundary_right = -(w_right + d)``, so they are signed offsets from the
    trajectory: left is non-negative, right is non-positive, and the car sits
    inside the corridor exactly when both signs hold.  A sign flip here would
    steer the tracker into a cone wall, so it is worth pinning.
    """
    for i, row in enumerate(exported_csv):
        left = float(row["boundary_left"])
        right = float(row["boundary_right"])
        assert left >= 0.0, f"row {i}: no room to the left ({left})"
        assert right <= 0.0, f"row {i}: right bound {right} should be non-positive"
        # left - right reconstructs the full corridor width, which must be real.
        assert left - right > 0.0, f"row {i}: corridor collapsed"
