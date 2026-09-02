"""The solver-profiling / constraint-activity text panel shared by every model.

Each model's panel figure ends with the same summary block: how the solve went,
how the objective split, and how often each constraint was active. It existed as
three near-identical copies, one per model, and they had drifted — the
four-wheel copy had lost the time-per-point and time-per-iteration lines, and
worded the objective split differently, for no reason anyone chose.

Only the constraint rows are genuinely per-model, because only the constraint
set is: pass them as ``activity_rows``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

# (label, key in the constraint-activity dict). Labels are padded to a fixed
# width so the percentages line up in the monospaced panel.
ActivityRows = Sequence[Tuple[str, str]]

_LABEL_WIDTH = 14


def plot_profiling_panel(
    profiling: Optional[Dict],
    constraint_activity: Optional[Dict[str, float]],
    ax: plt.Axes,
    activity_rows: ActivityRows,
) -> None:
    """Render the profiling + constraint-activity summary into ``ax``."""
    ax.axis("off")

    if profiling is None:
        ax.text(
            0.0,
            0.5,
            "No profiling data available.",
            transform=ax.transAxes,
            fontsize=10,
            va="center",
        )
        return

    lines = ["Solver profiling"]

    N = profiling.get("N")
    ds_m = profiling.get("ds_m")
    if N is not None and ds_m is not None:
        lines.append(f"N = {N}, ds = {ds_m:.3f} m")

    solve_time_s = profiling.get("solve_time_s")
    if solve_time_s is not None:
        lines.append(f"Time = {solve_time_s:.3f} s")

    time_per_point_ms = profiling.get("time_per_point_ms")
    if time_per_point_ms is not None:
        lines.append(f"Time / point = {time_per_point_ms:.3f} ms")

    iter_count = profiling.get("iter_count")
    time_per_iter_ms = profiling.get("time_per_iter_ms")
    if time_per_iter_ms is not None and iter_count not in (None, 0):
        lines.append(f"Iterations = {iter_count}, time / iter = {time_per_iter_ms:.3f} ms")
    elif iter_count is not None:
        lines.append(f"Iterations = {iter_count}")

    return_status = profiling.get("return_status")
    if return_status is not None:
        lines.append(f"Status = {return_status}")

    lap_time_s = profiling.get("lap_time_s")
    reg_term = profiling.get("reg_term")
    reg_term_rel = profiling.get("reg_term_relative")
    if lap_time_s is not None or reg_term is not None:
        lines.append("")
        lines.append("Objective split")
        if lap_time_s is not None:
            lines.append(f"Lap-time term = {lap_time_s:.3f} s")
        if reg_term is not None:
            if reg_term_rel not in (None, 0.0):
                lines.append(f"Reg term = {reg_term:.4f} ({reg_term_rel:.2%} of obj)")
            else:
                lines.append(f"Reg term = {reg_term:.4f}")

    lines.append("")
    lines.append("Constraint activity (fraction of lap):")

    if constraint_activity is not None:
        for label, key in activity_rows:
            value = constraint_activity.get(key)
            percent = float(value * 100.0) if value is not None else 0.0
            lines.append(f"{label:<{_LABEL_WIDTH}}≈ {percent:.1f}%")
    else:
        lines.append("(no constraint activity data)")

    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        family="monospace",
    )
