"""
Plot a track CSV: the two boundaries and the midline between them.

Input format: ``data/tracks/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

_SIDE_TO_BOUNDARY: Dict[str, str] = {"L": "left", "M": "middle", "R": "right"}


def _load_track_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load a track CSV into arrays grouped by boundary name."""

    import csv

    boundaries: Dict[str, list[tuple[float, float]]] = {
        "left": [],
        "middle": [],
        "right": [],
    }

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            boundary = _SIDE_TO_BOUNDARY.get(row["side"].strip().upper())
            if boundary is None:
                continue
            x = float(row["x"])
            y = float(row["y"])
            boundaries[boundary].append((x, y))

    return {name: np.asarray(points, dtype=float) for name, points in boundaries.items()}


def _set_equal_aspect(ax: plt.Axes) -> None:
    """Set equal aspect ratio based on current data limits."""

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    x_centre = 0.5 * (x_min + x_max)
    y_centre = 0.5 * (y_min + y_max)

    ax.set_xlim(x_centre - 0.5 * max_range, x_centre + 0.5 * max_range)
    ax.set_ylim(y_centre - 0.5 * max_range, y_centre + 0.5 * max_range)
    ax.set_aspect("equal", adjustable="box")


def plot_track_csv(
    csv_path: str | Path,
    *,
    title: str | None = None,
    show_points: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Visualise a track CSV containing left / middle / right boundaries.

    Parameters
    ----------
    csv_path:
        Path to the track CSV as produced by `generate_ellipse_track`.
    title:
        Optional plot title. If None, a title is generated from the file name.
    show_points:
        If True, draw small markers on top of the polylines to visualise
        individual sample points.
    show:
        If True, call `plt.show()` before returning.
    save_path:
        Optional path to save the figure as an image (e.g. PNG).

    Returns
    -------
    fig:
        The created Matplotlib Figure.
    """

    csv_path = Path(csv_path)
    boundaries = _load_track_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    colours = {
        "left": "#1f77b4",  # blue
        "middle": "#ff7f0e",  # orange
        "right": "#2ca02c",  # green
    }

    for name, points in boundaries.items():
        if points.size == 0:
            continue
        x = points[:, 0]
        y = points[:, 1]

        # Repeat the first point, to close the loop visually.
        x_line = np.concatenate([x, x[:1]])
        y_line = np.concatenate([y, y[:1]])

        ax.plot(x_line, y_line, color=colours.get(name, "black"), label=f"{name} boundary")
        if show_points:
            ax.scatter(x, y, s=10, color=colours.get(name, "black"), alpha=0.6)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    if title is None:
        title = f"Track: {csv_path.name}"
    ax.set_title(title)

    _set_equal_aspect(ax)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


__all__ = ["plot_track_csv"]


if __name__ == "__main__":
    plot_track_csv("data/tracks/fsg_random.csv")
