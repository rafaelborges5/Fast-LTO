"""
Compute lateral box constraints (w_left, w_right) for a discretized track.

Inputs:
- DiscretizedTrack JSON (centerline samples with headings/normals)
- Track CSV with left/right boundary polylines (progress-ordered)

Outputs:
- Optional JSON with full track data plus widths (for downstream use)
- Optional visualization overlaying normals and intersections

Note: Current intersection search is O(N*M) over center samples (N) and
boundary segments (M). # TODO: accelerate with spatial indexing / local
windowing if needed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from ..splines.discretized_track import DiscretizedTrack  # type: ignore
except ImportError:
    # Fallback for script-style imports when package context is missing
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from splines.discretized_track import DiscretizedTrack  # type: ignore


@dataclass
class LateralBoundsResult:
    w_left: np.ndarray
    w_right: np.ndarray
    misses_left: int
    misses_right: int


def load_boundaries(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load left/right/middle polylines from the track CSV."""
    points: Dict[str, list[Tuple[int, float, float]]] = {"left": [], "middle": [], "right": []}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = row["boundary"].strip().lower()
            if b not in points:
                continue
            idx = int(row["index"])
            x = float(row["x"])
            y = float(row["y"])
            points[b].append((idx, x, y))

    out: Dict[str, np.ndarray] = {}
    for key, arr in points.items():
        arr.sort(key=lambda p: p[0])
        out[key] = np.array([[p[1], p[2]] for p in arr], dtype=np.float64)
    return out


def _ray_segment_intersection(p: np.ndarray, n_hat: np.ndarray, q0: np.ndarray, q1: np.ndarray):
    """
    Solve p + t*n_hat = q0 + u*(q1-q0). Returns (t, u) or (None, None) if no hit.
    """
    v = q1 - q0
    A = np.array([[n_hat[0], -v[0]], [n_hat[1], -v[1]]], dtype=np.float64)
    b = q0 - p
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None, None
    inv_det = 1.0 / det
    t = inv_det * (b[0] * A[1, 1] - b[1] * A[0, 1])
    u = inv_det * (-b[0] * A[1, 0] + b[1] * A[0, 0])
    return t, u


def compute_lateral_bounds(track: DiscretizedTrack, left: np.ndarray, right: np.ndarray) -> LateralBoundsResult:
    """
    For each center sample, find intersections of its normal with left/right polylines.
    Returns w_left (positive along +n) and w_right (positive along -n).
    """
    n_samples = track.num_points
    w_left = np.full(n_samples, np.nan, dtype=np.float64)
    w_right = np.full(n_samples, np.nan, dtype=np.float64)

    def _segments(poly: np.ndarray) -> list[Tuple[np.ndarray, np.ndarray]]:
        segs = list(zip(poly[:-1], poly[1:]))
        # Close the loop
        segs.append((poly[-1], poly[0]))
        return segs

    left_segs = _segments(left)
    right_segs = _segments(right)

    misses_left = 0
    misses_right = 0

    tangents = np.column_stack((np.cos(track.headings), np.sin(track.headings)))
    normals = np.column_stack((-np.sin(track.headings), np.cos(track.headings)))

    for i in range(n_samples):
        p = track.positions[i]
        n_hat = normals[i]

        # Left: t > 0
        t_min = None
        for q0, q1 in left_segs:
            t, u = _ray_segment_intersection(p, n_hat, q0, q1)
            if t is None or u is None:
                continue
            if u < -1e-9 or u > 1 + 1e-9:
                continue
            if t <= 1e-9:
                continue
            if (t_min is None) or (t < t_min):
                t_min = t
        if t_min is None:
            misses_left += 1
        else:
            w_left[i] = t_min

        # Right: t < 0 (we store positive magnitude)
        t_min = None
        for q0, q1 in right_segs:
            t, u = _ray_segment_intersection(p, n_hat, q0, q1)
            if t is None or u is None:
                continue
            if u < -1e-9 or u > 1 + 1e-9:
                continue
            if t >= -1e-9:
                continue
            if (t_min is None) or (-t < t_min):
                t_min = -t  # store as positive distance
        if t_min is None:
            misses_right += 1
        else:
            w_right[i] = t_min

    return LateralBoundsResult(w_left=w_left, w_right=w_right, misses_left=misses_left, misses_right=misses_right)


def save_bounds_json(path: Path, track: DiscretizedTrack, csv_source: Path, result: LateralBoundsResult) -> None:
    data = {
        "source_boundaries_csv": str(csv_source),
        "discretized_track": str(path),
        "w_left": result.w_left.tolist(),
        "w_right": result.w_right.tolist(),
        "num_points": int(track.num_points),
        "misses_left": int(result.misses_left),
        "misses_right": int(result.misses_right),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_track_with_widths(path: Path, track: DiscretizedTrack, csv_source: Path, result: LateralBoundsResult) -> None:
    """
    Save a combined object that includes the discretized track plus lateral widths.
    """
    data = track.to_dict()
    data.update(
        {
            "w_left": result.w_left.tolist(),
            "w_right": result.w_right.tolist(),
            "source_boundaries_csv": str(csv_source),
            "misses_left": int(result.misses_left),
            "misses_right": int(result.misses_right),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_bounds(
    track: DiscretizedTrack,
    left: np.ndarray,
    right: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    every: int = 10,
    show: bool = True,
    out_path: Path | None = None,
) -> None:
    """Visualize centerline samples with normals clipped to their intersections."""
    fig, ax = plt.subplots(figsize=(8, 6))
    left_closed = np.vstack([left, left[0]])
    right_closed = np.vstack([right, right[0]])
    center_closed = np.vstack([track.positions, track.positions[0]])

    ax.plot(left_closed[:, 0], left_closed[:, 1], label="left boundary", color="tab:blue")
    ax.plot(right_closed[:, 0], right_closed[:, 1], label="right boundary", color="tab:orange")
    ax.plot(center_closed[:, 0], center_closed[:, 1], label="centerline", color="k", linewidth=1.2)

    normals = np.column_stack((-np.sin(track.headings), np.cos(track.headings)))
    stride_indices = np.arange(0, track.num_points, every)
    for i in stride_indices:
        p = track.positions[i]
        n_hat = normals[i]
        if np.isfinite(w_left[i]):
            left_pt = p + w_left[i] * n_hat
            ax.plot([p[0], left_pt[0]], [p[1], left_pt[1]], color="tab:green", linewidth=1.0)
            ax.scatter(left_pt[0], left_pt[1], color="tab:green", s=10)
        if np.isfinite(w_right[i]):
            right_pt = p - w_right[i] * n_hat
            ax.plot([p[0], right_pt[0]], [p[1], right_pt[1]], color="tab:red", linewidth=1.0)
            ax.scatter(right_pt[0], right_pt[1], color="tab:red", s=10)
        ax.scatter(p[0], p[1], color="k", s=5)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Lateral bounds per center sample")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _demo() -> None:
    """Minimal runnable example for computing and visualizing bounds."""
    repo_root = Path(__file__).resolve().parents[2]
    track_path = repo_root / "data" / "discretized" / "ellipse.json"
    csv_path = repo_root / "data" / "tracks" / "ellipse.csv"

    track = DiscretizedTrack.load(track_path)
    boundaries = load_boundaries(csv_path)
    left = boundaries["left"]
    right = boundaries["right"]

    result = compute_lateral_bounds(track, left=left, right=right)
    print(f"Misses left/right: {result.misses_left} / {result.misses_right}")
    out_json = repo_root / "data" / "discretized" / "ellipse_with_widths.json"
    save_track_with_widths(out_json, track=track, csv_source=csv_path, result=result)
    print(f"Saved bounds + track to {out_json}")

    out_plot = repo_root / "out" / "lateral_bounds.png"
    plot_bounds(track, left, right, result.w_left, result.w_right, every=10, show=True)
    print(f"Saved plot to {out_plot}")


if __name__ == "__main__":
    _demo()
 

