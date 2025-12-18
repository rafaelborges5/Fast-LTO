"""
Spline fitting and discretization for track centerlines.

This module provides functions to:
1. Load track middle line from CSV
2. Fit a periodic spline (C² or C⁴) to the centerline
3. Reparameterize by arc length
4. Sample at uniform arc-length intervals
5. Compute curvature at each sample point
6. Optionally visualize the result

The main entry point is `fit_and_discretize()`.

Example
-------
    from fast_lto.splines import fit_and_discretize

    track = fit_and_discretize(
        csv_path="data/tracks/ellipse.csv",
        ds_m=0.1,
        continuity="C2",
        viz=True,
    )
    track.save("data/discretized/ellipse.json")
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

if __name__ == "__main__":
    # Running as script - use absolute import
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
    from src.splines.discretized_track import DiscretizedTrack
else:
    # Imported as module - use relative import
    from .discretized_track import DiscretizedTrack 


ContinuityType = Literal["C2", "C4"]


def _load_middle_line(csv_path: Path) -> np.ndarray:
    """
    Load the middle line points from a track CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the track CSV with columns: boundary, x, y, index

    Returns
    -------
    np.ndarray
        Shape (N, 2) array of [x, y] points, ordered by index.
    """
    points: list[Tuple[int, float, float]] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["boundary"].strip().lower() == "middle":
                idx = int(row["index"])
                x = float(row["x"])
                y = float(row["y"])
                points.append((idx, x, y))

    # Sort by index to ensure correct ordering
    points.sort(key=lambda p: p[0])

    if len(points) < 4:
        raise ValueError(f"Need at least 4 middle line points, got {len(points)}")

    return np.array([[p[1], p[2]] for p in points], dtype=np.float64)


def _compute_chord_params(points: np.ndarray) -> np.ndarray:
    """
    Compute chord-length parameterization for the points.

    This gives a reasonable initial parameterization for spline fitting.
    For a closed curve, we include the wrap-around distance.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) array of [x, y] points.

    Returns
    -------
    np.ndarray
        Shape (N,) array of parameter values in [0, total_chord_length).
    """
    # Compute distances between consecutive points
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    # Cumulative chord length
    t = np.zeros(len(points))
    t[1:] = np.cumsum(segment_lengths)

    return t


def _fit_periodic_spline_c2(
    points: np.ndarray, t: np.ndarray
) -> Tuple[CubicSpline, CubicSpline]:
    """
    Fit periodic cubic splines (C²) to x(t) and y(t).

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) array of [x, y] points.
    t : np.ndarray
        Shape (N,) parameter values.

    Returns
    -------
    spline_x, spline_y : CubicSpline
        Periodic cubic splines for x(t) and y(t).
    """
    x = points[:, 0]
    y = points[:, 1]

    # For periodic splines, we need to ensure the function values match at endpoints
    # CubicSpline with bc_type='periodic' handles this, but requires f(t[0]) = f(t[-1])
    # We append the first point to close the loop at t = t[-1] + wrap_distance
    wrap_distance = np.linalg.norm(points[0] - points[-1])
    t_periodic = np.append(t, t[-1] + wrap_distance)
    x_periodic = np.append(x, x[0])
    y_periodic = np.append(y, y[0])

    spline_x = CubicSpline(t_periodic, x_periodic, bc_type="periodic")
    spline_y = CubicSpline(t_periodic, y_periodic, bc_type="periodic")

    return spline_x, spline_y


def _fit_periodic_spline_c4(
    points: np.ndarray, t: np.ndarray
) -> Tuple[object, object]:
    """
    Fit periodic quintic splines (C⁴) to x(t) and y(t).

    Uses scipy's make_interp_spline with k=5 (quintic).
    For periodic boundary conditions, we replicate points at the boundary.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) array of [x, y] points.
    t : np.ndarray
        Shape (N,) parameter values.

    Returns
    -------
    spline_x, spline_y : BSpline
        Periodic quintic splines for x(t) and y(t).
    """
    x = points[:, 0]
    y = points[:, 1]

    # Close the loop by appending the first point at t = t[-1] + wrap_distance
    wrap_distance = np.linalg.norm(points[0] - points[-1])
    t_periodic = np.append(t, t[-1] + wrap_distance)
    x_periodic = np.append(x, x[0])
    y_periodic = np.append(y, y[0])

    # Use make_interp_spline with k=5 (quintic)
    # Note: This uses not-a-knot BCs, not truly periodic.
    # For most tracks this works well enough.
    spline_x = make_interp_spline(t_periodic, x_periodic, k=5)
    spline_y = make_interp_spline(t_periodic, y_periodic, k=5)

    return spline_x, spline_y


def _compute_arc_length_mapping(
    spline_x, spline_y, t_max: float, num_integration_points: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mapping from parameter t to arc length s.

    Parameters
    ----------
    spline_x, spline_y : spline objects
        Fitted splines for x(t) and y(t).
    t_max : float
        Maximum parameter value (corresponds to one full lap).
    num_integration_points : int
        Number of points for numerical integration.

    Returns
    -------
    t_samples : np.ndarray
        Parameter values used for integration.
    s_samples : np.ndarray
        Corresponding cumulative arc lengths.
    """
    t_samples = np.linspace(0, t_max, num_integration_points)

    # Compute derivatives dx/dt, dy/dt
    dx_dt = spline_x(t_samples, 1)  # First derivative
    dy_dt = spline_y(t_samples, 1)

    # Speed = ds/dt = sqrt((dx/dt)² + (dy/dt)²)
    speed = np.sqrt(dx_dt**2 + dy_dt**2)

    # Integrate to get arc length: s(t) = ∫₀ᵗ speed(τ) dτ
    dt = t_max / (num_integration_points - 1)
    s_samples = np.zeros_like(t_samples)
    s_samples[1:] = np.cumsum(speed[:-1] + speed[1:]) * dt / 2  # Trapezoidal rule

    return t_samples, s_samples


def _sample_at_arc_lengths(
    spline_x,
    spline_y,
    t_samples: np.ndarray,
    s_samples: np.ndarray,
    ds_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Sample the spline at uniform arc-length intervals for a closed track.

    For a periodic track, we sample N points evenly distributed around the
    full loop, where N = round(L / ds). The actual spacing is L/N ≈ ds.

    Parameters
    ----------
    spline_x, spline_y : spline objects
        Fitted splines for x(t) and y(t).
    t_samples, s_samples : np.ndarray
        Arc-length mapping from _compute_arc_length_mapping.
    ds_m : float
        Desired spacing in meters (actual spacing will be L/N).

    Returns
    -------
    positions : np.ndarray
        Shape (N, 2) sampled positions.
    headings : np.ndarray
        Shape (N,) tangent angles in radians.
    curvatures : np.ndarray
        Shape (N,) signed curvatures.
    arc_lengths : np.ndarray
        Shape (N,) arc length values from 0 to L (exclusive).
    total_length : float
        Total track length L.
    actual_ds : float
        Actual discretization step (L / N).
    """
    total_length = s_samples[-1]

    # For a closed track: choose N such that spacing is close to ds_m
    # and points are evenly distributed around the full loop
    num_points = int(np.round(total_length / ds_m))
    actual_ds = total_length / num_points

    # Sample at s = 0, L/N, 2L/N, ..., (N-1)*L/N
    # (don't include s=L since that's the same as s=0 for a closed track)
    s_targets = np.linspace(0, total_length, num_points, endpoint=False)

    # Interpolate to find t values corresponding to target s values
    t_at_targets = np.interp(s_targets, s_samples, t_samples)

    # Evaluate splines at these t values
    x = spline_x(t_at_targets)
    y = spline_y(t_at_targets)
    positions = np.column_stack([x, y])

    # First derivatives for heading
    dx_dt = spline_x(t_at_targets, 1)
    dy_dt = spline_y(t_at_targets, 1)
    headings = np.arctan2(dy_dt, dx_dt)

    # Second derivatives for curvature
    d2x_dt2 = spline_x(t_at_targets, 2)
    d2y_dt2 = spline_y(t_at_targets, 2)

    # Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
    denominator = (dx_dt**2 + dy_dt**2) ** 1.5
    curvatures = numerator / denominator

    return positions, headings, curvatures, s_targets, total_length, actual_ds


def fit_and_discretize(
    csv_path: str | Path,
    ds_m: float = 0.1,
    continuity: ContinuityType = "C2",
    viz: bool = False,
    save_path: str | Path | None = None,
) -> DiscretizedTrack:
    """
    Fit a periodic spline to the track centerline and discretize it.

    This is the main entry point for spline fitting.

    Parameters
    ----------
    csv_path : str | Path
        Path to the track CSV file with columns: boundary, x, y, index
    ds_m : float
        Discretization step in meters. Default 0.1 (10 cm).
    continuity : "C2" | "C4"
        Spline continuity. C2 = cubic (continuous curvature),
        C4 = quintic (continuous curvature derivative).
    viz : bool
        If True, display a visualization of the fitted spline and samples.
    save_path : str | Path | None
        If provided, save the discretized track to this JSON path.

    Returns
    -------
    DiscretizedTrack
        The discretized track data structure.
    """
    csv_path = Path(csv_path)

    # 1. Load middle line points
    points = _load_middle_line(csv_path)

    # 2. Compute initial chord-length parameterization
    t = _compute_chord_params(points)

    # 3. Fit periodic spline
    if continuity == "C2":
        spline_x, spline_y = _fit_periodic_spline_c2(points, t)
    elif continuity == "C4":
        spline_x, spline_y = _fit_periodic_spline_c4(points, t)
    else:
        raise ValueError(f"Unknown continuity type: {continuity}. Use 'C2' or 'C4'.")

    # The period is the parameter value at which we complete one lap
    t_max = t[-1] + np.linalg.norm(points[0] - points[-1])

    # 4. Compute arc-length mapping
    t_samples, s_samples = _compute_arc_length_mapping(spline_x, spline_y, t_max)

    # 5. Sample at uniform arc-length intervals
    positions, headings, curvatures, arc_lengths, total_length, actual_ds = _sample_at_arc_lengths(
        spline_x, spline_y, t_samples, s_samples, ds_m
    )

    # 6. Create the discretized track
    track = DiscretizedTrack(
        positions=positions,
        headings=headings,
        curvatures=curvatures,
        arc_lengths=arc_lengths,
        ds_m=actual_ds,  # Use actual spacing (L/N), not requested ds_m
        total_length_m=total_length,
        num_points=len(positions),
        continuity=continuity,
        source_file=str(csv_path),
    )

    # 7. Optionally save
    if save_path is not None:
        track.save(save_path)

    # 8. Optionally visualize
    if viz:
        _visualize_spline_fit(
            original_points=points,
            spline_x=spline_x,
            spline_y=spline_y,
            t_max=t_max,
            track=track,
        )

    return track


def _visualize_spline_fit(
    original_points: np.ndarray,
    spline_x,
    spline_y,
    t_max: float,
    track: DiscretizedTrack,
) -> None:
    """
    Visualize the spline fitting result.

    Shows:
    - Original input points
    - Fitted spline (smooth curve)
    - Discretized sample points
    - Curvature colormap
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left plot: Track with spline and samples ---
    ax1 = axes[0]

    # Dense spline evaluation for smooth curve
    t_dense = np.linspace(0, t_max, 1000)
    x_spline = spline_x(t_dense)
    y_spline = spline_y(t_dense)

    # Plot smooth spline
    ax1.plot(x_spline, y_spline, "b-", linewidth=1.5, label="Fitted spline", alpha=0.7)

    # Plot original points
    ax1.scatter(
        original_points[:, 0],
        original_points[:, 1],
        c="red",
        s=60,
        marker="x",
        linewidths=2,
        label="Original points",
        zorder=5,
    )

    # Plot discretized samples
    ax1.scatter(
        track.positions[:, 0],
        track.positions[:, 1],
        c="green",
        s=15,
        marker="o",
        label=f"Samples (ds={track.ds_m:.2f}m, N={track.num_points})",
        zorder=4,
        alpha=0.8,
    )

    # Mark start point
    ax1.scatter(
        track.positions[0, 0],
        track.positions[0, 1],
        c="yellow",
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=1,
        label="Start",
        zorder=6,
    )

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(f"Spline Fit ({track.continuity}) - {Path(track.source_file).name}")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_aspect("equal")

    # --- Right plot: Curvature profile ---
    ax2 = axes[1]

    ax2.plot(track.arc_lengths, track.curvatures, "b-", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Arc length s [m]")
    ax2.set_ylabel("Curvature κ [1/m]")
    ax2.set_title("Curvature Profile")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Add statistics
    kappa_max = np.max(np.abs(track.curvatures))
    r_min = 1.0 / kappa_max if kappa_max > 0 else float("inf")
    stats_text = (
        f"Total length: {track.total_length_m:.2f} m\n"
        f"Max |κ|: {kappa_max:.4f} 1/m\n"
        f"Min radius: {r_min:.2f} m"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


__all__ = ["fit_and_discretize", "ContinuityType"]


if __name__ == "__main__":
    track = fit_and_discretize(
        csv_path="data/tracks/ellipse.csv",
        ds_m=0.5,
        continuity="C2",
        viz=True,
        save_path="data/discretized/ellipse.json",
    )
    print(f"Fitted spline: {track}")

