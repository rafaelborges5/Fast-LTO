"""Spline fitting and arc-length discretisation for track centrelines.

Periodic spline through midline cones, sampled at uniform ``ds``.
Entry point: ``fit_and_discretize``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline
from scipy.signal import savgol_filter

from fast_lto.splines.discretized_track import DiscretizedTrack

#: A fitted 1-D interpolator: CubicSpline for C2 fits, BSpline for C4.
Spline1D = Union[CubicSpline, BSpline]

ContinuityType = Literal["C2", "C4"]

_SEG_RATIO_THRESHOLD = 0.5
_CURVATURE_RATE_THRESHOLD = 0.5  # 1/m²


def _check_centerline_quality(points: np.ndarray) -> dict:
    """Run quality checks on raw middle line points.

    Returns a dict with diagnostic metrics and an overall ``needs_smoothing``
    flag.
    """
    segs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    median_seg = float(np.median(segs))
    min_seg = float(segs.min())
    seg_ratio = min_seg / median_seg if median_seg > 0 else 1.0

    # 3-point Menger curvature at each interior point.
    kappas = np.zeros(len(points) - 2)
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        ab = np.linalg.norm(p1 - p0)
        bc = np.linalg.norm(p2 - p1)
        ca = np.linalg.norm(p2 - p0)
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        denom = ab * bc * ca
        kappas[i - 1] = 2.0 * cross / denom if denom > 1e-12 else 0.0

    ds_local = 0.5 * (segs[:-1] + segs[1:])
    dk = np.abs(np.diff(kappas))
    dkds = dk / ds_local[:-1] if len(ds_local) > 1 else np.zeros(0)
    max_curv_rate = float(dkds.max()) if len(dkds) > 0 else 0.0

    needs_smoothing = seg_ratio < _SEG_RATIO_THRESHOLD and max_curv_rate > _CURVATURE_RATE_THRESHOLD

    return {
        "seg_ratio": seg_ratio,
        "min_seg": min_seg,
        "median_seg": median_seg,
        "max_curv_rate": max_curv_rate,
        "needs_smoothing": needs_smoothing,
    }


def _smooth_middle_line(points: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Apply circular Savitzky-Golay smoothing to middle line points."""
    if window < 3 or window % 2 == 0:
        raise ValueError(f"smooth_centerline must be an odd integer >= 3, got {window}")
    n = len(points)
    if window >= n:
        raise ValueError(f"smooth_centerline ({window}) must be < number of points ({n})")
    pad = window
    padded = np.vstack([points[-pad:], points, points[:pad]])
    smoothed_x = savgol_filter(padded[:, 0], window, polyorder)
    smoothed_y = savgol_filter(padded[:, 1], window, polyorder)
    return np.column_stack([smoothed_x[pad : pad + n], smoothed_y[pad : pad + n]])


def _load_middle_line(csv_path: Path) -> np.ndarray:
    """The (N, 2) midline points of a track CSV, ordered by ``cone_id``."""
    points: list[Tuple[int, float, float]] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["side"].strip().upper() == "M":
                idx = int(row["cone_id"])
                x = float(row["x"])
                y = float(row["y"])
                points.append((idx, x, y))

    points.sort(key=lambda p: p[0])

    if len(points) < 4:
        raise ValueError(f"Need at least 4 middle line points, got {len(points)}")

    arr = np.array([[p[1], p[2]] for p in points], dtype=np.float64)

    # Strip a closing duplicate, if the track repeats its first point.
    if np.linalg.norm(arr[0] - arr[-1]) < 1e-6:
        arr = arr[:-1]

    return arr


def _compute_chord_params(points: np.ndarray) -> np.ndarray:
    """Chord-length parameter values in ``[0, total_chord_length)``.

    The initial parameterisation the spline is fitted against, before it is
    reparameterised by true arc length.
    """
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    t = np.zeros(len(points))
    t[1:] = np.cumsum(segment_lengths)

    return t


def _fit_periodic_spline_c2(points: np.ndarray, t: np.ndarray) -> Tuple[CubicSpline, CubicSpline]:
    """Fit periodic cubic splines (C2) to x(t) and y(t)."""
    x = points[:, 0]
    y = points[:, 1]

    # bc_type="periodic" requires f(t[0]) == f(t[-1]).
    wrap_distance = np.linalg.norm(points[0] - points[-1])
    t_periodic = np.append(t, t[-1] + wrap_distance)
    x_periodic = np.append(x, x[0])
    y_periodic = np.append(y, y[0])

    spline_x = CubicSpline(t_periodic, x_periodic, bc_type="periodic")
    spline_y = CubicSpline(t_periodic, y_periodic, bc_type="periodic")

    return spline_x, spline_y


def _fit_periodic_spline_c4(points: np.ndarray, t: np.ndarray) -> Tuple[object, object]:
    """Fit quintic splines (C4) to x(t) and y(t), closing the loop by hand."""
    x = points[:, 0]
    y = points[:, 1]

    wrap_distance = np.linalg.norm(points[0] - points[-1])
    t_periodic = np.append(t, t[-1] + wrap_distance)
    x_periodic = np.append(x, x[0])
    y_periodic = np.append(y, y[0])

    # Not-a-knot, so not truly periodic; the seam is one sample in hundreds.
    spline_x = make_interp_spline(t_periodic, x_periodic, k=5)
    spline_y = make_interp_spline(t_periodic, y_periodic, k=5)

    return spline_x, spline_y


def _compute_arc_length_mapping(
    spline_x: Spline1D,
    spline_y: Spline1D,
    t_max: float,
    num_integration_points: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """The mapping from spline parameter t to arc length s, by quadrature.

    Returns ``(t_samples, s_samples)``, a lookup table the caller inverts by
    interpolation to place nodes at a uniform arc-length spacing.
    """
    t_samples = np.linspace(0, t_max, num_integration_points)

    dx_dt = spline_x(t_samples, 1)
    dy_dt = spline_y(t_samples, 1)

    speed = np.sqrt(dx_dt**2 + dy_dt**2)

    # s(t) = ∫₀ᵗ speed(τ) dτ
    dt = t_max / (num_integration_points - 1)
    s_samples = np.zeros_like(t_samples)
    s_samples[1:] = np.cumsum(speed[:-1] + speed[1:]) * dt / 2  # Trapezoidal rule

    return t_samples, s_samples


def _sample_at_arc_lengths(
    spline_x: Spline1D,
    spline_y: Spline1D,
    t_samples: np.ndarray,
    s_samples: np.ndarray,
    ds_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
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
        Shape (N,) signed curvatures at grid points.
    curvatures_half : np.ndarray
        Shape (N,) signed curvatures at interval midpoints s_i + actual_ds/2.
        The last entry is the midpoint between point N-1 and the wrap-around
        point 0 (i.e. at s = (N - 0.5) * actual_ds).
    arc_lengths : np.ndarray
        Shape (N,) arc length values from 0 to L (exclusive).
    total_length : float
        Total track length L.
    actual_ds : float
        Actual discretization step (L / N).
    """
    total_length = s_samples[-1]

    # Even division of the loop; actual_ds ≈ ds_m.
    num_points = int(np.round(total_length / ds_m))
    actual_ds = total_length / num_points

    # Excludes s = L, which is s = 0 again on a closed track.
    s_targets = np.linspace(0, total_length, num_points, endpoint=False)

    t_at_targets = np.interp(s_targets, s_samples, t_samples)

    x = spline_x(t_at_targets)
    y = spline_y(t_at_targets)
    positions = np.column_stack([x, y])

    dx_dt = spline_x(t_at_targets, 1)
    dy_dt = spline_y(t_at_targets, 1)
    headings = np.arctan2(dy_dt, dx_dt)

    d2x_dt2 = spline_x(t_at_targets, 2)
    d2y_dt2 = spline_y(t_at_targets, 2)

    # Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
    denominator = (dx_dt**2 + dy_dt**2) ** 1.5
    curvatures = numerator / denominator

    # Midpoint curvatures, for RK4.
    s_mids = s_targets + actual_ds / 2.0

    t_at_mids = np.interp(s_mids, s_samples, t_samples)

    dx_dt_m = spline_x(t_at_mids, 1)
    dy_dt_m = spline_y(t_at_mids, 1)
    d2x_dt2_m = spline_x(t_at_mids, 2)
    d2y_dt2_m = spline_y(t_at_mids, 2)

    num_m = dx_dt_m * d2y_dt2_m - dy_dt_m * d2x_dt2_m
    den_m = (dx_dt_m**2 + dy_dt_m**2) ** 1.5
    curvatures_half = num_m / den_m

    return positions, headings, curvatures, curvatures_half, s_targets, total_length, actual_ds


def fit_and_discretize(
    csv_path: str | Path,
    ds_m: float = 0.1,
    continuity: ContinuityType = "C2",
    viz: bool = False,
    save_path: str | Path | None = None,
    smooth_centerline: int = 0,
) -> DiscretizedTrack:
    """Fit a periodic centreline spline and discretize at spacing ``ds_m``.

    This is the main entry point for spline fitting.

    Parameters
    ----------
    csv_path : str | Path
        Path to the track CSV file with columns: side, cone_id, x, y
    ds_m : float
        Discretization step in meters. Default 0.1 (10 cm).
    continuity : "C2" | "C4"
        Spline continuity. C2 = cubic (continuous curvature),
        C4 = quintic (continuous curvature derivative).
    viz : bool
        If True, display a visualization of the fitted spline and samples.
    save_path : str | Path | None
        If provided, save the discretized track to this JSON path.
    smooth_centerline : int
        Savitzky-Golay window length for smoothing the raw middle line
        points before spline fitting.  Must be an odd integer >= 3.
        0 (default) disables smoothing.  Any value > 0 is always applied;
        the quality check only warns when smoothing looks warranted but
        smooth_centerline is left at 0.

    Returns
    -------
    DiscretizedTrack
        The discretized track data structure.
    """
    csv_path = Path(csv_path)

    points = _load_middle_line(csv_path)

    # Advisory: an explicit smooth_centerline is always honoured.
    quality = _check_centerline_quality(points)
    if smooth_centerline > 0:
        print(
            f"  Applying Savgol smoothing to centerline (window={smooth_centerline}) "
            f"[seg_ratio={quality['seg_ratio']:.2f}, "
            f"max_dκ/ds={quality['max_curv_rate']:.2f} 1/m²]"
        )
        points = _smooth_middle_line(points, smooth_centerline)
    elif quality["needs_smoothing"]:
        print(
            f"  WARNING: centerline noise detected "
            f"(seg_ratio={quality['seg_ratio']:.2f}, "
            f"max_dκ/ds={quality['max_curv_rate']:.2f} 1/m²). "
            f"Consider setting smooth_centerline >= 3 in the config."
        )

    t = _compute_chord_params(points)

    if continuity == "C2":
        spline_x, spline_y = _fit_periodic_spline_c2(points, t)
    elif continuity == "C4":
        spline_x, spline_y = _fit_periodic_spline_c4(points, t)
    else:
        raise ValueError(f"Unknown continuity type: {continuity}. Use 'C2' or 'C4'.")

    # The period: the parameter value at which the lap closes.
    t_max = t[-1] + np.linalg.norm(points[0] - points[-1])

    t_samples, s_samples = _compute_arc_length_mapping(spline_x, spline_y, t_max)

    positions, headings, curvatures, curvatures_half, arc_lengths, total_length, actual_ds = (
        _sample_at_arc_lengths(spline_x, spline_y, t_samples, s_samples, ds_m)
    )

    track = DiscretizedTrack(
        positions=positions,
        headings=headings,
        curvatures=curvatures,
        curvatures_half=curvatures_half,
        arc_lengths=arc_lengths,
        ds_m=actual_ds,  # the achieved spacing L/N, not the requested ds_m
        total_length_m=total_length,
        num_points=len(positions),
        continuity=continuity,
        source_file=str(csv_path),
    )

    if save_path is not None:
        track.save(save_path)

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
    spline_x: Spline1D,
    spline_y: Spline1D,
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

    ax1 = axes[0]

    t_dense = np.linspace(0, t_max, 1000)
    x_spline = spline_x(t_dense)
    y_spline = spline_y(t_dense)

    ax1.plot(x_spline, y_spline, "b-", linewidth=1.5, label="Fitted spline", alpha=0.7)

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

    ax2 = axes[1]

    ax2.plot(track.arc_lengths, track.curvatures, "b-", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Arc length s [m]")
    ax2.set_ylabel("Curvature κ [1/m]")
    ax2.set_title("Curvature Profile")
    ax2.grid(True, linestyle="--", alpha=0.4)

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
        csv_path="data/tracks/fsg_random.csv",
        ds_m=0.5,
        continuity="C4",
        viz=True,
        save_path="data/discretized/fsg_random.json",
    )
    print(f"Fitted spline: {track}")
