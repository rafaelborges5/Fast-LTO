"""
Random FSG Trackdrive track generator, to FSG rules D 8.1.

A periodic spline through randomly placed control points, scaled to the target
length, sampled at even arc length and offset to boundaries. The result is
checked against the rules that bound the shape -- straights no longer than 80 m
(D 8.1.1), turn radius at least 4.5 m (D 1.1.10), lap length 200-500 m
(D 8.1.2) -- and warns rather than raises on a violation.

Entry point ``generate_fsg_track``. Output format: ``data/tracks/README.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

BoundaryName = Literal["left", "middle", "right"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FSGTrackConfig:
    """Configuration for FSG Trackdrive track generation (D 8.1).

    The defaults produce a valid track. Set ``seed`` for a reproducible one, or
    ``control_points_xy`` to bypass random generation entirely.
    """

    # ── Track dimensions ────────────────────────────────────────────
    target_midline_length_m: float = 400.0  # desired midline length (200–500 m per D 8.1.2)
    track_width_m: float = 3.8  # constant track width (m); always 3 m
    nominal_spacing_m: float = 4.0  # target point spacing along midline for discretisation (m)

    # ── Shape control ───────────────────────────────────────────────
    num_control_points: int = 25  # number of random control points around the loop
    base_radius_m: float = 15.0  # mean distance of control points from centroid before scaling (m)
    radial_variance: float = 0.3  # fractional spread of radii: r ∈ base*(1 ± variance)
    angular_variance: float = 0.15  # fractional jitter of angular positions (0 = evenly spaced)
    smoothing_iterations: int = 1  # Laplacian smoothing passes on control polygon (0 = none)

    # ── Turn complexity (chicanes / S-curves) ────────────────────────
    num_harmonics: int = 3  # random sinusoidal harmonics added to radii (0 = convex oval)
    harmonic_amplitude: float = 0.45  # max amplitude of each harmonic as fraction of base_radius

    # ── FSG regulatory constraints ──────────────────────────────────
    min_turn_radius_m: float = 4.5  # D 1.1.10: minimum turning radius = diameter / 2 = 9 / 2 m
    max_straight_length_m: float = 80.0  # D 8.1.1: straights no longer than 80 m

    # ── Reproducibility ─────────────────────────────────────────────
    seed: int | None = None  # random seed; None for non-deterministic generation

    # ── Manual override ─────────────────────────────────────────────
    # A non-self-intersecting loop; skips random generation and smoothing, but
    # is still scaled to target_midline_length_m.
    control_points_xy: list[tuple[float, float]] | None = None

    # ── Internal parameters ─────────────────────────────────────────
    integration_points: int = 6000  # resolution for arc-length integration
    rotation_rad: float = 0.0  # pre-rotation before reference-pose alignment (rad)


# ---------------------------------------------------------------------------
# Control-point generation
# ---------------------------------------------------------------------------


def _generate_control_points(config: FSGTrackConfig) -> np.ndarray:
    """Return (N, 2) control points forming a closed loop.

    If ``config.control_points_xy`` is set, those points are returned directly.
    Otherwise, random points are generated in polar coordinates around a
    circle of radius ``base_radius_m`` and optionally Laplacian-smoothed.
    """
    if config.control_points_xy is not None:
        return np.array(config.control_points_xy, dtype=np.float64)

    rng = np.random.default_rng(config.seed)
    n = config.num_control_points

    spacing = 2.0 * np.pi / n
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angle_perturb = rng.uniform(
        -config.angular_variance * spacing,
        config.angular_variance * spacing,
        n,
    )
    angles = np.sort((base_angles + angle_perturb) % (2.0 * np.pi))

    radii = config.base_radius_m * (
        1.0 + rng.uniform(-config.radial_variance, config.radial_variance, n)
    )

    # Harmonics push some sections inward and others outward, so the curvature
    # changes sign; without them the loop is convex and turns only one way.
    for _ in range(config.num_harmonics):
        freq = rng.integers(2, 8)  # angular frequency
        amp = rng.uniform(0.2, 1.0) * config.harmonic_amplitude * config.base_radius_m
        phase = rng.uniform(0.0, 2.0 * np.pi)
        radii += amp * np.cos(freq * angles + phase)

    # Strictly positive, to avoid fold-overs.
    radii = np.maximum(radii, 0.15 * config.base_radius_m)

    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    for _ in range(config.smoothing_iterations):
        smoothed = np.empty_like(points)
        for i in range(n):
            prev = points[(i - 1) % n]
            nxt = points[(i + 1) % n]
            smoothed[i] = 0.5 * points[i] + 0.25 * (prev + nxt)
        points = smoothed

    return points


# ---------------------------------------------------------------------------
# Periodic spline fitting
# ---------------------------------------------------------------------------


def _fit_periodic_spline(
    points: np.ndarray,
) -> Tuple[CubicSpline, CubicSpline, float]:
    """Fit C²-periodic cubic splines to x(t) and y(t).

    Parameters
    ----------
    points : np.ndarray
        (N, 2) control points forming a closed loop.

    Returns
    -------
    spline_x, spline_y : CubicSpline
        Periodic cubic splines for x(t) and y(t).
    t_max : float
        Parameter value corresponding to one full lap (closure point).
    """
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(points))
    t[1:] = np.cumsum(seg_lens)

    wrap_dist = np.linalg.norm(points[0] - points[-1])
    t_max = t[-1] + wrap_dist
    t_periodic = np.append(t, t_max)
    x_periodic = np.append(points[:, 0], points[0, 0])
    y_periodic = np.append(points[:, 1], points[0, 1])

    spline_x = CubicSpline(t_periodic, x_periodic, bc_type="periodic")
    spline_y = CubicSpline(t_periodic, y_periodic, bc_type="periodic")

    return spline_x, spline_y, t_max


# ---------------------------------------------------------------------------
# Perimeter & scale factor
# ---------------------------------------------------------------------------


def _compute_scale_factor(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    t_max: float,
    config: FSGTrackConfig,
) -> float:
    """Scale factor so that the spline perimeter matches ``target_midline_length_m``."""
    n = config.integration_points
    t = np.linspace(0.0, t_max, n)
    dx = spline_x(t, 1)
    dy = spline_y(t, 1)
    speed = np.sqrt(dx * dx + dy * dy)

    dt = t_max / (n - 1)
    perimeter = float(np.sum((speed[:-1] + speed[1:]) * dt * 0.5))

    return config.target_midline_length_m / perimeter


# ---------------------------------------------------------------------------
# Arc-length parameterisation
# ---------------------------------------------------------------------------


def _arc_length_parameterisation(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    t_max: float,
    config: FSGTrackConfig,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a mapping from parameter *t* to arc length *s* on the scaled curve.

    Returns
    -------
    t_samples : np.ndarray  – dense parameter grid.
    s_samples : np.ndarray  – cumulative arc length at each grid point.
    """
    n = config.integration_points
    t_samples = np.linspace(0.0, t_max, n)

    dx = spline_x(t_samples, 1) * scale
    dy = spline_y(t_samples, 1) * scale
    speed = np.sqrt(dx * dx + dy * dy)

    dt = t_max / (n - 1)
    ds_segments = (speed[:-1] + speed[1:]) * dt * 0.5
    s_samples = np.zeros(n)
    s_samples[1:] = np.cumsum(ds_segments)

    # Normalise to the exact target length (remove numerical drift).
    s_samples *= config.target_midline_length_m / s_samples[-1]

    return t_samples, s_samples


# ---------------------------------------------------------------------------
# Even-spacing sampling
# ---------------------------------------------------------------------------


def _sample_evenly(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    t_dense: np.ndarray,
    s_dense: np.ndarray,
    config: FSGTrackConfig,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample midline, tangents, and curvatures at even arc-length intervals.

    Returns
    -------
    midline   : (N, 2)  – sampled positions.
    tangents  : (N, 2)  – (unnormalised) tangent vectors.
    curvatures : (N,)   – signed curvature at each sample.
    """
    total_length = s_dense[-1]
    num_segments = int(np.floor(total_length / config.nominal_spacing_m))
    s_targets = np.linspace(0.0, total_length, num_segments, endpoint=False)

    t_at = np.interp(s_targets, s_dense, t_dense)

    x = spline_x(t_at) * scale
    y = spline_y(t_at) * scale
    midline = np.column_stack([x, y])

    dx = spline_x(t_at, 1) * scale
    dy = spline_y(t_at, 1) * scale
    tangents = np.column_stack([dx, dy])

    ddx = spline_x(t_at, 2) * scale
    ddy = spline_y(t_at, 2) * scale

    # Signed curvature: κ = (x'y'' – y'x'') / (x'² + y'²)^{3/2}
    numerator = dx * ddy - dy * ddx
    denominator = (dx * dx + dy * dy) ** 1.5
    curvatures = numerator / (denominator + 1e-30)

    return midline, tangents, curvatures


# ---------------------------------------------------------------------------
# Boundary computation + pose alignment
# ---------------------------------------------------------------------------


def _compute_boundaries(
    midline: np.ndarray,
    tangents: np.ndarray,
    config: FSGTrackConfig,
) -> Dict[BoundaryName, np.ndarray]:
    """Offset the midline by ±half-width and align the reference pose.

    The first midline point is moved to the origin with heading aligned to +y.
    """
    speeds = np.linalg.norm(tangents, axis=1, keepdims=True)
    unit_tangents = tangents / speeds
    left_normals = np.column_stack([-unit_tangents[:, 1], unit_tangents[:, 0]])

    half_width = 0.5 * config.track_width_m
    left = midline + half_width * left_normals
    right = midline - half_width * left_normals

    # ── Align reference pose so that heading is +y at the first point ──
    ref_point = midline[0]
    ref_tangent = unit_tangents[0]

    current_angle = math.atan2(ref_tangent[1], ref_tangent[0])
    target_angle = math.pi / 2.0
    delta_angle = target_angle - current_angle

    cos_d = math.cos(delta_angle)
    sin_d = math.sin(delta_angle)

    def _rotate(points: np.ndarray) -> np.ndarray:
        shifted = points - ref_point
        xr = cos_d * shifted[:, 0] - sin_d * shifted[:, 1]
        yr = sin_d * shifted[:, 0] + cos_d * shifted[:, 1]
        return np.column_stack([xr, yr])

    return {
        "left": _rotate(left),
        "middle": _rotate(midline),
        "right": _rotate(right),
    }


# ---------------------------------------------------------------------------
# FSG constraint validation
# ---------------------------------------------------------------------------


def _validate_fsg_constraints(
    midline: np.ndarray,
    curvatures: np.ndarray,
    config: FSGTrackConfig,
) -> List[str]:
    """Check FSG D 8.1 constraints.  Returns a list of human-readable warnings."""
    warnings: List[str] = []
    n = len(midline)

    seg_lens = np.linalg.norm(np.diff(np.vstack([midline, midline[:1]]), axis=0), axis=1)
    total_length = float(seg_lens.sum())
    if total_length < 200.0:
        warnings.append(f"Track length {total_length:.1f} m < 200 m (D 8.1.2)")
    if total_length > 500.0:
        warnings.append(f"Track length {total_length:.1f} m > 500 m (D 8.1.2)")

    max_abs_kappa = float(np.max(np.abs(curvatures)))
    min_radius = 1.0 / max_abs_kappa if max_abs_kappa > 1e-12 else float("inf")
    if min_radius < config.min_turn_radius_m:
        warnings.append(
            f"Min turn radius {min_radius:.2f} m < {config.min_turn_radius_m:.2f} m (D 1.1.10)"
        )

    # "Straight" is radius > 200 m, i.e. |kappa| < 0.005.
    kappa_threshold = 0.005
    ds = config.nominal_spacing_m
    current_straight = 0.0
    max_straight = 0.0
    for i in range(n):
        if abs(curvatures[i]) < kappa_threshold:
            current_straight += ds
        else:
            max_straight = max(max_straight, current_straight)
            current_straight = 0.0
    max_straight = max(max_straight, current_straight)
    if max_straight > config.max_straight_length_m:
        warnings.append(
            f"Max straight ~{max_straight:.1f} m > {config.max_straight_length_m:.1f} m (D 8.1.1)"
        )

    return warnings


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_SIDE_CODES: Dict[BoundaryName, str] = {"left": "L", "middle": "M", "right": "R"}


def _write_boundaries_csv(
    boundaries: Dict[BoundaryName, np.ndarray],
    output_csv: Path,
) -> None:
    """Write boundaries to CSV with columns ``side,cone_id,x,y``."""
    import csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "cone_id", "x", "y"])
        for boundary_name, points in boundaries.items():
            code = _SIDE_CODES[boundary_name]
            for idx, (x, y) in enumerate(points):
                writer.writerow([code, idx, f"{x:.6f}", f"{y:.6f}"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_fsg_track(
    output_csv: str | Path | None = None,
    *,
    config: FSGTrackConfig | None = None,
) -> Dict[BoundaryName, np.ndarray]:
    """Generate a random FSG Trackdrive track and optionally write it to CSV.

    Parameters
    ----------
    output_csv : str | Path | None
        Path to write the track CSV.  ``None`` ⇒ no file is written.
    config : FSGTrackConfig | None
        Configuration object.  ``None`` ⇒ use defaults.

    Returns
    -------
    boundaries : dict
        ``{"left": (N,2), "middle": (N,2), "right": (N,2)}`` NumPy arrays.
    """
    if config is None:
        config = FSGTrackConfig()

    control_points = _generate_control_points(config)
    spline_x, spline_y, t_max = _fit_periodic_spline(control_points)
    scale = _compute_scale_factor(spline_x, spline_y, t_max, config)
    t_dense, s_dense = _arc_length_parameterisation(spline_x, spline_y, t_max, config, scale)
    midline, tangents, curvatures = _sample_evenly(
        spline_x, spline_y, t_dense, s_dense, config, scale
    )
    boundaries = _compute_boundaries(midline, tangents, config)

    for w in _validate_fsg_constraints(midline, curvatures, config):
        print(f"\u26a0  FSG constraint violation: {w}")

    if output_csv is not None:
        _write_boundaries_csv(boundaries, Path(output_csv))

    return boundaries


__all__ = [
    "FSGTrackConfig",
    "BoundaryName",
    "generate_fsg_track",
]


if __name__ == "__main__":
    generate_fsg_track(output_csv="data/tracks/fsg_random.csv")
