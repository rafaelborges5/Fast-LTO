"""Ellipse track generator (~75 m midline, L/M/R boundaries).

Entrypoint: ``generate_ellipse_track``. CSV columns: ``side,cone_id,x,y``
with ``side`` in ``{L, M, R}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

BoundaryName = Literal["left", "middle", "right"]


@dataclass
class EllipseTrackConfig:
    target_midline_length_m: float = 75.0  # desired midline length after scaling (m)
    track_width_m: float = 3.5  # cone-to-cone width; widen to loosen the optimal line (m)
    nominal_spacing_m: float = 4.0  # target point spacing along midline for discretisation (m)
    aspect_ratio: float = 0.6  # a/b ratio (<1 squashes in x, >1 stretches in x)
    integration_points: int = (
        5000  # resolution for arc-length integration; higher = smoother, slower
    )
    rotation_rad: float = np.pi / 2.0  # pre-rotation so the long axis roughly aligns with +y (rad)


def _ellipse_xy(
    theta: np.ndarray,
    a: float,
    b: float,
    rotation_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return x(theta), y(theta) for a possibly rotated ellipse."""

    x0 = a * np.cos(theta)
    y0 = b * np.sin(theta)

    cos_r = float(np.cos(rotation_rad))
    sin_r = float(np.sin(rotation_rad))

    x = cos_r * x0 - sin_r * y0
    y = sin_r * x0 + cos_r * y0
    return x, y


def _ellipse_derivatives(
    theta: np.ndarray,
    a: float,
    b: float,
    rotation_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dx/dtheta, dy/dtheta for the (possibly rotated) ellipse."""

    dx0 = -a * np.sin(theta)
    dy0 = b * np.cos(theta)

    cos_r = float(np.cos(rotation_rad))
    sin_r = float(np.sin(rotation_rad))

    dx = cos_r * dx0 - sin_r * dy0
    dy = sin_r * dx0 + cos_r * dy0
    return dx, dy


def _compute_scaled_axes(config: EllipseTrackConfig) -> Tuple[float, float, float]:
    """
    Compute semi-axes (a, b) scaled such that the ellipse midline length
    is approximately `config.target_midline_length_m`.
    """

    b0 = 10.0
    a0 = config.aspect_ratio * b0

    n = config.integration_points
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # Rotation does not change the perimeter, so measure it unrotated.
    dx, dy = _ellipse_derivatives(theta, a0, b0, rotation_rad=0.0)
    speed = np.sqrt(dx * dx + dy * dy)  # ds/dtheta
    perimeter_base = float(speed.mean() * (2.0 * np.pi))

    scale = config.target_midline_length_m / perimeter_base
    a = a0 * scale
    b = b0 * scale
    perimeter_scaled = perimeter_base * scale

    return a, b, perimeter_scaled


def _arc_length_parameterisation(
    a: float, b: float, config: EllipseTrackConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a mapping from parameter theta to arc length s along the ellipse.

    Returns
    -------
    theta : np.ndarray
        Parameter samples in [0, 2π).
    s : np.ndarray
        Cumulative arc length at each theta, starting at 0.
    """

    n = config.integration_points
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dx, dy = _ellipse_derivatives(theta, a, b, rotation_rad=0.0)
    speed = np.sqrt(dx * dx + dy * dy)  # ds/dtheta
    dtheta = (2.0 * np.pi) / n
    ds = speed * dtheta
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0.0)
    theta = np.insert(theta, 0, 0.0)

    # Normalise away the quadrature's drift from the exact perimeter.
    total_length = s[-1]
    s *= config.target_midline_length_m / total_length

    return theta, s


def _generate_theta_samples(
    theta: np.ndarray, s: np.ndarray, config: EllipseTrackConfig
) -> np.ndarray:
    """
    Generate parameter values theta_i such that the corresponding points
    along the midline are approximately evenly spaced in arc length.
    """

    total_length = s[-1]
    ds_nominal = config.nominal_spacing_m

    num_segments = int(np.floor(total_length / ds_nominal))
    s_targets = np.linspace(0.0, total_length, num_segments, endpoint=False)

    # Invert s(theta) to get theta(s).
    theta_samples = np.interp(s_targets, s, theta)
    return theta_samples


def _compute_boundaries(
    theta_samples: np.ndarray, a: float, b: float, config: EllipseTrackConfig
) -> Dict[BoundaryName, np.ndarray]:
    """
    Compute left, middle, and right boundaries for the given theta samples.

    Returns a dictionary mapping boundary name to an (N, 2) array of points.
    """

    x, y = _ellipse_xy(theta_samples, a, b, rotation_rad=config.rotation_rad)
    dx, dy = _ellipse_derivatives(theta_samples, a, b, rotation_rad=config.rotation_rad)

    tangents = np.stack((dx, dy), axis=1)
    speeds = np.linalg.norm(tangents, axis=1, keepdims=True)
    unit_tangents = tangents / speeds

    # The left normal is the tangent rotated +90 degrees.
    left_normals = np.stack((-unit_tangents[:, 1], unit_tangents[:, 0]), axis=1)

    midline = np.stack((x, y), axis=1)
    half_width = 0.5 * config.track_width_m

    left = midline + half_width * left_normals
    right = midline - half_width * left_normals

    # First midline point to the origin, heading +y, as on a real track.
    ref_point = midline[0]
    ref_tangent = unit_tangents[0]

    import math

    current_angle = math.atan2(ref_tangent[1], ref_tangent[0])
    target_angle = math.pi / 2.0
    delta_angle = target_angle - current_angle

    cos_d = math.cos(delta_angle)
    sin_d = math.sin(delta_angle)

    def _rotate(points: np.ndarray) -> np.ndarray:
        shifted = points - ref_point  # move reference point to origin
        x = shifted[:, 0]
        y = shifted[:, 1]
        xr = cos_d * x - sin_d * y
        yr = sin_d * x + cos_d * y
        return np.stack((xr, yr), axis=1)

    midline_t = _rotate(midline)
    left_t = _rotate(left)
    right_t = _rotate(right)

    return {
        "left": left_t,
        "middle": midline_t,
        "right": right_t,
    }


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


def generate_ellipse_track(
    output_csv: str | Path | None = None,
    *,
    config: EllipseTrackConfig | None = None,
) -> Dict[BoundaryName, np.ndarray]:
    """
    Generate an ellipse track and optionally write it to a CSV file.

    Parameters
    ----------
    output_csv:
        Path to the CSV file to write. If None, no file is written.
    config:
        Optional configuration object. If None, defaults are used.

    Returns
    -------
    boundaries:
        Dictionary with keys "left", "middle", "right", each mapped to an
        (N, 2) NumPy array of ordered points.
    """

    if config is None:
        config = EllipseTrackConfig()

    a, b, _ = _compute_scaled_axes(config)
    theta, s = _arc_length_parameterisation(a, b, config)
    theta_samples = _generate_theta_samples(theta, s, config)
    boundaries = _compute_boundaries(theta_samples, a, b, config)

    if output_csv is not None:
        output_path = Path(output_csv)
        _write_boundaries_csv(boundaries, output_path)

    return boundaries


__all__ = [
    "EllipseTrackConfig",
    "BoundaryName",
    "generate_ellipse_track",
]


if __name__ == "__main__":
    generate_ellipse_track(output_csv="data/tracks/ellipse.csv")
