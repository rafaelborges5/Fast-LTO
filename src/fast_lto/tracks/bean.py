"""
Kidney / bean shaped track generator.

The generator follows the same interface as `generate_ellipse_track` and returns
left, middle, and right boundaries with roughly even point spacing. The shape is
defined in polar coordinates with a few harmonics to create the characteristic
pinched bean outline and then scaled to a target midline length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

BoundaryName = Literal["left", "middle", "right"]


@dataclass
class BeanTrackConfig:
    target_midline_length_m: float = 175.0  # desired midline length after scaling (m)
    track_width_m: float = 3.5  # cone-to-cone width; widen to loosen the optimal line (m)
    nominal_spacing_m: float = 4.0  # target point spacing along midline for discretisation (m)
    integration_points: int = (
        6000  # resolution for arc-length integration; higher = smoother, slower
    )
    base_radius_m: float = 12.0  # unscaled base radius before perimeter matching (m)
    asymmetry: float = 0.65  # pushes one side outward; increase for a bigger outer bulge
    pinch: float = 0.4  # pulls the opposite side inward; increase for a tighter inner pinch
    double_lobe: float = (
        0.12  # adds secondary curvature; increase to round transitions / soften corners
    )
    rotation_rad: float = 0.0  # pre-rotation applied before reference pose alignment (rad)


def _radial_profile(theta: np.ndarray, config: BeanTrackConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return dimensionless r(theta) and dr/dtheta that define the bean outline.
    """

    # Baseline offset keeps radius strictly positive.
    r = (
        1.1
        + config.asymmetry * np.sin(theta)
        - config.pinch * np.cos(theta)
        + config.double_lobe * np.cos(2.0 * theta)
    )

    dr = (
        config.asymmetry * np.cos(theta)
        + config.pinch * np.sin(theta)
        - 2.0 * config.double_lobe * np.sin(2.0 * theta)
    )

    # Guards a small radius from becoming a sharp kink.
    r = np.maximum(r, 0.35)
    return r, dr


def _compute_scale_factor(config: BeanTrackConfig) -> float:
    """
    Scale the base bean so that its perimeter matches the target midline length.
    """

    n = config.integration_points
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r_dimless, dr_dimless = _radial_profile(theta, config)

    r = config.base_radius_m * r_dimless
    dr = config.base_radius_m * dr_dimless

    dx = dr * np.cos(theta) - r * np.sin(theta)
    dy = dr * np.sin(theta) + r * np.cos(theta)
    speed = np.sqrt(dx * dx + dy * dy)

    perimeter = float(speed.mean() * (2.0 * np.pi))
    return config.target_midline_length_m / perimeter


def _arc_length_parameterisation(
    config: BeanTrackConfig, scale_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build theta-to-arc-length mapping for the scaled bean shape.
    """

    n = config.integration_points
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r_dimless, dr_dimless = _radial_profile(theta, config)

    r = scale_factor * config.base_radius_m * r_dimless
    dr = scale_factor * config.base_radius_m * dr_dimless

    dx = dr * np.cos(theta) - r * np.sin(theta)
    dy = dr * np.sin(theta) + r * np.cos(theta)
    speed = np.sqrt(dx * dx + dy * dy)

    dtheta = (2.0 * np.pi) / n
    ds = speed * dtheta
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0.0)
    theta = np.insert(theta, 0, 0.0)

    total_length = s[-1]
    s *= config.target_midline_length_m / total_length
    return theta, s


def _generate_theta_samples(
    theta: np.ndarray, s: np.ndarray, config: BeanTrackConfig
) -> np.ndarray:
    """
    Generate theta samples with approximately even arc-length spacing.
    """

    total_length = s[-1]
    num_segments = int(np.floor(total_length / config.nominal_spacing_m))
    s_targets = np.linspace(0.0, total_length, num_segments, endpoint=False)
    theta_samples = np.interp(s_targets, s, theta)
    return theta_samples


def _bean_xy(
    theta: np.ndarray,
    config: BeanTrackConfig,
    scale_factor: float,
    rotation_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return x(theta), y(theta) for the scaled and rotated bean.
    """

    r_dimless, _ = _radial_profile(theta, config)
    r = scale_factor * config.base_radius_m * r_dimless

    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    cos_r = float(np.cos(rotation_rad))
    sin_r = float(np.sin(rotation_rad))

    x = cos_r * x0 - sin_r * y0
    y = sin_r * x0 + cos_r * y0
    return x, y


def _bean_derivatives(
    theta: np.ndarray,
    config: BeanTrackConfig,
    scale_factor: float,
    rotation_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return dx/dtheta, dy/dtheta for the scaled and rotated bean.
    """

    r_dimless, dr_dimless = _radial_profile(theta, config)
    r = scale_factor * config.base_radius_m * r_dimless
    dr = scale_factor * config.base_radius_m * dr_dimless

    dx0 = dr * np.cos(theta) - r * np.sin(theta)
    dy0 = dr * np.sin(theta) + r * np.cos(theta)

    cos_r = float(np.cos(rotation_rad))
    sin_r = float(np.sin(rotation_rad))

    dx = cos_r * dx0 - sin_r * dy0
    dy = sin_r * dx0 + cos_r * dy0
    return dx, dy


def _compute_boundaries(
    theta_samples: np.ndarray,
    config: BeanTrackConfig,
    scale_factor: float,
) -> Dict[BoundaryName, np.ndarray]:
    """
    Compute left, middle, and right boundaries for the bean shape.
    """

    x, y = _bean_xy(theta_samples, config, scale_factor, config.rotation_rad)
    dx, dy = _bean_derivatives(theta_samples, config, scale_factor, config.rotation_rad)

    tangents = np.stack((dx, dy), axis=1)
    speeds = np.linalg.norm(tangents, axis=1, keepdims=True)
    unit_tangents = tangents / speeds
    left_normals = np.stack((-unit_tangents[:, 1], unit_tangents[:, 0]), axis=1)

    midline = np.stack((x, y), axis=1)
    half_width = 0.5 * config.track_width_m
    left = midline + half_width * left_normals
    right = midline - half_width * left_normals

    # Align reference pose so that heading is +y at the first point.
    import math

    ref_point = midline[0]
    ref_tangent = unit_tangents[0]

    current_angle = math.atan2(ref_tangent[1], ref_tangent[0])
    target_angle = math.pi / 2.0
    delta_angle = target_angle - current_angle

    cos_d = math.cos(delta_angle)
    sin_d = math.sin(delta_angle)

    def _rotate(points: np.ndarray) -> np.ndarray:
        shifted = points - ref_point
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


def generate_bean_track(
    output_csv: str | Path | None = None,
    *,
    config: BeanTrackConfig | None = None,
) -> Dict[BoundaryName, np.ndarray]:
    """
    Generate a bean-shaped track and optionally write it to CSV.
    """

    if config is None:
        config = BeanTrackConfig()

    scale_factor = _compute_scale_factor(config)
    theta, s = _arc_length_parameterisation(config, scale_factor)
    theta_samples = _generate_theta_samples(theta, s, config)
    boundaries = _compute_boundaries(theta_samples, config, scale_factor)

    if output_csv is not None:
        output_path = Path(output_csv)
        _write_boundaries_csv(boundaries, output_path)

    return boundaries


__all__ = [
    "BeanTrackConfig",
    "BoundaryName",
    "generate_bean_track",
]


if __name__ == "__main__":
    generate_bean_track(output_csv="data/tracks/bean.csv")
