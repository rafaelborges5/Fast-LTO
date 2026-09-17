"""
Visualization helpers for OCP solutions.

Functions accept numpy arrays and can be used from scripts or the main pipeline.
Run directly to plot the last saved solution (ellipse_point_mass.npz) if present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from fast_lto.visualization.summary_panel import plot_profiling_panel


def plot_path_with_speed(
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    path_xy: np.ndarray,
    v: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(cones_left[:, 0], cones_left[:, 1], c="tab:blue", s=12, label="left cones")
    ax.scatter(cones_right[:, 0], cones_right[:, 1], c="tab:orange", s=12, label="right cones")
    sc = ax.scatter(
        path_xy[:, 0], path_xy[:, 1], c=v, cmap="viridis", s=8, label="path (v-colored)"
    )
    plt.colorbar(sc, ax=ax, label="speed [m/s]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Below the plot, so it clears the path and the colorbar.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
    )
    ax.set_title("Path vs cones (speed colored)")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def _contiguous_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index ranges of contiguous True runs in ``mask``."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(int(sp[0]), int(sp[-1])) for sp in splits]


def plot_speed_profile(
    s: np.ndarray,
    v: np.ndarray,
    v_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    timed_mask: Optional[np.ndarray] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, v, label="v(s)")
    if timed_mask is not None:
        # Only the last untimed block -- where the objective stops rewarding
        # speed -- not everywhere the car happens to be slowing down.
        mask = np.asarray(timed_mask)
        untimed_blocks = _contiguous_blocks(mask < 0.5)
        if untimed_blocks:
            a, b = untimed_blocks[-1]
            ax.axvspan(s[a], s[b], color="tab:red", alpha=0.15, label="untimed (post-finish)")
    if v_max is not None:
        ax.axhline(v_max, color="r", linestyle="--", label="v_max")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("v [m/s]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Speed profile")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_gg(
    a_long: np.ndarray,
    a_lat: np.ndarray,
    mu_g: float,
    a_long_bounds: Optional[tuple[float, float]] = None,
    a_lat_bounds: Optional[tuple[float, float]] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(a_long, a_lat, s=8, alpha=0.7, label="samples")
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_x = mu_g * np.cos(theta)
    circ_y = mu_g * np.sin(theta)
    ax.plot(circ_x, circ_y, "r--", label="friction circle")
    if a_long_bounds is not None and a_lat_bounds is not None:
        ax.plot(
            [
                a_long_bounds[0],
                a_long_bounds[1],
                a_long_bounds[1],
                a_long_bounds[0],
                a_long_bounds[0],
            ],
            [a_lat_bounds[0], a_lat_bounds[0], a_lat_bounds[1], a_lat_bounds[1], a_lat_bounds[0]],
            "k--",
            alpha=0.5,
            label="a box",
        )
    ax.set_xlabel("a_long [m/s^2]")
    ax.set_ylabel("a_lat [m/s^2]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("GG diagram")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_offsets(
    s: np.ndarray,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, d, label="d(s)")
    ax.plot(s, w_left, "g--", label="w_left")
    ax.plot(s, -w_right, "r--", label="-(w_right)")
    ax.fill_between(s, -w_right, w_left, color="gray", alpha=0.1)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("d [m]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Lateral offset and bounds")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_inputs(
    s: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, a_long, label="a_long")
    ax.plot(s, a_lat, label="a_lat")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("accel [m/s^2]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Inputs vs s")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def _compute_constraint_activity(
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    v: np.ndarray,
    params: Dict,
) -> Dict[str, float]:
    """
    Compute simple constraint activity metrics from a solved trajectory.

    Fractions are in [0, 1] and indicate how often each constraint family
    is (approximately) active along the lap.
    """
    n = d.shape[0]
    if n == 0:
        return {
            "track_bounds_active": 0.0,
            "friction_circle_active": 0.0,
            "a_long_bounds_active": 0.0,
            "a_lat_bounds_active": 0.0,
            "v_bounds_active": 0.0,
        }

    # Heuristic tolerances: this drives shading, not a decision.
    tol_d = 0.05  # [m]
    tol_a = 0.5  # [m/s^2]
    tol_fc = 0.05
    tol_v = 0.5  # [m/s]

    near_left = np.isfinite(w_left) & (np.abs(d - w_left) < tol_d)
    near_right = np.isfinite(w_right) & (np.abs(d + w_right) < tol_d)
    track_bounds_active = float(np.count_nonzero(near_left | near_right) / n)

    mu = params.get("mu", 1.2)
    g_val = params.get("g", 9.81)
    mu_g = mu * g_val
    if mu_g > 0:
        fc = (a_long / mu_g) ** 2 + (a_lat / mu_g) ** 2
        friction_circle_active = float(np.count_nonzero(fc > 1.0 - tol_fc) / n)
    else:
        friction_circle_active = 0.0

    a_long_min = params.get("a_long_min", -np.inf)
    a_long_max = params.get("a_long_max", np.inf)
    a_lat_min = params.get("a_lat_min", -np.inf)
    a_lat_max = params.get("a_lat_max", np.inf)

    near_a_long_min = np.isfinite(a_long_min) & (np.abs(a_long - a_long_min) < tol_a)
    near_a_long_max = np.isfinite(a_long_max) & (np.abs(a_long - a_long_max) < tol_a)
    near_a_lat_min = np.isfinite(a_lat_min) & (np.abs(a_lat - a_lat_min) < tol_a)
    near_a_lat_max = np.isfinite(a_lat_max) & (np.abs(a_lat - a_lat_max) < tol_a)

    a_long_bounds_active = float(np.count_nonzero(near_a_long_min | near_a_long_max) / n)
    a_lat_bounds_active = float(np.count_nonzero(near_a_lat_min | near_a_lat_max) / n)

    v_min = params.get("v_min", -np.inf)
    v_max = params.get("v_max", np.inf)
    near_v_min = np.isfinite(v_min) & (np.abs(v - v_min) < tol_v)
    near_v_max = np.isfinite(v_max) & (np.abs(v - v_max) < tol_v)
    v_bounds_active = float(np.count_nonzero(near_v_min | near_v_max) / n)

    return {
        "track_bounds_active": track_bounds_active,
        "friction_circle_active": friction_circle_active,
        "a_long_bounds_active": a_long_bounds_active,
        "a_lat_bounds_active": a_lat_bounds_active,
        "v_bounds_active": v_bounds_active,
    }


ACTIVITY_ROWS = (
    ("Track bounds", "track_bounds_active"),
    ("Friction circ", "friction_circle_active"),
    ("a_long bounds", "a_long_bounds_active"),
    ("a_lat bounds", "a_lat_bounds_active"),
    ("Speed bounds", "v_bounds_active"),
)


def plot_all_panels(
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    path_xy: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    mu_g: float,
    profiling: Optional[Dict] = None,
    constraint_activity: Optional[Dict[str, float]] = None,
    timed_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    plot_path_with_speed(
        cones_left, cones_right, path_xy, v, out_path=None, show=False, fig=fig, ax=axes[0]
    )
    axes[0].set_title("Path vs cones")

    plot_speed_profile(
        s, v, v_max=None, out_path=None, show=False, fig=fig, ax=axes[1], timed_mask=timed_mask
    )
    axes[1].set_title("Speed profile")

    plot_offsets(s, d, w_left, w_right, out_path=None, show=False, fig=fig, ax=axes[2])
    axes[2].set_title("Lateral offset")

    plot_inputs(s, a_long, a_lat, out_path=None, show=False, fig=fig, ax=axes[3])
    axes[3].set_title("Inputs")

    plot_gg(a_long, a_lat, mu_g, out_path=None, show=False, fig=fig, ax=axes[4])
    axes[4].set_title("GG diagram")

    plot_profiling_panel(profiling, constraint_activity, ax=axes[5], activity_rows=ACTIVITY_ROWS)
    axes[5].set_title("Profiling & activity")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
