"""
Visualization helpers for dynamic-bicycle OCP solutions.

This module is intentionally separate from `visualization/ocp_plots.py`:
- point-mass solutions have inputs [a_long, a_lat]
- dynamic bicycle solutions have inputs [a_long, delta]

We keep the point-mass plots unchanged and provide bicycle-specific panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from fast_lto.visualization.ocp_plots import (
    plot_offsets,
    plot_path_with_speed,
    plot_speed_profile,
)
from fast_lto.visualization.summary_panel import plot_profiling_panel


def plot_tire_and_yaw_diagnostics(
    *,
    s: np.ndarray,
    v: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    delta: np.ndarray,
    params: Dict,
    diagnostics: Dict[str, np.ndarray],
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Compact diagnostics plot (3 stacked panels):
    1) slip angles alpha_f, alpha_r
    2) lateral tire forces Fy_f, Fy_r
    3) yaw_rate

    Slip angles and tire forces come from the model (VehicleModel.diagnostics),
    not from a second implementation of the Magic Formula living here.
    """
    alpha_f = diagnostics["alpha_f"]
    alpha_r = diagnostics["alpha_r"]
    Fy_f = diagnostics["Fy_f"]
    Fy_r = diagnostics["Fy_r"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(s, alpha_f, label="alpha_f")
    axes[0].plot(s, alpha_r, label="alpha_r")
    axes[0].set_ylabel("slip [rad]")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[0].set_title("Tire + yaw diagnostics")

    axes[1].plot(s, Fy_f, label="Fy_f")
    axes[1].plot(s, Fy_r, label="Fy_r")
    axes[1].set_ylabel("Fy [N]")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    axes[2].plot(s, yaw_rate, label="yaw_rate")
    axes[2].set_xlabel("s [m]")
    axes[2].set_ylabel("yaw_rate [rad/s]")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_inputs_dynamic_bicycle(
    s: np.ndarray,
    a_long: np.ndarray,
    delta: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    # a_long on the left axis, delta on the right.
    color_a = "tab:blue"
    color_d = "tab:orange"

    (ln1,) = ax.plot(s, a_long, color=color_a, label="a_long")
    ax.set_ylabel("a_long [m/s^2]", color=color_a)
    ax.tick_params(axis="y", labelcolor=color_a)

    ax2 = ax.twinx()
    (ln2,) = ax2.plot(s, delta, color=color_d, label="delta")
    ax2.set_ylabel("delta [rad]", color=color_d)
    ax2.tick_params(axis="y", labelcolor=color_d)

    ax.set_xlabel("s [m]")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(handles=[ln1, ln2], loc="best")
    ax.set_title("Inputs vs s")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_gg_dynamic_bicycle(
    a_long: np.ndarray,
    a_lat: np.ndarray,
    mu_g_env: float,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(a_long, a_lat, s=8, alpha=0.7, label="samples")
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_x = mu_g_env * np.cos(theta)
    circ_y = mu_g_env * np.sin(theta)
    ax.plot(circ_x, circ_y, "r--", label="GG envelope (gamma*mu*g)")

    ax.set_xlabel("a_long [m/s^2]")
    ax.set_ylabel("a_lat (from tires) [m/s^2]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("GG diagram (dynamic bicycle)")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def _compute_constraint_activity_dynamic_bicycle(
    *,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    v: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    delta: np.ndarray,
    params: Dict,
) -> Dict[str, float]:
    n = d.shape[0]
    if n == 0:
        return {
            "track_bounds_active": 0.0,
            "gg_active": 0.0,
            "a_long_bounds_active": 0.0,
            "delta_bounds_active": 0.0,
            "v_bounds_active": 0.0,
        }

    tol_d = 0.05
    tol_u = 0.02
    tol_gg = 0.05
    tol_v = 0.5

    near_left = np.isfinite(w_left) & (np.abs(d - w_left) < tol_d)
    near_right = np.isfinite(w_right) & (np.abs(d + w_right) < tol_d)
    track_bounds_active = float(np.count_nonzero(near_left | near_right) / n)

    mu = float(params.get("mu", 1.2))
    g_val = float(params.get("g", 9.81))
    gamma = float(params.get("gamma_ellipse", 1.0))
    mu_g_env = gamma * mu * g_val
    if mu_g_env > 0:
        gg = (a_long / mu_g_env) ** 2 + (a_lat / mu_g_env) ** 2
        gg_active = float(np.count_nonzero(gg > 1.0 - tol_gg) / n)
    else:
        gg_active = 0.0

    a_long_min = float(params.get("a_long_min", -np.inf))
    a_long_max = float(params.get("a_long_max", np.inf))
    delta_max = float(params.get("delta_max", np.inf))

    near_a_long_min = np.isfinite(a_long_min) & (np.abs(a_long - a_long_min) < 0.5)
    near_a_long_max = np.isfinite(a_long_max) & (np.abs(a_long - a_long_max) < 0.5)
    a_long_bounds_active = float(np.count_nonzero(near_a_long_min | near_a_long_max) / n)

    near_delta = np.isfinite(delta_max) & (np.abs(np.abs(delta) - delta_max) < tol_u)
    delta_bounds_active = float(np.count_nonzero(near_delta) / n)

    v_min = float(params.get("v_min", -np.inf))
    v_max = float(params.get("v_max", np.inf))
    near_v_min = np.isfinite(v_min) & (np.abs(v - v_min) < tol_v)
    near_v_max = np.isfinite(v_max) & (np.abs(v - v_max) < tol_v)
    v_bounds_active = float(np.count_nonzero(near_v_min | near_v_max) / n)

    return {
        "track_bounds_active": track_bounds_active,
        "gg_active": gg_active,
        "a_long_bounds_active": a_long_bounds_active,
        "delta_bounds_active": delta_bounds_active,
        "v_bounds_active": v_bounds_active,
    }


ACTIVITY_ROWS = (
    ("Track bounds", "track_bounds_active"),
    ("GG envelope", "gg_active"),
    ("a_long bounds", "a_long_bounds_active"),
    ("delta bounds", "delta_bounds_active"),
    ("Speed bounds", "v_bounds_active"),
)


def plot_all_panels_dynamic_bicycle(
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    path_xy: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    a_long: np.ndarray,
    delta: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    params: Dict,
    diagnostics: Dict[str, np.ndarray],
    profiling: Optional[Dict] = None,
    timed_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    a_lat = diagnostics["a_lat_tires"]
    mu = float(params.get("mu", 1.2))
    g_val = float(params.get("g", 9.81))
    gamma = float(params.get("gamma_ellipse", 1.0))
    mu_g_env = gamma * mu * g_val

    constraint_activity = _compute_constraint_activity_dynamic_bicycle(
        d=d,
        w_left=w_left,
        w_right=w_right,
        v=v,
        a_long=a_long,
        a_lat=a_lat,
        delta=delta,
        params=params,
    )

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

    plot_inputs_dynamic_bicycle(s, a_long, delta, out_path=None, show=False, fig=fig, ax=axes[3])
    axes[3].set_title("Inputs")

    plot_gg_dynamic_bicycle(a_long, a_lat, mu_g_env, out_path=None, show=False, fig=fig, ax=axes[4])
    axes[4].set_title("GG diagram")

    plot_profiling_panel(profiling, constraint_activity, ax=axes[5], activity_rows=ACTIVITY_ROWS)
    axes[5].set_title("Profiling & activity")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        # A second figure, next to the main panels.
        plot_tire_and_yaw_diagnostics(
            s=s,
            v=v,
            v_lat=v_lat,
            yaw_rate=yaw_rate,
            delta=delta,
            params=params,
            diagnostics=diagnostics,
            out_path=out_path.parent / "tire_yaw_diagnostics.png",
            show=show,
        )
    if show:
        plt.show()
    plt.close(fig)
