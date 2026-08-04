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

from visualization.ocp_plots import (
    plot_offsets,
    plot_path_with_speed,
    plot_speed_profile,
)


def _pacejka_lat(alpha: np.ndarray, Fz: float, B: float, C: float, Dmf: float) -> np.ndarray:
    # Simplified Magic Formula consistent with DynamicBicycleModel
    return -Fz * Dmf * np.sin(C * np.arctan(B * alpha))


def _compute_slip_and_forces(
    *,
    v: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    delta: np.ndarray,
    params: Dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (alpha_f, alpha_r, Fy_f, Fy_r) using the same formulas as the model.
    """
    m = float(params.get("m", 180.0))
    g = float(params.get("g", 9.81))
    lf = float(params.get("lf", params.get("l_f", 0.78)))
    lr = float(params.get("lr", params.get("l_r", 0.74)))

    Bf = float(params.get("Bf", params.get("B_tire_lat", 10.0)))
    Cf = float(params.get("Cf", params.get("C_tire_lat", 1.3)))
    Dmf_f = float(params.get("Dmf_f", params.get("D_tire_lat", 1.0)))
    Br = float(params.get("Br", params.get("B_tire_lat", 10.0)))
    Cr = float(params.get("Cr", params.get("C_tire_lat", 1.3)))
    Dmf_r = float(params.get("Dmf_r", params.get("D_tire_lat", 1.0)))

    v_eps = float(params.get("v_eps", 0.5))
    v_safe = np.maximum(v, v_eps)

    L = lf + lr
    Fz_f = m * g * (lr / L)
    Fz_r = m * g * (lf / L)

    alpha_f = np.arctan((v_lat + lf * yaw_rate) / v_safe) - delta
    alpha_r = np.arctan((v_lat - lr * yaw_rate) / v_safe)

    Fy_f = _pacejka_lat(alpha_f, Fz_f, Bf, Cf, Dmf_f)
    Fy_r = _pacejka_lat(alpha_r, Fz_r, Br, Cr, Dmf_r)

    return alpha_f, alpha_r, Fy_f, Fy_r


def _compute_a_lat_from_tires(
    *,
    v: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    delta: np.ndarray,
    params: Dict,
) -> np.ndarray:
    """
    Compute lateral acceleration at CG from tire forces:
      a_lat = (Fy_f*cos(delta) + Fy_r) / m
    using the same slip-angle + Pacejka structure as DynamicBicycleModel.
    """
    m = float(params.get("m", 180.0))
    _, _, Fy_f, Fy_r = _compute_slip_and_forces(
        v=v,
        v_lat=v_lat,
        yaw_rate=yaw_rate,
        delta=delta,
        params=params,
    )
    return (Fy_f * np.cos(delta) + Fy_r) / m


def plot_tire_and_yaw_diagnostics(
    *,
    s: np.ndarray,
    v: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    delta: np.ndarray,
    params: Dict,
    out_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Compact diagnostics plot (3 stacked panels):
    1) slip angles alpha_f, alpha_r
    2) lateral tire forces Fy_f, Fy_r
    3) yaw_rate
    """
    alpha_f, alpha_r, Fy_f, Fy_r = _compute_slip_and_forces(
        v=v,
        v_lat=v_lat,
        yaw_rate=yaw_rate,
        delta=delta,
        params=params,
    )

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
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    # Dual-axis plot: a_long on left axis, delta on right axis.
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

    # Combined legend.
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
):
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


def _plot_profiling_panel_dynamic(
    profiling: Optional[Dict],
    constraint_activity: Optional[Dict[str, float]],
    ax: plt.Axes,
) -> None:
    ax.axis("off")

    if profiling is None:
        ax.text(0.0, 0.5, "No profiling data available.", transform=ax.transAxes, fontsize=10, va="center")
        return

    lines = []
    N = profiling.get("N")
    ds_m = profiling.get("ds_m")
    solve_time_s = profiling.get("solve_time_s")
    iter_count = profiling.get("iter_count")
    return_status = profiling.get("return_status")
    time_per_point_ms = profiling.get("time_per_point_ms")
    time_per_iter_ms = profiling.get("time_per_iter_ms")
    lap_time_s = profiling.get("lap_time_s")
    reg_term = profiling.get("reg_term")
    reg_term_rel = profiling.get("reg_term_relative")

    lines.append("Solver profiling")
    if N is not None and ds_m is not None:
        lines.append(f"N = {N}, ds = {ds_m:.3f} m")
    if solve_time_s is not None:
        lines.append(f"Time = {solve_time_s:.3f} s")
    if time_per_point_ms is not None:
        lines.append(f"Time / point = {time_per_point_ms:.3f} ms")
    if time_per_iter_ms is not None and iter_count not in (None, 0):
        lines.append(f"Iterations = {iter_count}, time / iter = {time_per_iter_ms:.3f} ms")
    elif iter_count is not None:
        lines.append(f"Iterations = {iter_count}")
    if return_status is not None:
        lines.append(f"Status = {return_status}")

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
        def pct(key: str) -> float:
            val = constraint_activity.get(key)
            return float(val * 100.0) if val is not None else 0.0

        lines.append(f"Track bounds  ≈ {pct('track_bounds_active'):.1f}%")
        lines.append(f"GG envelope   ≈ {pct('gg_active'):.1f}%")
        lines.append(f"a_long bounds ≈ {pct('a_long_bounds_active'):.1f}%")
        lines.append(f"delta bounds  ≈ {pct('delta_bounds_active'):.1f}%")
        lines.append(f"Speed bounds  ≈ {pct('v_bounds_active'):.1f}%")
    else:
        lines.append("(no constraint activity data)")

    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, fontsize=9, va="top", family="monospace")


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
    profiling: Optional[Dict] = None,
    timed_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
):
    a_lat = _compute_a_lat_from_tires(v=v, v_lat=v_lat, yaw_rate=yaw_rate, delta=delta, params=params)
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

    plot_path_with_speed(cones_left, cones_right, path_xy, v, out_path=None, show=False, fig=fig, ax=axes[0])
    axes[0].set_title("Path vs cones")

    plot_speed_profile(s, v, v_max=None, out_path=None, show=False, fig=fig, ax=axes[1], timed_mask=timed_mask)
    axes[1].set_title("Speed profile")

    plot_offsets(s, d, w_left, w_right, out_path=None, show=False, fig=fig, ax=axes[2])
    axes[2].set_title("Lateral offset")

    plot_inputs_dynamic_bicycle(s, a_long, delta, out_path=None, show=False, fig=fig, ax=axes[3])
    axes[3].set_title("Inputs")

    plot_gg_dynamic_bicycle(a_long, a_lat, mu_g_env, out_path=None, show=False, fig=fig, ax=axes[4])
    axes[4].set_title("GG diagram")

    _plot_profiling_panel_dynamic(profiling, constraint_activity, ax=axes[5])
    axes[5].set_title("Profiling & activity")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        # Save an additional diagnostics plot next to the main panels.
        plot_tire_and_yaw_diagnostics(
            s=s,
            v=v,
            v_lat=v_lat,
            yaw_rate=yaw_rate,
            delta=delta,
            params=params,
            out_path=out_path.parent / "tire_yaw_diagnostics.png",
            show=show,
        )
    if show:
        plt.show()
    plt.close(fig)

