"""
Visualization helpers for four-wheel OCP solutions.

Main panels (3x2):
  1. Track + speed   2. Speed profile
  3. Lateral offset  4. Per-wheel Fx
  5. GG diagram      6. Profiling

Diagnostics page (4x2):
  1. Steering angle & rate  2. Per-wheel slip angles
  3. Yaw moments (TV+total) 4. Wheel force rates
  5. Friction utilization   6. Vertical loads
  7. Lateral forces         8. Lateral velocity & yaw rate
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

WHEEL_COLORS = {
    "FL": "tab:blue",
    "FR": "tab:orange",
    "RR": "tab:green",
    "RL": "tab:red",
}


ACTIVITY_ROWS = (
    ("Track bounds", "track_bounds_active"),
    ("Friction FL", "friction_fl_active"),
    ("Friction FR", "friction_fr_active"),
    ("Friction RR", "friction_rr_active"),
    ("Friction RL", "friction_rl_active"),
    ("Force bounds", "force_bounds_active"),
    ("Speed bounds", "speed_bounds_active"),
)


def _compute_constraint_activity_four_wheel(
    *,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    v: np.ndarray,
    Fx_fl: np.ndarray,
    Fx_fr: np.ndarray,
    Fx_rr: np.ndarray,
    Fx_rl: np.ndarray,
    util_fl: np.ndarray,
    util_fr: np.ndarray,
    util_rr: np.ndarray,
    util_rl: np.ndarray,
    params: Dict,
) -> Dict[str, float]:
    n = len(d)
    if n == 0:
        return {}

    tol_d = 0.05
    near_left = np.abs(d - w_left) < tol_d
    near_right = np.abs(d + w_right) < tol_d
    track_active = float(np.count_nonzero(near_left | near_right) / n)

    tol_fric = 5.0
    fric_fl = float(np.count_nonzero(util_fl > 100.0 - tol_fric) / n)
    fric_fr = float(np.count_nonzero(util_fr > 100.0 - tol_fric) / n)
    fric_rr = float(np.count_nonzero(util_rr > 100.0 - tol_fric) / n)
    fric_rl = float(np.count_nonzero(util_rl > 100.0 - tol_fric) / n)

    Fx_max = float(params.get("Fx_max", 1500.0))
    tol_f = 50.0
    all_fx = np.concatenate([Fx_fl, Fx_fr, Fx_rr, Fx_rl])
    force_active = float(np.count_nonzero(np.abs(np.abs(all_fx) - Fx_max) < tol_f) / (4 * n))

    v_min = float(params.get("v_min", 0.4))
    v_max = float(params.get("v_max", 20.0))
    tol_v = 0.5
    v_active = float(
        np.count_nonzero((np.abs(v - v_min) < tol_v) | (np.abs(v - v_max) < tol_v)) / n
    )

    return {
        "track_bounds_active": track_active,
        "friction_fl_active": fric_fl,
        "friction_fr_active": fric_fr,
        "friction_rr_active": fric_rr,
        "friction_rl_active": fric_rl,
        "force_bounds_active": force_active,
        "speed_bounds_active": v_active,
    }


def plot_all_panels_four_wheel(
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    path_xy: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    Fx_fl: np.ndarray,
    Fx_fr: np.ndarray,
    Fx_rr: np.ndarray,
    Fx_rl: np.ndarray,
    delta: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    params: Dict,
    diagnostics: Dict[str, np.ndarray],
    profiling: Optional[Dict] = None,
    input_data: Optional[Dict] = None,
    timed_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    # Tyre loads, slip angles and friction usage come from the model itself,
    # so the figure shows the solve rather than a second opinion about it.
    forces = diagnostics
    a_x = diagnostics["a_long_body"]
    a_y = diagnostics["a_lat_body"]

    constraint_activity = _compute_constraint_activity_four_wheel(
        d=d,
        w_left=w_left,
        w_right=w_right,
        v=v,
        Fx_fl=Fx_fl,
        Fx_fr=Fx_fr,
        Fx_rr=Fx_rr,
        Fx_rl=Fx_rl,
        util_fl=forces["util_fl"],
        util_fr=forces["util_fr"],
        util_rr=forces["util_rr"],
        util_rl=forces["util_rl"],
        params=params,
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    axes = axes.flatten()

    # 1. Track + speed
    plot_path_with_speed(
        cones_left, cones_right, path_xy, v, out_path=None, show=False, fig=fig, ax=axes[0]
    )
    axes[0].set_title("Path (speed colormap)")

    # 2. Speed profile
    plot_speed_profile(
        s, v, v_max=None, out_path=None, show=False, fig=fig, ax=axes[1], timed_mask=timed_mask
    )
    axes[1].set_title("Speed profile")

    # 3. Lateral offset
    plot_offsets(s, d, w_left, w_right, out_path=None, show=False, fig=fig, ax=axes[2])
    axes[2].set_title("Lateral offset")

    # 4. Per-wheel Fx
    ax4 = axes[3]
    for name, arr in [("FL", Fx_fl), ("FR", Fx_fr), ("RR", Fx_rr), ("RL", Fx_rl)]:
        ax4.plot(s, arr, color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    Fx_max = float(params.get("Fx_max", 1500.0))
    ax4.axhline(Fx_max, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax4.axhline(-Fx_max, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax4.set_xlabel("s [m]")
    ax4.set_ylabel("Fx [N]")
    ax4.legend(fontsize=8)
    ax4.grid(True, ls="--", alpha=0.4)
    ax4.set_title("Per-wheel Fx")

    # 5. GG diagram (a_lat on x, a_long on y — standard convention)
    ax5 = axes[4]
    ax5.scatter(a_y, a_x, s=6, alpha=0.6)
    ax5.set_xlabel("a_lat [m/s^2]")
    ax5.set_ylabel("a_long [m/s^2]")
    ax5.set_aspect("equal", adjustable="box")
    ax5.grid(True, ls="--", alpha=0.4)
    ax5.set_title("GG diagram")

    # 6. Profiling
    plot_profiling_panel(profiling, constraint_activity, axes[5], activity_rows=ACTIVITY_ROWS)
    axes[5].set_title("Profiling & activity")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)

        plot_diagnostics_four_wheel(
            s=s,
            v=v,
            v_lat=v_lat,
            yaw_rate=yaw_rate,
            Fx_fl=Fx_fl,
            Fx_fr=Fx_fr,
            Fx_rr=Fx_rr,
            Fx_rl=Fx_rl,
            delta=delta,
            forces=forces,
            params=params,
            input_data=input_data,
            out_path=out_path.parent / "diagnostics.png",
            show=show,
        )

    if show:
        plt.show()
    plt.close(fig)


def plot_diagnostics_four_wheel(
    *,
    s: np.ndarray,
    v: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    Fx_fl: np.ndarray,
    Fx_fr: np.ndarray,
    Fx_rr: np.ndarray,
    Fx_rl: np.ndarray,
    delta: np.ndarray,
    forces: Dict[str, np.ndarray],
    params: Dict,
    input_data: Optional[Dict] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    Mz_Fx = forces["Mz_Fx"]
    Mz_total = forces["Mz_total"]

    dFxmax = float(params.get("dFxmax", 1000.0))
    ddeltamax = float(params.get("ddeltamax", 1.3))

    # Physical rates, recovered from the normalised inputs.
    if input_data is not None:
        Fx_fl_dot = np.array(input_data.get("Fx_fl_dot_norm", np.zeros_like(s))) * dFxmax
        Fx_fr_dot = np.array(input_data.get("Fx_fr_dot_norm", np.zeros_like(s))) * dFxmax
        Fx_rr_dot = np.array(input_data.get("Fx_rr_dot_norm", np.zeros_like(s))) * dFxmax
        Fx_rl_dot = np.array(input_data.get("Fx_rl_dot_norm", np.zeros_like(s))) * dFxmax
        delta_dot = np.array(input_data.get("delta_dot_norm", np.zeros_like(s))) * ddeltamax
    else:
        Fx_fl_dot = Fx_fr_dot = Fx_rr_dot = Fx_rl_dot = np.zeros_like(s)
        delta_dot = np.zeros_like(s)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    # 1. Steering angle (rad) + steering rate (rad/s) on secondary axis
    ax1 = axes[0]
    ax1.plot(s, delta, color="tab:purple", label="delta")
    ax1.set_ylabel("delta [rad]", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.set_xlabel("s [m]")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1r = ax1.twinx()
    ax1r.plot(s, delta_dot, color="tab:gray", label="delta_dot", linewidth=0.8, alpha=0.7)
    ax1r.set_ylabel("delta_dot [rad/s]", color="tab:gray")
    ax1r.tick_params(axis="y", labelcolor="tab:gray")
    ax1.set_title("Steering angle & rate")

    # 2. Per-wheel slip angles
    for name, key in [
        ("FL", "alpha_fl"),
        ("FR", "alpha_fr"),
        ("RR", "alpha_rr"),
        ("RL", "alpha_rl"),
    ]:
        axes[1].plot(
            s, np.degrees(forces[key]), color=WHEEL_COLORS[name], label=name, linewidth=0.8
        )
    axes[1].set_ylabel("alpha [deg]")
    axes[1].set_xlabel("s [m]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", alpha=0.4)
    axes[1].set_title("Slip angles")

    # 3. Yaw moments (TV + total merged)
    axes[2].plot(s, Mz_Fx, color="tab:cyan", label="Mz (Fx only)", linewidth=0.8)
    axes[2].plot(s, Mz_total, color="tab:brown", label="Mz total", linewidth=0.8)
    axes[2].set_ylabel("Mz [Nm]")
    axes[2].set_xlabel("s [m]")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, ls="--", alpha=0.4)
    axes[2].set_title("Yaw moments")

    # 4. Force rates of change
    for name, arr in [("FL", Fx_fl_dot), ("FR", Fx_fr_dot), ("RR", Fx_rr_dot), ("RL", Fx_rl_dot)]:
        axes[3].plot(s, arr, color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[3].axhline(dFxmax, color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[3].axhline(-dFxmax, color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[3].set_ylabel("dFx/dt [N/s]")
    axes[3].set_xlabel("s [m]")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, ls="--", alpha=0.4)
    axes[3].set_title("Wheel force rates")

    # 5. Friction utilization
    for name, key in [("FL", "util_fl"), ("FR", "util_fr"), ("RR", "util_rr"), ("RL", "util_rl")]:
        axes[4].plot(s, forces[key], color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[4].axhline(100, color="gray", ls="--", lw=0.7)
    axes[4].set_ylabel("utilization [%]")
    axes[4].set_xlabel("s [m]")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, ls="--", alpha=0.4)
    axes[4].set_title("Friction utilization")

    # 6. Vertical loads
    for name, key in [("FL", "Fz_fl"), ("FR", "Fz_fr"), ("RR", "Fz_rr"), ("RL", "Fz_rl")]:
        axes[5].plot(s, forces[key], color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[5].set_ylabel("Fz [N]")
    axes[5].set_xlabel("s [m]")
    axes[5].legend(fontsize=8)
    axes[5].grid(True, ls="--", alpha=0.4)
    axes[5].set_title("Vertical loads")

    # 7. Lateral forces
    for name, key in [("FL", "Fy_fl"), ("FR", "Fy_fr"), ("RR", "Fy_rr"), ("RL", "Fy_rl")]:
        axes[6].plot(s, forces[key], color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[6].set_ylabel("Fy [N]")
    axes[6].set_xlabel("s [m]")
    axes[6].legend(fontsize=8)
    axes[6].grid(True, ls="--", alpha=0.4)
    axes[6].set_title("Lateral forces")

    # 8. v_lat and yaw_rate
    ax8 = axes[7]
    ax8.plot(s, v_lat, color="tab:blue", label="v_lat")
    ax8.set_ylabel("v_lat [m/s]", color="tab:blue")
    ax8.tick_params(axis="y", labelcolor="tab:blue")
    ax8r = ax8.twinx()
    ax8r.plot(s, yaw_rate, color="tab:orange", label="yaw_rate")
    ax8r.set_ylabel("yaw_rate [rad/s]", color="tab:orange")
    ax8r.tick_params(axis="y", labelcolor="tab:orange")
    ax8.set_xlabel("s [m]")
    ax8.grid(True, ls="--", alpha=0.4)
    ax8.set_title("Lateral velocity & yaw rate")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
